// sidepanel.js — the extension's logic.
//
// One chat per tab. The side panel is a single window, but we keep a
// separate "session" (page info + chat messages) for every tab and swap
// them in and out as the user switches tabs.
//
// Flow:
//   1. Panel opens → read the active tab → index it → show its chat.
//   2. User switches tab → save current chat, restore that tab's chat
//      (indexing only happens the first time a page is seen).
//   3. User asks → re-check the page hasn't changed → call /ask for that
//      tab's URL → stream the answer and render [n] citations.

const DEFAULT_BACKEND_URL = "http://localhost:8000";

// Settings live in chrome.storage.local so they survive reloads.
const settings = { backendUrl: DEFAULT_BACKEND_URL };

const pageTitleElement = document.getElementById("page-title");
const statusElement = document.getElementById("status");
const chatElement = document.getElementById("chat");
const questionInput = document.getElementById("question-input");
const askButton = document.getElementById("ask-button");
const settingsButton = document.getElementById("settings-button");
const settingsPanel = document.getElementById("settings");
const backendUrlInput = document.getElementById("backend-url");
const saveSettingsButton = document.getElementById("save-settings");

// ---------------------------------------------------------------------------
// Sessions: one per tab
// ---------------------------------------------------------------------------
//   sessions[tabId] = {
//     url, title, status, ready,
//     contentHash:  fingerprint of the page text at index time (freshness),
//     chatNodes:    the message elements for this tab's chat,
//     sources:      chunks used by the most recent answer (for citations),
//   }
const sessions = {};
let activeTabId = null;

function createSession() {
  const welcome = document.createElement("div");
  welcome.className = "message assistant";
  welcome.textContent = "Ask me anything about this page.";
  return { url: null, title: "", status: "", ready: false, contentHash: null,
           chatNodes: [welcome], sources: [] };
}

function saveCurrentChat() {
  const session = sessions[activeTabId];
  if (session) session.chatNodes = Array.from(chatElement.children);
}

function showSession(session) {
  chatElement.innerHTML = "";
  session.chatNodes.forEach((node) => chatElement.appendChild(node));
  pageTitleElement.textContent = session.title || session.url || "Loading page…";
  setStatus(session.status, !session.ready && session.status !== "");
  setInputEnabled(session.ready);
  scrollChatToBottom();
}

// ---------------------------------------------------------------------------
// Step 1: extract structured page content
// ---------------------------------------------------------------------------

// This function is NOT run inside the side panel. Chrome serialises it and
// runs it inside the web page (see chrome.scripting.executeScript below).
// That is why it can access the page's `document` but nothing from this file.
//
// It walks the page and emits *blocks* — headings, paragraphs, list items,
// code, table cells, quotes — in reading order. The backend groups those
// under their headings, so every answer can say which section it came from.
function extractPageContent() {
  const NOISE = "script, style, noscript, nav, header, footer, aside, iframe, form, " +
                "[role=navigation], [role=banner], [role=contentinfo], [aria-hidden=true]";
  const HEADINGS = { H1: 1, H2: 2, H3: 3, H4: 4, H5: 5, H6: 6 };
  const LEAF = {
    P: "paragraph", LI: "list", PRE: "code", BLOCKQUOTE: "quote",
    TD: "table", TH: "table", DT: "list", DD: "list", FIGCAPTION: "paragraph",
  };
  const CONTAINERS = new Set(["DIV", "SECTION", "ARTICLE", "MAIN", "BODY", "UL", "OL",
                              "DL", "TABLE", "TBODY", "THEAD", "TR", "SPAN", "DETAILS",
                              "SUMMARY", "FIGURE", "LABEL", "A", "B", "I", "STRONG", "EM"]);

  const bodyCopy = document.body.cloneNode(true);
  bodyCopy.querySelectorAll(NOISE).forEach((element) => element.remove());
  const root = bodyCopy.querySelector("article") || bodyCopy.querySelector("main") || bodyCopy;

  const blocks = [];
  const clean = (text) => text.replace(/\s+/g, " ").trim();
  const push = (type, node, level) => {
    const text = type === "code" ? node.textContent.trim() : clean(node.textContent);
    if (text) blocks.push(level ? { type, level, text } : { type, text });
  };

  function walk(node) {
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const tag = node.tagName;
    if (HEADINGS[tag]) return push("heading", node, HEADINGS[tag]);
    if (LEAF[tag]) {
      // A leaf that contains other leaves (e.g. <li><p>…</p></li>) — recurse
      // so we don't emit the same text twice.
      const hasInnerBlocks = Array.from(node.children).some(
        (child) => LEAF[child.tagName] || HEADINGS[child.tagName]);
      if (!hasInnerBlocks) return push(LEAF[tag], node);
    }
    if (!CONTAINERS.has(tag) && !LEAF[tag]) {
      // Unknown element with only inline content → treat as a paragraph
      const hasBlockChild = Array.from(node.children).some(
        (child) => CONTAINERS.has(child.tagName) || LEAF[child.tagName] || HEADINGS[child.tagName]);
      if (!hasBlockChild) return push("paragraph", node);
    }
    // Text sitting directly in a container (common in bare <div> layouts)
    const loose = Array.from(node.childNodes)
      .filter((child) => child.nodeType === Node.TEXT_NODE)
      .map((child) => child.textContent).join(" ");
    if (clean(loose).split(" ").length > 8) blocks.push({ type: "paragraph", text: clean(loose) });
    Array.from(node.children).forEach(walk);
  }
  walk(root);

  const text = clean(root.textContent);   // fallback for the backend
  return { url: location.href, title: document.title, blocks, text };
}

async function readTab(tab) {
  if (!tab.url || !tab.url.startsWith("http")) {
    throw new Error("Open a normal web page first (not a Chrome settings page).");
  }
  const [result] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: extractPageContent,
  });
  return result.result; // { url, title, blocks, text }
}

// Fingerprint of the page text, computed in the browser. Cheap, and lets us
// notice that a page changed without sending it to the backend again.
async function hashText(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------------------
// Step 2: talk to the backend
// ---------------------------------------------------------------------------

function apiHeaders() {
  const headers = { "Content-Type": "application/json" };
  return headers;
}

async function apiPost(path, body) {
  const response = await fetch(`${settings.backendUrl}${path}`, {
    method: "POST", headers: apiHeaders(), body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "Request failed.";
    try { detail = (await response.json()).detail || detail; } catch (_) { /* not JSON */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response;
}

async function indexPage(page) {
  const response = await apiPost("/index", page);
  return response.json(); // { cached, refreshed, num_chunks, num_sections, ... }
}

// Read + index one tab and fill in its session.
async function prepareTab(tab, session = sessions[tab.id]) {
  try {
    const page = await readTab(tab);
    session.url = page.url;
    session.title = page.title;
    session.status = "Indexing page…";
    if (tab.id === activeTabId) showSession(session);

    const summary = await indexPage(page);
    session.contentHash = await hashText(page.text);
    const note = summary.refreshed ? " (page changed — re-indexed)"
               : summary.cached ? " (cached)" : "";
    session.status = `Ready · ${summary.num_chunks} chunks` +
      (summary.num_sections ? ` · ${summary.num_sections} sections` : "") + note;
    session.ready = true;
  } catch (error) {
    session.status = friendlyError(error);
    session.ready = false;
  }
  if (tab.id === activeTabId) showSession(session);
}

function friendlyError(error) {
  if (error.message.includes("Failed to fetch")) {
    return `Backend not reachable at ${settings.backendUrl}. Is it running? (⚙ to change)`;
  }
  return error.message;
}

// Freshness check: has the page changed since we indexed it? If so,
// re-index before asking so the answer reflects what's on screen now.
async function ensureFresh(tab, session) {
  const page = await readTab(tab);
  const hash = await hashText(page.text);
  if (hash === session.contentHash && page.url === session.url) return;
  session.status = "Page changed — re-indexing…";
  setStatus(session.status);
  const summary = await indexPage(page);
  session.url = page.url;
  session.title = page.title;
  session.contentHash = hash;
  session.status = `Ready · ${summary.num_chunks} chunks (re-indexed)`;
  setStatus(session.status);
}

// ---------------------------------------------------------------------------
// Step 3: ask a question and stream the answer
// ---------------------------------------------------------------------------

async function askQuestion(question, session) {
  const response = await apiPost("/ask", { url: session.url, question });

  const answerElement = addMessage("", "assistant");
  answerElement.classList.add("streaming");
  let answerText = "";

  // The backend sends one JSON object per line (NDJSON).
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let leftover = "";

  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (error) {
      throw new Error(`Connection to ${settings.backendUrl} dropped while streaming the answer.`);
    }
    const { value, done } = chunk;
    if (done) break;
    leftover += decoder.decode(value, { stream: true });
    const lines = leftover.split("\n");
    leftover = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const message = JSON.parse(line);
      if (message.type === "token") {
        answerText += message.text;
        answerElement.textContent = answerText;
        scrollChatToBottom();
      } else if (message.type === "sources") {
        session.sources = message.chunks;
        showSources(message.chunks, message.timings, message.mode);
      } else if (message.type === "error") {
        answerElement.classList.add("error");
        answerElement.textContent = message.message;
      }
    }
  }
  answerElement.classList.remove("streaming");
  renderCitations(answerElement, answerText, session.sources);
}

// ---------------------------------------------------------------------------
// Citations: [n] in the answer → clickable chip → highlight on the page
// ---------------------------------------------------------------------------

function renderCitations(element, text, sources) {
  element.textContent = "";
  const parts = text.split(/(\[\d+(?:\]\[\d+)*\])/g);   // "[1]" or "[2][3]"
  for (const part of parts) {
    if (!/^\[\d+/.test(part)) { element.appendChild(document.createTextNode(part)); continue; }
    for (const number of part.match(/\d+/g)) {
      const index = Number(number) - 1;
      const chip = document.createElement("button");
      chip.className = "cite";
      chip.textContent = number;
      if (sources[index]) {
        chip.title = sources[index].heading_path?.join(" › ") || "Show on page";
        chip.addEventListener("click", () => highlightOnPage(sources[index].text));
      } else {
        chip.disabled = true;
      }
      element.appendChild(chip);
    }
  }
}

// Runs inside the web page: find `snippet`, scroll to it, highlight it.
function highlightSnippetInPage(snippet) {
  const style = document.getElementById("askpage-highlight-style") || document.createElement("style");
  style.id = "askpage-highlight-style";
  style.textContent = "::highlight(askpage) { background: #ffe9a8; color: inherit; }";
  document.head.appendChild(style);
  if (CSS.highlights) CSS.highlights.delete("askpage");

  // Search in progressively shorter prefixes: layout can split a chunk
  // across elements, but its first sentence is usually contiguous.
  const words = snippet.split(/\s+/);
  for (const length of [40, 25, 15, 8]) {
    const needle = words.slice(0, length).join(" ");
    window.getSelection().removeAllRanges();
    if (window.find(needle, false, false, true, false, false, false)) {
      const selection = window.getSelection();
      if (selection.rangeCount) {
        const range = selection.getRangeAt(0).cloneRange();
        selection.removeAllRanges();
        if (CSS.highlights) CSS.highlights.set("askpage", new Highlight(range));
        range.startContainer.parentElement.scrollIntoView({ behavior: "smooth", block: "center" });
      }
      return true;
    }
  }
  return false;
}

async function highlightOnPage(text) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId: activeTabId }, func: highlightSnippetInPage, args: [text],
    });
  } catch (error) {
    console.warn("Could not highlight on page:", error);
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function addMessage(text, role) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  chatElement.appendChild(element);
  scrollChatToBottom();
  return element;
}

function showSources(chunks, timings, mode) {
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  const ms = (s) => `${Math.round((s || 0) * 1000)} ms`;
  summary.textContent = `${chunks.length} passages · search ${ms(timings?.retrieval)}` +
    (timings?.rerank ? ` · rerank ${ms(timings.rerank)}` : "") + (mode ? ` · ${mode}` : "");
  details.appendChild(summary);
  chunks.forEach((chunk, i) => {
    const chunkElement = document.createElement("div");
    chunkElement.className = "chunk";
    const where = document.createElement("div");
    where.className = "where";
    where.textContent = `[${i + 1}] ${chunk.heading_path?.length ? chunk.heading_path.join(" › ") : "Top of page"}` +
      (chunk.rerank_score != null ? ` · score ${chunk.rerank_score.toFixed(1)}` : "");
    const body = document.createElement("div");
    body.textContent = chunk.text.length > 220 ? chunk.text.slice(0, 220) + "…" : chunk.text;
    chunkElement.append(where, body);
    chunkElement.title = "Click to show on the page";
    chunkElement.addEventListener("click", () => highlightOnPage(chunk.text));
    details.appendChild(chunkElement);
  });
  chatElement.appendChild(details);
}

function setStatus(text, isError = false) {
  statusElement.textContent = text;
  statusElement.classList.toggle("error", isError);
}

function setInputEnabled(enabled) {
  questionInput.disabled = !enabled;
  askButton.disabled = !enabled;
}

function scrollChatToBottom() {
  chatElement.scrollTop = chatElement.scrollHeight;
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

async function handleAsk() {
  const question = questionInput.value.trim();
  const session = sessions[activeTabId];
  if (!question || !session || !session.ready) return;

  addMessage(question, "user");
  questionInput.value = "";
  setInputEnabled(false);

  try {
    const tab = await chrome.tabs.get(activeTabId);
    await ensureFresh(tab, session);
    try {
      await askQuestion(question, session);
    } catch (error) {
      // Backend restarted or index expired → re-index once and retry
      if (error.status !== 404) throw error;
      await prepareTab(tab, session);
      await askQuestion(question, session);
    }
  } catch (error) {
    addMessage(friendlyError(error), "error");
  } finally {
    if (sessions[activeTabId] === session) {
      setInputEnabled(true);
      questionInput.focus();
    }
  }
}

askButton.addEventListener("click", handleAsk);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleAsk();
  }
});

// Settings panel
settingsButton.addEventListener("click", () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  backendUrlInput.value = settings.backendUrl;
});
saveSettingsButton.addEventListener("click", async () => {
  settings.backendUrl = (backendUrlInput.value.trim() || DEFAULT_BACKEND_URL).replace(/\/+$/, "");
  await chrome.storage.local.set({ settings });
  settingsPanel.hidden = true;
  // Backend may have changed → forget all sessions and start over
  Object.keys(sessions).forEach((id) => delete sessions[id]);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab) switchToTab(tab);
});

async function switchToTab(tab) {
  saveCurrentChat();
  activeTabId = tab.id;
  if (!sessions[tab.id]) {
    sessions[tab.id] = createSession();
    showSession(sessions[tab.id]);
    await prepareTab(tab);
  } else {
    showSession(sessions[tab.id]);
  }
}

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId);
  switchToTab(tab);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const session = sessions[tabId];
  const urlChanged = session && session.url && tab.url !== session.url;
  if (changeInfo.status === "complete" && (urlChanged || !session)) {
    delete sessions[tabId];
    if (tabId === activeTabId) switchToTab(tab);
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  delete sessions[tabId];
});

async function initialise() {
  const stored = await chrome.storage.local.get("settings");
  if (stored.settings) Object.assign(settings, stored.settings);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab) switchToTab(tab);
}

initialise();

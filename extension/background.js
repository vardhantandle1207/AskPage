// background.js — runs in the background as a "service worker".
//
// Its only job: make clicking the extension icon open the side panel.
// Everything else (extracting text, talking to the backend) happens in
// sidepanel.js.

chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error) => console.error("Could not set side panel behavior:", error));

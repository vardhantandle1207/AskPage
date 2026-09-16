"""
ingest.py — smart data ingestion: turn a raw web page into clean, structured
chunks that retrieval can work with.

Pipeline (in order):
    1. clean    — normalise unicode/whitespace, drop boilerplate lines.
    2. parse    — group the page's blocks (headings, paragraphs, lists, code,
                  table cells) into *sections*, each with a heading path like
                  "Considerations for teachers > Open-ended".
    3. split    — cut each section into chunks at sentence boundaries, with a
                  little overlap so a sentence on a boundary keeps its context.
    4. dedupe   — drop chunks that are exact repeats (cookie banners, repeated
                  nav text, "share this" blocks).

The extension sends either:
    - `blocks`: [{"type": "heading"|"paragraph"|"list"|"code"|"table"|"quote",
                  "level": 1-6 (headings only), "text": "..."}]     (preferred)
    - `text`:   one flat string                                     (fallback)

Every chunk we produce is a dict:
    {
      "id":           "c12",
      "text":         "...",
      "section":      "Open-ended",                  # nearest heading
      "heading_path": ["Considerations", "Open-ended"],
      "block_types":  ["paragraph", "list"],
      "position":     12,                            # order on the page
    }
That metadata is what makes citations possible later on.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CHUNK_SIZE_WORDS = 220      # target size; a chunk may run a bit over to
                            # finish its last sentence
CHUNK_OVERLAP_SENTENCES = 2 # sentences repeated from the previous chunk
MIN_CHUNK_WORDS = 15        # merge anything smaller into its neighbour

# Lines that are almost certainly page furniture rather than content.
BOILERPLATE_PATTERNS = [
    r"^(accept|reject|manage)( all)? cookies?$",
    r"^cookie (settings|policy|preferences)$",
    r"^(share|tweet|print|copy link|subscribe|sign up|log ?in|sign ?in)$",
    r"^(advertisement|sponsored|skip to (main )?content)$",
    r"^(read more|show more|see more|learn more)$",
    r"^\W*$",                     # punctuation / symbols only
]
_BOILERPLATE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Rough sentence splitter: split after . ! ? (plus closing quotes/brackets)
# when followed by whitespace and an uppercase letter, digit or quote.
# Good enough for web prose; we deliberately avoid a heavy NLP dependency.
# (Two alternations because Python look-behinds must be fixed width.)
_SENTENCE_END = re.compile(
    r'(?:(?<=[.!?])|(?<=[.!?]["\')\]]))\s+(?=["\'(\[]?[A-Z0-9])'
)


# ---------------------------------------------------------------------------
# 1. Clean
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise a piece of text so equal content compares equal."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", "").replace("﻿", "")   # zero-width junk
    text = re.sub(r"[ \t\r\f\v]+", " ", text)                  # runs of spaces
    text = re.sub(r"\n\s*\n+", "\n", text)                      # blank lines
    return text.strip()


def is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE.match(text.strip()))


# ---------------------------------------------------------------------------
# 2. Parse blocks into sections
# ---------------------------------------------------------------------------

def blocks_from_text(text: str) -> list[dict]:
    """Fallback for a flat string: every paragraph becomes a block."""
    paragraphs = [p for p in clean_text(text).split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        # A single blob (old extension, or textContent squashed to one line):
        # break on sentences so the splitter has something to work with.
        return [{"type": "paragraph", "text": clean_text(text)}]
    return [{"type": "paragraph", "text": p} for p in paragraphs]


def build_sections(blocks: list[dict]) -> list[dict]:
    """
    Walk the blocks in page order and attach every non-heading block to the
    heading stack that is open at that point.

    Returns: [{"heading_path": [...], "blocks": [block, ...]}, ...]
    """
    heading_stack: list[tuple[int, str]] = []   # (level, text), outermost first
    sections: list[dict] = []
    current: dict | None = None

    def open_section():
        nonlocal current
        current = {"heading_path": [h for _, h in heading_stack], "blocks": []}
        sections.append(current)

    for block in blocks:
        text = clean_text(block.get("text", ""))
        if not text or is_boilerplate(text):
            continue

        if block.get("type") == "heading":
            level = int(block.get("level") or 6)
            # Close any headings at the same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text[:120]))
            open_section()
            continue

        if current is None:
            open_section()
        current["blocks"].append({"type": block.get("type", "paragraph"), "text": text})

    return [s for s in sections if s["blocks"]]


# ---------------------------------------------------------------------------
# 3. Split each section into chunks
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_section(section: dict) -> list[dict]:
    """
    Pack a section's sentences into chunks of ~CHUNK_SIZE_WORDS.

    Sentence units keep meaning intact; overlapping the last couple of
    sentences into the next chunk keeps cross-boundary context. Code and
    table blocks are never split mid-way — they become their own unit.
    """
    # Build a list of (sentence, block_type) units in order.
    units: list[tuple[str, str]] = []
    for block in section["blocks"]:
        if block["type"] in ("code", "table"):
            units.append((block["text"], block["type"]))
        else:
            for sentence in split_sentences(block["text"]):
                units.append((sentence, block["type"]))

    chunks: list[dict] = []
    buffer: list[tuple[str, str]] = []   # units waiting to become a chunk
    buffer_words = 0
    fresh_units = 0                       # units in buffer that are NOT overlap

    def emit():
        chunks.append({
            "text": " ".join(u for u, _ in buffer),
            "block_types": sorted({t for _, t in buffer}),
        })

    for unit, block_type in units:
        words = _word_count(unit)
        if fresh_units and buffer_words + words > CHUNK_SIZE_WORDS:
            emit()
            # Overlap: carry the tail of this chunk into the next one
            buffer = buffer[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            buffer_words = sum(_word_count(u) for u, _ in buffer)
            fresh_units = 0
        buffer.append((unit, block_type))
        buffer_words += words
        fresh_units += 1

    if fresh_units:          # don't emit a chunk made only of overlap
        emit()

    # Merge a tiny trailing chunk into the previous one
    if len(chunks) >= 2 and _word_count(chunks[-1]["text"]) < MIN_CHUNK_WORDS:
        last = chunks.pop()
        chunks[-1]["text"] += " " + last["text"]
        chunks[-1]["block_types"] = sorted(set(chunks[-1]["block_types"]) | set(last["block_types"]))

    for chunk in chunks:
        chunk["heading_path"] = section["heading_path"]
        chunk["section"] = section["heading_path"][-1] if section["heading_path"] else ""
    return chunks


# ---------------------------------------------------------------------------
# 4. Dedupe + the one function main.py calls
# ---------------------------------------------------------------------------

def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def ingest(blocks: list[dict] | None, text: str | None) -> list[dict]:
    """
    Full pipeline. Prefer `blocks`; fall back to `text`.
    Returns the list of chunk dicts described at the top of this file.
    """
    if not blocks:
        blocks = blocks_from_text(text or "")

    chunks: list[dict] = []
    seen: set[str] = set()
    for section in build_sections(blocks):
        for chunk in chunk_section(section):
            key = _dedupe_key(chunk["text"])
            if key in seen or len(key) < 10:
                continue
            seen.add(key)
            chunk["id"] = f"c{len(chunks)}"
            chunk["position"] = len(chunks)
            chunks.append(chunk)
    return chunks

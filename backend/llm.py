"""
llm.py — the generation part of the RAG pipeline.

Responsibilities:
    1. Build a prompt that contains the retrieved chunks + the question.
    2. Send it to an LLM — either Groq (cloud, fast) or Ollama (local, free).
    3. Return the answer as a stream (token by token) or all at once.

Which provider is used is decided by environment variables, so the same
code runs with a local model (Ollama) or a hosted one (Groq):

    LLM_PROVIDER = "ollama" or  "groq"          (default: ollama)
    LLM_MODEL    = model name for that provider (sensible defaults below)
    GROQ_API_KEY = required when LLM_PROVIDER=groq
"""

import os

import groq
import ollama

# ---------------------------------------------------------------------------
# Provider configuration (read once at startup)
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",     # free tier, ~1000 tokens/sec (llama-3.1-8b-instant was retired Aug 2026)
    "ollama": "llama3.2",             # run `ollama pull llama3.2` first (3B)
}

if LLM_PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER must be 'groq' or 'ollama', got '{LLM_PROVIDER}'")

MODEL_NAME = os.getenv("LLM_MODEL", DEFAULT_MODELS[LLM_PROVIDER])

# Answer length budget. Ollama defaults to 128 new tokens, which truncates a
# thorough answer mid-sentence, so we set it explicitly for both providers.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "800"))

# The Groq client reads GROQ_API_KEY from the environment automatically.
# We only create it when needed so Ollama-only users don't need a key.
groq_client = groq.Groq() if LLM_PROVIDER == "groq" else None

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The system prompt sets the rules. The most important rule: only answer
# from the given context. This is our simple guard against hallucination.
# The second: cite sources with [n] so the UI can link each claim back to
# the exact passage (and section) on the page it came from.
SYSTEM_PROMPT = (
    "You are AskPage, an assistant that answers questions about a web page.\n"
    "Rules:\n"
    "1. Answer ONLY using the SOURCES below.\n"
    "2. If the answer is not in the sources, reply exactly: "
    "\"I couldn't find that on this page.\"\n"
    "3. After each sentence or claim, cite the source(s) it came from using "
    "square brackets, e.g. [1] or [2][3]. Only cite numbers that exist.\n"
    "4. Be thorough. Give the direct answer first, then explain it using the "
    "detail the sources actually provide: the exact names, commands, numbers, "
    "options, defaults, caveats and exceptions that appear in them. Aim for "
    "roughly 4-8 sentences for a normal question.\n"
    "5. When the sources cover several distinct points, list them as short "
    "bullets ('- ' at the start of a line), one point per bullet, each cited.\n"
    "6. If the sources mention a related restriction, exception or gotcha, "
    "include it even if the question did not ask for it.\n"
    "7. Never pad. Every sentence must add a fact from the sources -- do not "
    "restate the question, do not add filler, do not speculate beyond them.\n"
    "8. Use plain language. Do not mention the words 'context' or 'sources'."
)


def build_messages(question: str, chunks: list[dict], page_title: str) -> list[dict]:
    """
    Put the retrieved chunks and the question into the chat format both
    providers expect. Each chunk is numbered and labelled with the section
    it came from, so the model can cite [n] and mention where on the page.
    """
    source_parts = []
    for i, chunk in enumerate(chunks, start=1):
        section = " > ".join(chunk.get("heading_path") or []) or "(top of page)"
        source_parts.append(f"[{i}] (section: {section})\n{chunk['text']}")
    sources_text = "\n\n".join(source_parts) or "(no relevant passages found)"

    user_content = (
        f"PAGE TITLE: {page_title}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        f"QUESTION: {question}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Provider-specific streaming (each yields small strings)
# ---------------------------------------------------------------------------

def _stream_groq(messages: list[dict]):
    """Stream from Groq. Groq's API is OpenAI-compatible."""
    try:
        stream = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.2,          # low = more factual, less creative
            max_tokens=MAX_ANSWER_TOKENS,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:               # some chunks are empty (role markers etc.)
                yield text
    except groq.AuthenticationError:
        raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY.")
    except groq.RateLimitError:
        raise RuntimeError("Groq rate limit hit. Wait a minute and try again.")
    except groq.APIConnectionError:
        raise RuntimeError("Cannot reach Groq. Check the server's internet access.")
    except groq.NotFoundError:
        raise RuntimeError(
            f"Groq does not know the model '{MODEL_NAME}'. It may have been "
            "retired — set LLM_MODEL in .env to a current model."
        )
    except groq.APIStatusError as error:
        # Any other HTTP error from Groq (400, 403, 413, 5xx ...)
        raise RuntimeError(f"Groq error {error.status_code}: {error.message}")


def _stream_ollama(messages: list[dict]):
    """Stream from a local Ollama server."""
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            options={"num_predict": MAX_ANSWER_TOKENS, "temperature": 0.2},
        )
        for part in stream:
            yield part["message"]["content"]
    except ollama.ResponseError as error:
        raise RuntimeError(
            f"Ollama error: {error.error}. Did you run `ollama pull {MODEL_NAME}`?"
        )
    except ConnectionError:
        raise RuntimeError("Cannot reach Ollama. Is `ollama serve` running?")


def stream_chat(messages: list[dict]):
    """Pick the provider and stream its output. Everything below uses this."""
    if LLM_PROVIDER == "groq":
        yield from _stream_groq(messages)
    else:
        yield from _stream_ollama(messages)


def complete_chat(messages: list[dict]) -> str:
    """Non-streaming helper: run the chat and return the full text."""
    return "".join(stream_chat(messages))


# ---------------------------------------------------------------------------
# Public functions used by main.py and evaluate.py
# ---------------------------------------------------------------------------

def stream_answer(question: str, chunks: list[dict], page_title: str):
    """
    Generator: yields the answer one small piece at a time.
    main.py forwards each piece to the browser so the user sees
    text appear immediately instead of waiting for the whole answer.
    """
    messages = build_messages(question, chunks, page_title)
    yield from stream_chat(messages)


def generate_answer(question: str, chunks: list[dict], page_title: str) -> str:
    """Non-streaming version. Used by the evaluation script."""
    return complete_chat(build_messages(question, chunks, page_title))

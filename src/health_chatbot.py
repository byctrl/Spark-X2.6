"""Minimal health-information chatbot built on Spark-X2.6-4B.

Smallest runnable demo for the AI-for-Good challenge: an inclusive,
plain-language patient-education assistant that never diagnoses or
prescribes and always points the user to a qualified clinician.

It talks to a local OpenAI-compatible server (SGLang / vLLM / Ollama /
Spark-MLX-LLM), so the demo runs offline and needs no extra dependency.

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py "How do I prepare for a clinic visit?"

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_MODEL     default XHToken/Spark-X2.6-4B
    SPARK_API_KEY   default EMPTY
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a health-education assistant. Explain health topics in simple, "
    "respectful language that a person without medical training can follow. "
    "You are not a doctor: never give a diagnosis, never name a prescription "
    "medicine or dose, and never tell the user to stop or change a treatment. "
    "If the question suggests an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), say "
    "clearly that the person should contact local emergency services now. "
    "Otherwise, give the general information that is useful, mention what a "
    "clinician would need to know, and recommend seeing a qualified health "
    "professional. Keep answers short, use plain words, and say when you are "
    "unsure instead of guessing."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a mild "
    "fever. I don't have insurance and I'm not sure whether this is serious. "
    "What should I know, and what should I ask a doctor?"
)


def build_messages(question: str, history: list[dict] | None = None) -> list[dict]:
    """Build the chat messages sent to the model."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    question: str,
    history: list[dict] | None = None,
    model: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict:
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data: dict) -> str:
    """Extract the answer text and reject empty or malformed responses."""
    try:
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"unexpected response from model server: {data!r}") from exc
    if not text:
        raise RuntimeError("model returned an empty answer; check the thinking/output budget")
    return text


def _post(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(
    question: str,
    history: list[dict] | None = None,
    base_url: str | None = None,
    model: str = MODEL_ID,
    api_key: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Send one question to Spark-X2.6-4B and return the answer text."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("SPARK_MODEL", model)
    api_key = api_key or os.environ.get("SPARK_API_KEY", "EMPTY")
    payload = build_payload(question, history, model=model, max_new_tokens=max_new_tokens)
    return parse_response(_post(f"{base_url}/chat/completions", payload, api_key, timeout))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print(f"[{MODEL_ID}] question: {question}\n")
    try:
        print(ask(question))
    except urllib.error.URLError as exc:
        print(f"Could not reach the Spark-X2.6-4B server: {exc}", file=sys.stderr)
        print("Start an OpenAI-compatible server and set SPARK_BASE_URL.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

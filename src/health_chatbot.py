"""Minimal health-information chatbot demo built on Spark-X2.6-4B.

The model is served by any OpenAI-compatible runtime (SGLang, vLLM,
Ollama, Spark-MLX-LLM, ...). Nothing is loaded at import time: the HTTP
request only happens when ``chat()`` / ``main()`` is invoked, so this
module stays importable in a bare Python environment.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

try:  # pragma: no cover - typing helper only
    from typing import Dict, List, Optional
    History = List[Dict[str, str]]
    Payload = Dict[str, object]
except Exception:  # pragma: no cover
    History = list
    Payload = dict


MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for community health workers. "
    "Explain general health information in plain and simple language. "
    "You must NOT diagnose, prescribe, recommend a treatment or suggest a dose. "
    "Always name the symptoms that need urgent in-person care and direct the "
    "user to a qualified clinician. "
    "If the request is outside general health education, say so briefly. "
    "Answer in the same language the user writes in."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a cough "
    "and a mild fever in the evenings. I have no insurance and I am not sure "
    "whether I should wait or see someone. What should I know?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please consult a qualified clinician about your own situation."
)


def build_messages(question: str, history: Optional[History] = None) -> History:
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history entries must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    question: str,
    history: Optional[History] = None,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Payload:
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def _post(path: str, payload: Payload, base_url: Optional[str] = None,
          timeout: int = 120) -> dict:
    url = (base_url or BASE_URL).rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(response: dict) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(
            "unexpected response shape: missing choices[0].message.content"
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty content")
    return content.strip()


def chat(question: str, history: Optional[History] = None,
         base_url: Optional[str] = None, timeout: int = 120) -> dict:
    """Ask Spark-X2.6-4B a general health-education question."""
    payload = build_payload(question, history)
    response = _post("/chat/completions", payload, base_url=base_url, timeout=timeout)
    return {
        "model": MODEL_ID,
        "answer": extract_answer(response),
        "disclaimer": DISCLAIMER,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal Spark-X2.6-4B health-information chatbot demo."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)

    result = chat(args.question, base_url=args.base_url)
    print(result["answer"])
    print()
    print(result["disclaimer"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

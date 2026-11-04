"""Minimal patient-education chatbot built on Spark-X2.6-4B.

Spark-X2.6-4B is a text-generation model. This application wraps it with a
fixed safety-oriented system prompt, a bounded prompt and validation of the
model response. The model is served by a local OpenAI-compatible endpoint
(vLLM / SGLang / Ollama), so the demo needs no cloud account.

Run:
    python src/health_chatbot.py "I have a cough and fever after travelling"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to "
    "health information. Explain things in plain language, short paragraphs and "
    "concrete next steps. You must not diagnose, prescribe or give dosages. Say "
    "clearly when a symptom needs urgent in-person care. Always remind the reader "
    "to confirm with a qualified clinician and to contact local emergency "
    "services in an emergency. If the question is outside general health "
    "education, say so instead of guessing."
)

DISCLAIMER = (
    "General health education only - not medical advice, diagnosis or "
    "prescription. Please confirm with a qualified clinician; in an emergency "
    "contact your local emergency services."
)

DEMO_QUESTION = (
    "I came back from a trip three days ago and now I have a dry cough, "
    "38.5 C fever and I feel very tired. What should I watch for, and when "
    "should I see a doctor?"
)

Transport = Callable[[str, Dict[str, Any], Dict[str, str], int], Dict[str, Any]]


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Return the bounded chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history entries need role user/assistant and text content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def extract_text(response: Dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected Spark response shape: %r" % (response,)) from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Spark returned an empty message")
    return content.strip()


def _http_post(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach Spark endpoint %s (%s); start a local OpenAI-compatible "
            "server or set SPARK_BASE_URL" % (url, exc)
        ) from exc
    return json.loads(raw)


class SparkClient:
    """Thin client for an OpenAI-compatible Spark-X2.6-4B endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = MODEL_ID,
        api_key: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT,
        transport: Optional[Transport] = None,
    ) -> None:
        resolved = base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("SPARK_API_KEY", "")
        self.timeout = timeout
        self._transport: Transport = transport or _http_post

    def payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.2,
    ) -> str:
        payload = self.payload(messages, max_tokens=max_tokens, temperature=temperature)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        response = self._transport(
            self.base_url + "/chat/completions", payload, headers, self.timeout
        )
        return extract_text(response)


def answer(
    question: str,
    client: Optional[SparkClient] = None,
    history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Answer one health-education question with Spark-X2.6-4B."""
    client = client or SparkClient()
    messages = build_messages(question, history)
    text = client.chat(messages, max_tokens=max_tokens)
    return {
        "model": client.model,
        "answer": text,
        "disclaimer": DISCLAIMER,
        "messages": messages,
    }


def main(argv: List[str]) -> int:
    question = " ".join(argv[1:]).strip() or DEMO_QUESTION
    try:
        result = answer(question)
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("Q:", question)
    print()
    print(result["answer"])
    print()
    print("--", result["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Minimal health-information chatbot built on Spark-X2.6-4B.

This is the smallest working demo for the HER Hack-Astron #7 entry: it turns a
plain-language health question into a chat request against a locally hosted
Spark-X2.6-4B OpenAI-compatible endpoint (SGLang / vLLM / Ollama) and returns a
short, non-diagnostic educational answer.

The model is used for exactly one thing: rewriting vetted health information
into simple language. It never diagnoses, never prescribes, and always points
the reader to a qualified clinician.

Run it with::

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py \
        "I have a cough and fever after travelling. What should I do?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
REQUEST_TIMEOUT = 120.0
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in simple, "
    "calm, plain language at roughly a 6th-grade reading level.\n"
    "Rules you must never break:\n"
    "1. Do not diagnose and do not recommend prescription medicines or dosages.\n"
    "2. Do not tell the reader to delay or skip professional care.\n"
    "3. If the question suggests an emergency (chest pain, trouble breathing, "
    "heavy bleeding, sudden weakness, thoughts of self-harm), say clearly that "
    "they should contact emergency services right now.\n"
    "4. Suggest only general self-care and the type of clinician to see.\n"
    "5. End with one sentence reminding the reader that this is general "
    "information and not medical advice.\n"
    "Answer in under 200 words."
)

DEMO_QUESTION = (
    "I came back from a trip two days ago and now I have a cough and a mild "
    "fever. What should I do, and when should I see a doctor?"
)

DISCLAIMER = (
    "This is general health information, not medical advice. "
    "Please talk to a qualified clinician about your own situation."
)

# transport(url, payload, headers, timeout) -> decoded JSON response body
Transport = Callable[[str, Dict[str, object], Dict[str, str], float], Dict[str, object]]


class ChatbotError(RuntimeError):
    """Raised when the request or the model response cannot be used."""


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat payload: system safety prompt, prior turns, new question."""
    if not isinstance(question, str) or not question.strip():
        raise ChatbotError("Question must be a non-empty string.")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def extract_text(data: Dict[str, object]) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatbotError("Unexpected response shape from the model endpoint.") from exc
    if not isinstance(content, str) or not content.strip():
        raise ChatbotError("The model returned an empty answer.")
    return content.strip()


def with_disclaimer(text: str) -> str:
    """Guarantee the safety sentence is present exactly once."""
    body = text.strip()
    if DISCLAIMER in body:
        return body
    return "{0}\n\n{1}".format(body, DISCLAIMER)


def _http_transport(
    url: str,
    payload: Dict[str, object],
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, object]:
    """Default transport: a single POST to an OpenAI-compatible endpoint."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ChatbotError(
            "Could not reach the model endpoint at {0}: {1}".format(url, exc)
        ) from exc
    except ValueError as exc:
        raise ChatbotError("The model endpoint returned invalid JSON.") from exc


class HealthChatbot:
    """Thin wrapper around a locally served Spark-X2.6-4B."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = MODEL_ID,
        api_key: Optional[str] = None,
        max_tokens: int = MAX_NEW_TOKENS,
        transport: Optional[Transport] = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.transport: Transport = transport or _http_transport

    @property
    def endpoint(self) -> str:
        return "{0}/chat/completions".format(self.base_url)

    def answer(
        self, question: str, history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        messages = build_messages(question, history)
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {0}".format(self.api_key)
        data = self.transport(self.endpoint, payload, headers, REQUEST_TIMEOUT)
        return with_disclaimer(extract_text(data))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health info chatbot")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEMO_QUESTION,
        help="health question to ask (a demo question is used by default)",
    )
    args = parser.parse_args(argv)

    bot = HealthChatbot(
        base_url=DEFAULT_BASE_URL,
        api_key=os.environ.get("SPARK_API_KEY"),
    )
    try:
        print(bot.answer(args.question))
    except ChatbotError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

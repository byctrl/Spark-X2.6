"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7 demo).

The application answers general health questions in plain language so that people
with low health literacy can understand everyday topics.  The model is
``XHToken/Spark-X2.6-4B`` served by any OpenAI-compatible endpoint (SGLang, vLLM,
llama.cpp server, Spark-MLX-LLM, ...).

Safety boundary: this is patient education only.  The assistant never diagnoses,
never prescribes and never replaces a clinician; a fixed disclaimer is appended to
every reply by application code, not by the model.

Usage::

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py "Apa itu demam berdarah?"
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120
MAX_QUESTION_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a patient-education assistant inside a public-interest health "
    "information app. Explain everyday health topics in plain language at roughly "
    "a 6th-grade reading level, using short sentences and defining medical terms. "
    "Answer in the language of the question. Say clearly when information is "
    "uncertain or depends on the individual. "
    "Never diagnose, never prescribe, never give drug dosages, and never claim to "
    "replace a qualified health professional. If the question asks for a diagnosis, "
    "a treatment plan or a medication dose, say briefly that you cannot provide it "
    "and offer general information instead. Always point the reader to a qualified "
    "health professional for personal decisions, and to local emergency services "
    "for urgent or severe symptoms. Do not ask for, repeat or store names, "
    "addresses, ID numbers or other personal data."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari setelah pulang dari "
    "perjalanan. Informasi umum apa yang perlu saya ketahui, dan kapan sebaiknya "
    "saya menghubungi tenaga kesehatan?"
)

DISCLAIMER = (
    "This is general education, not medical advice. It cannot diagnose or treat a "
    "condition. For personal decisions, or if symptoms are severe or getting worse, "
    "contact a qualified health professional or your local emergency service."
)


class SparkError(RuntimeError):
    """Raised when the Spark endpoint cannot be used or returns an unusable reply."""


def validate_question(question: str) -> str:
    """Return the cleaned question, or raise ValueError if it is unusable."""
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question is too long ({len(cleaned)} > {MAX_QUESTION_CHARS} characters)"
        )
    return cleaned


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat-completion messages sent to Spark-X2.6-4B."""
    cleaned = validate_question(question)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": cleaned})
    return messages


def _http_post_json(
    url: str, payload: Dict[str, Any], timeout: int = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """POST JSON to the endpoint without third-party dependencies."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SparkError(f"cannot reach Spark endpoint at {url}: {exc}") from exc
    try:
        return json.loads(body)
    except ValueError as exc:
        raise SparkError("Spark endpoint returned a non-JSON response") from exc


def extract_message_text(response: Any) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    if not isinstance(response, dict):
        raise SparkError("unexpected response type from Spark endpoint")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SparkError("Spark response contained no choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise SparkError("Spark returned an empty message")
    return content.strip()


class SparkChatClient:
    """Small OpenAI-compatible client for a locally hosted Spark-X2.6-4B."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = MODEL_ID,
        timeout: int = REQUEST_TIMEOUT,
        transport: Optional[Callable[[str, Dict[str, Any], int], Dict[str, Any]]] = None,
    ) -> None:
        resolved = base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport or _http_post_json

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "stream": False,
        }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self._transport(self.endpoint, self.payload(messages), self.timeout)
        return extract_message_text(response)


def format_reply(reply: str) -> str:
    """Attach the fixed safety disclaimer that the model cannot remove."""
    return reply.strip() + "\n\n" + DISCLAIMER


def answer_question(
    question: str,
    client: Optional[Any] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer one health-information question and append the disclaimer."""
    cleaned = validate_question(question)
    active_client = client if client is not None else SparkChatClient()
    messages = build_messages(cleaned, history=history)
    return format_reply(active_client.chat(messages))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B for plain-language health information."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="question text; the demo question is used when omitted",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible base URL (default: $SPARK_BASE_URL or {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip() or DEMO_QUESTION
    try:
        reply = answer_question(question, client=SparkChatClient(base_url=args.base_url))
    except ValueError as exc:
        print(f"invalid question: {exc}")
        return 2
    except SparkError as exc:
        print(f"error: {exc}")
        return 1
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

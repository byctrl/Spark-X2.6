"""Minimal health-information chatbot built on Spark-X2.6-4B.

The model is served through an OpenAI-compatible endpoint (vLLM, SGLang,
Spark-MLX-LLM, Ollama, ...), so this file only uses the Python standard library.

The assistant is deliberately bounded: it explains general health information in
plain language, it never diagnoses or prescribes, and it points the user to a
qualified clinician or to emergency services when that is what the situation
needs.

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py "I have had a cough and a fever for three days"
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py --base-url ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = """You are a patient-education assistant powered by Spark-X2.6-4B.

Your job is to explain general health information in plain, respectful language
so that people with limited health literacy, limited language skills or limited
access to care can understand it.

Rules:
1. Explain, do not diagnose. Never state or imply that the user has a specific
   condition, and never prescribe, dose or change any medication.
2. Answer in the same language the user writes in. Use short sentences,
   everyday words and short bullet points where they help.
3. Say clearly when something needs a qualified clinician, and name the kind of
   professional or service that is appropriate (for example a general
   practitioner, a pharmacist, or a sexual and reproductive health clinic).
4. If the message suggests an emergency (for example chest pain, difficulty
   breathing, heavy bleeding, fainting, a seizure, or thoughts of self-harm),
   tell the user to contact local emergency services right away and keep the
   rest of the answer short.
5. Do not ask for or repeat names, addresses, ID numbers, phone numbers or other
   identifying details. Nothing is stored by this application.
6. If you are unsure, say so instead of guessing. You are not a substitute for
   professional medical care.
"""

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days, starting two days after "
    "I came back from a trip. What should I pay attention to, and when should I "
    "see a doctor?"
)

Transport = Callable[[str, Dict[str, object], float], Dict[str, object]]


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cleaned},
    ]


def build_payload(
    messages: List[Dict[str, str]],
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.3,
) -> Dict[str, object]:
    """Build an OpenAI-compatible /chat/completions request body."""
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def parse_response(data: Dict[str, object]) -> str:
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response from the Spark endpoint: %r" % (data,)) from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the Spark endpoint returned an empty answer")
    return content.strip()


def _http_post(url: str, payload: Dict[str, object], timeout: float) -> Dict[str, object]:
    """Default transport: a single JSON POST with the standard library."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(
    question: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    timeout: float = 180.0,
    transport: Optional[Transport] = None,
) -> str:
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    messages = build_messages(question)
    payload = build_payload(messages, model=model, max_tokens=max_tokens)
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    post = transport or _http_post
    return parse_response(post(url, payload, timeout))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health question (education only, no diagnosis)."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION, help="question in your own words")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint root")
    parser.add_argument("--model", default=MODEL_ID, help="served model name (default: %s)" % MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    try:
        answer = ask(
            args.question,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI reports any failure to the user
        print("Could not get an answer from Spark-X2.6-4B: %s" % exc, file=sys.stderr)
        print(
            "Start a local OpenAI-compatible server for %s (vLLM, SGLang, Spark-MLX-LLM "
            "or Ollama) and pass its root URL with --base-url." % MODEL_ID,
            file=sys.stderr,
        )
        return 1

    print(answer)
    print(
        "\nEducational information only - not a diagnosis. Please consult a qualified clinician.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

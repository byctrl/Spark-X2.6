"""Minimal Spark-X2.6-4B health-information chatbot.

An accessibility-oriented chatbot that answers plain-language health questions
with the local Spark-X2.6-4B model served through an OpenAI-compatible
endpoint (vLLM / SGLang / Spark-MLX-LLM). The model never diagnoses or
prescribes: the system prompt keeps it in the role of a patient-education
assistant and sends the reader to a clinician.

Run:
    python src/health_chatbot.py                 # demo question
    python src/health_chatbot.py -q "..."        # your question
    python src/health_chatbot.py --dry-run       # print the request, no network

Environment:
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    optional, forwarded as a bearer token
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "plain language for a reader with no medical training, at roughly a "
    "6th-grade reading level, using short sentences and short headings.\n"
    "Rules:\n"
    "1. Never diagnose, never name a likely disease as the user's own "
    "condition, and never prescribe or dose a medicine.\n"
    "2. Say plainly what is general knowledge and what needs a clinician.\n"
    "3. Always list the specific signs that mean 'seek care now'.\n"
    "4. Suggest two or three questions the person can ask a clinician.\n"
    "5. If the question looks like an emergency, tell the person to contact "
    "local emergency services first.\n"
    "6. Do not invent sources, statistics, or local phone numbers.\n"
    "End every answer with: 'This is general information, not medical advice.'"
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and a fever "
    "of 38.5 C. What should I watch for, and when should I see a doctor?"
)

EMERGENCY_HINTS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "stroke",
)

EMERGENCY_BANNER = (
    "If this may be an emergency, contact your local emergency number now "
    "instead of waiting for an answer here."
)


def build_messages(question: str, history=None) -> list:
    """Build the chat messages sent to Spark-X2.6-4B."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }


def extract_text(response) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from Spark endpoint") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Spark returned an empty answer")
    return content.strip()


def needs_emergency_banner(question: str) -> bool:
    text = (question or "").lower()
    return any(hint in text for hint in EMERGENCY_HINTS)


def add_emergency_banner(answer: str, question: str) -> str:
    """Deterministic safety net applied in application code, not by the model."""
    if needs_emergency_banner(question) and EMERGENCY_BANNER not in answer:
        return EMERGENCY_BANNER + "\n\n" + answer
    return answer


def request_answer(payload, base_url=None, api_key=None, timeout=REQUEST_TIMEOUT):
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.environ.get("SPARK_API_KEY") or ""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base + "/chat/completions", data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def answer_question(question: str, base_url=None, api_key=None, timeout=REQUEST_TIMEOUT) -> dict:
    """Ask Spark-X2.6-4B one health question and return the checked answer."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    payload = build_payload(build_messages(question))
    response = request_answer(payload, base_url=base_url, api_key=api_key, timeout=timeout)
    text = add_emergency_banner(extract_text(response), question)
    return {"model": response.get("model", MODEL_ID), "answer": text}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health-information chatbot"
    )
    parser.add_argument("-q", "--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible Spark endpoint")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request payload and exit without calling the model",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        payload = build_payload(build_messages(args.question), args.max_new_tokens)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    try:
        result = answer_question(args.question, base_url=args.base_url)
    except (urllib.error.URLError, OSError) as exc:
        print("Could not reach the Spark-X2.6-4B endpoint: %s" % exc)
        print(
            "Start a local OpenAI-compatible server (see FAQ.md), or use "
            "--dry-run to inspect the prompt without a model."
        )
        return 1
    except ValueError as exc:
        print("Invalid model response: %s" % exc)
        return 1

    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7, issue #16).

A dependency-free patient-education assistant. It sends one question to a
locally served Spark-X2.6-4B OpenAI-compatible endpoint (SGLang, vLLM or
Spark-MLX-LLM) and prints the answer.

The chatbot explains health topics in plain language. It never diagnoses,
prescribes or replaces a clinician, and it hands emergency situations to
local emergency services.

Example:
    python src/health_chatbot.py
    python src/health_chatbot.py "How can I lower my risk of heat stroke?"
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py "..."
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 120
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

DEMO_QUESTION = (
    "I have had a cough and a mild fever since I got back from travelling "
    "three weeks ago. What should I know before I see a doctor?"
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain health topics in plain, everyday language for people with no "
    "medical training.\n"
    "Rules you must always follow:\n"
    "1. Never diagnose a condition, never prescribe or name a specific "
    "medicine or dose, and never give a treatment plan.\n"
    "2. Share general, well-established information only; say clearly when "
    "something is uncertain or when guidance varies by country or clinic.\n"
    "3. Always point the reader to a qualified clinician, pharmacist or local "
    "health service for anything personal, worsening or unclear.\n"
    "4. If the message suggests an emergency, tell the reader to contact "
    "local emergency services now.\n"
    "5. Ask at most one short clarifying question and keep answers under "
    "250 words.\n"
    "6. Never ask for or repeat names, addresses, ID numbers, phone numbers "
    "or other personal data.\n"
    "You are not a doctor and you say so when it matters."
)

EMERGENCY_NOTE = (
    "This message may describe a medical emergency. Start your answer by "
    "telling the reader to contact local emergency services (for example "
    "112, 911 or 119) or go to the nearest emergency department right now, "
    "before any other information."
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "severe bleeding",
    "unconscious",
    "not breathing",
    "seizure",
    "suicidal",
    "overdose",
)


class ChatbotError(RuntimeError):
    """Raised when the Spark endpoint cannot be reached or returns no answer."""


def looks_like_emergency(text):
    """Return True when the question contains a clear emergency signal."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def build_messages(question):
    """Build the chat messages for one health question."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must not be empty")

    system_content = SYSTEM_PROMPT
    if looks_like_emergency(cleaned):
        system_content = SYSTEM_PROMPT + "\n\n" + EMERGENCY_NOTE

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": cleaned},
    ]


def build_payload(question, model=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS,
                  temperature=TEMPERATURE):
    """Build the OpenAI-compatible request body for the local Spark server."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def extract_reply(response):
    """Pull the assistant text out of an OpenAI-compatible response object."""
    if not isinstance(response, dict):
        raise ChatbotError("Spark response was not a JSON object")

    choices = response.get("choices") or []
    if not choices:
        raise ChatbotError("Spark response contained no choices")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ChatbotError("Spark response contained an empty answer")
    return content


def call_spark(question, base_url=DEFAULT_BASE_URL, model=MODEL_ID,
               timeout=DEFAULT_TIMEOUT):
    """Send one question to the local Spark-X2.6-4B server and return the answer."""
    payload = build_payload(question, model=model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ChatbotError(
            "Spark endpoint returned HTTP %s: %s" % (exc.code, detail)
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ChatbotError(
            "Cannot reach the Spark endpoint at %s: %s" % (base_url, exc)
        ) from exc

    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise ChatbotError("Spark endpoint returned non-JSON output") from exc

    return extract_reply(parsed)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="health question; defaults to the built-in demo question",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    print("Model: %s" % args.model)
    print("Endpoint: %s" % args.base_url)
    print("Question: %s" % question)
    print("")

    try:
        answer = call_spark(question, base_url=args.base_url, model=args.model)
    except (ChatbotError, ValueError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

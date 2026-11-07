"""Minimal health-information chatbot for the Spark-X2.6 AI for Good challenge.

Wraps a locally hosted, OpenAI-compatible Spark-X2.6-4B endpoint with a
safety-bounded patient-education prompt: plain language, general education
only, no diagnosis, no treatment or dosing advice, always routes the user to
a clinician, and escalates possible emergencies.

Only the Python standard library is used, so the demo runs anywhere Python 3.8+
is available. Nothing is sent anywhere unless the operator points
SPARK_BASE_URL at a server.

Usage:
    python src/health_chatbot.py                    # run the demo question
    python src/health_chatbot.py "my question"      # ask a question
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service.\n"
    "Rules:\n"
    "1. Use short, plain sentences at roughly a 6th-grade reading level. "
    "Explain any medical term you must use.\n"
    "2. You provide general health education only. Never diagnose, never "
    "name a likely disease as the user's condition, never prescribe or give "
    "doses, and never tell the user to stop a prescribed treatment.\n"
    "3. Always point the user to a qualified clinician or pharmacist for "
    "their own case.\n"
    "4. If the message suggests an emergency (for example chest pain, "
    "difficulty breathing, heavy bleeding, fainting, confusion, or severe "
    "dehydration), say clearly that this may be an emergency and that the "
    "user should contact local emergency services now.\n"
    "5. Do not ask for names, addresses, ID numbers or other identifying "
    "details.\n"
    "6. If you are unsure, say what you do not know instead of guessing.\n"
    "Answer with: a short explanation, 2-4 practical self-care or "
    "preparation points, and when to seek care."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days, after a long bus "
    "trip. What can I do at home, and when should I see a doctor?"
)


class SparkEndpointError(RuntimeError):
    """Raised when the Spark-X2.6 endpoint cannot be reached or answers badly."""


class SparkResponseError(ValueError):
    """Raised when the endpoint returns an unusable response body."""


def build_messages(question, system_prompt=SYSTEM_PROMPT):
    """Return the chat messages sent to Spark-X2.6 for one user question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def build_payload(question, model=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Return the OpenAI-compatible request body for one question."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_new_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_answer(raw_response):
    """Pull the assistant text out of an OpenAI-compatible response body."""
    try:
        data = json.loads(raw_response)
    except (TypeError, ValueError) as exc:
        raise SparkResponseError("endpoint did not return JSON") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SparkResponseError("unexpected response shape from model endpoint") from exc
    content = (content or "").strip()
    if not content:
        raise SparkResponseError("model returned an empty answer")
    return content


def resolve_base_url(base_url=None):
    """Resolve the endpoint base URL from argument, environment or default."""
    chosen = base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    return chosen.rstrip("/")


def chat(question, base_url=None, model=MODEL_ID, timeout=TIMEOUT_SECONDS):
    """Send one question to Spark-X2.6 and return the plain-language answer."""
    payload = build_payload(question, model=model)
    url = resolve_base_url(base_url) + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SparkEndpointError(
            "Could not reach the Spark-X2.6 endpoint at %s. Start a local "
            "OpenAI-compatible server (SGLang, vLLM or llama.cpp) or set "
            "SPARK_BASE_URL. Original error: %s" % (url, exc)
        ) from exc
    return extract_answer(raw)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Spark-X2.6 health information demo")
    print("Model:", MODEL_ID)
    print("Question:", question)
    print("-" * 60)
    try:
        print(chat(question))
    except (SparkEndpointError, SparkResponseError, ValueError) as exc:
        print("Error:", exc, file=sys.stderr)
        return 1
    print("-" * 60)
    print("Reminder: general health education only, not medical advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

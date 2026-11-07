"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7 demo).

Public-interest use case: plain-language patient education for people with
limited medical literacy or limited access to a clinician.

Safety boundary: the model never diagnoses or prescribes; it explains general
health topics and points the reader to a qualified health worker. The app runs
fully locally against any OpenAI-compatible server hosting
``XHToken/Spark-X2.6-4B`` (for example SGLang/vLLM on port 30000).

Usage::

    python src/health_chatbot.py "my question here"
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TIMEOUT_SECONDS = 120
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Explain health topics in plain, simple language for "
    "readers with limited medical literacy.\n"
    "Rules:\n"
    "1. Do not diagnose, do not prescribe, and do not give medicine dosages. "
    "Never tell the user which condition they have.\n"
    "2. You do not replace a clinician. Recommend seeing a qualified health "
    "worker for any personal medical decision.\n"
    "3. Say clearly that the information is general and may not match local "
    "guidance or the user's situation.\n"
    "4. If the message suggests an emergency (for example chest pain, "
    "difficulty breathing, heavy bleeding, fainting, or thoughts of self-harm), "
    "tell the user to contact local emergency services or a crisis line now.\n"
    "5. Protect privacy: do not ask for names, ID numbers, addresses, or "
    "contact details.\n"
    "6. Keep answers short, use short sentences, avoid jargon, and end with one "
    "practical next step.\n"
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I have had a cough and a mild "
    "fever for four days. What should I know, and when should I see someone?"
)

REFERRAL_LINE = (
    "This is general education, not medical advice. Please see a qualified "
    "health worker about your own symptoms."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(question, history=None):
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def parse_completion(payload):
    """Extract the answer text from an OpenAI-compatible response payload."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected completion payload")
    text = (content or "").strip()
    if not text:
        raise ValueError("empty completion")
    return text


def ask(question, history=None, base_url=None, opener=urllib.request.urlopen):
    """Send one question to the local Spark-X2.6-4B endpoint and return text."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    body = json.dumps(build_payload(question, history)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach the Spark-X2.6-4B endpoint at %s (%s); start a "
            "local OpenAI-compatible server or set SPARK_BASE_URL" % (url, exc)
        )
    return parse_completion(payload)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Model: %s" % MODEL_ID)
    print("Question: %s" % question)
    print()
    try:
        answer = ask(question)
    except (RuntimeError, ValueError) as exc:
        print("Error: %s" % exc)
        return 1
    print(answer)
    if REFERRAL_LINE not in answer:
        print()
        print(REFERRAL_LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

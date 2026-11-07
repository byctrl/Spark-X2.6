"""Minimal health-information chatbot demo for Spark-X2.6-4B.

Public-interest demo (HER Hack-Astron #7): an inclusive, plain-language health
education assistant that helps people understand general health information and
know when to seek in-person care. It never diagnoses, prescribes or replaces a
clinician.

Standard library only. The model is reached through an OpenAI-compatible
endpoint (vLLM / SGLang / Ollama / Spark-MLX-LLM server) serving
``XHToken/Spark-X2.6-4B``.

Run:
    python src/health_chatbot.py
    python src/health_chatbot.py "I have had a cough and fever for three days."
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 2000
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language that a person without medical training can understand, "
    "and keep the answer short and concrete.\n"
    "Rules you must never break:\n"
    "1. Do not diagnose, do not name a likely disease as the user's condition, and "
    "do not prescribe medicines, doses or treatments.\n"
    "2. Do not tell the user to delay, skip or stop care.\n"
    "3. Say clearly when something needs urgent in-person assessment, and name the "
    "warning signs to watch for.\n"
    "4. Always end by advising the user to confirm with a qualified clinician or "
    "local health service.\n"
    "5. If the question is outside health education, say so and offer only general "
    "guidance."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a dry cough "
    "and fever in the evenings. I have no health insurance and I am not sure "
    "whether this is serious. What should I know before I see someone?"
)

DISCLAIMER = (
    "This is general education, not a diagnosis. "
    "Please confirm with a qualified clinician."
)


def _clean_question(question):
    """Return a stripped, length-bounded question or raise ValueError."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS].rstrip() + " ..."
    return question


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first and cannot be overridden by history
    entries supplied by a caller or a client.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": _clean_question(question)})
    return messages


def build_payload(question, history=None, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body for the Spark endpoint."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": int(max_new_tokens),
        "temperature": 0.3,
        "top_p": 0.9,
    }


def parse_response(data):
    """Extract the answer text from an OpenAI-compatible response body."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ValueError("response has no choices")
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    text = text.strip()
    if not text:
        raise ValueError("response contained no text")
    return text


def with_disclaimer(text):
    """Ensure every answer points the reader to a qualified clinician."""
    lowered = text.lower()
    if "clinician" in lowered or "doctor" in lowered or "health worker" in lowered:
        return text
    return text + "\n\n" + DISCLAIMER


def chat(question, history=None, base_url=None, api_key=None, timeout=REQUEST_TIMEOUT):
    """Send one question to a local Spark-X2.6-4B server and return the answer."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    api_key = api_key or os.environ.get("SPARK_API_KEY") or "not-needed"
    body = json.dumps(build_payload(question, history)).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return with_disclaimer(parse_response(data))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model:", MODEL_ID)
    print("Question:", question)
    try:
        print("\nAnswer:\n" + chat(question))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        print("\nCould not get an answer from the Spark-X2.6-4B endpoint: %s" % exc, file=sys.stderr)
        print(
            "Start an OpenAI-compatible server for %s and set SPARK_BASE_URL "
            "(default %s)." % (MODEL_ID, DEFAULT_BASE_URL),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

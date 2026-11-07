"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Scope: general patient education in plain language. The model is never asked
to diagnose, prescribe, or replace a clinician, and every answer is anchored
to a safety system prompt that redirects emergencies to local services.

Run against a local OpenAI-compatible Spark-X2.6 server:

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py

No third-party dependencies: only the Python standard library is used.
"""

import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are the patient-education assistant of a public-interest health access "
    "app. Explain health topics in plain language (aim for a 6th-8th grade "
    "reading level), short and structured. "
    "Rules you must never break: do not diagnose; do not name a prescription "
    "medicine, dose or treatment plan; do not tell the user to start, stop or "
    "change any treatment. "
    "Always state that this is general education and not medical advice, and "
    "direct the user to a qualified clinician or their local health service. "
    "If the question suggests an emergency (chest pain, trouble breathing, "
    "fainting, heavy bleeding, thoughts of self-harm), tell the user to contact "
    "emergency services now and keep the rest of the answer very brief."
)

DISCLAIMER = (
    "General education only - not a diagnosis or a prescription. "
    "Please consult a qualified clinician."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a fever in the evenings. What could it be and what should I do?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages):
    """Wrap messages in an OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_answer(data):
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response from Spark endpoint: %r" % (data,))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Spark returned an empty answer")
    return content.strip()


def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(question, history=None, base_url=None, timeout=120, transport=None):
    """Ask Spark-X2.6-4B one health-education question and return its answer."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(build_messages(question, history))
    if transport is None:
        data = _post_json(url, payload, timeout)
    else:
        data = transport(url, payload)
    return extract_answer(data)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Q:", question)
    print("A:", chat(question))
    print("-", DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Minimal Spark-X2.6-4B health-information chatbot demo.

Purpose: accessible, plain-language patient education for people who cannot
easily reach a clinician. The model is used only to explain general health
information in simple words. It must never diagnose or prescribe. The caller
remains responsible for directing users to professional care.

Runs against any OpenAI-compatible Spark-X2.6-4B endpoint
(default: http://127.0.0.1:30000/v1, override with SPARK_BASE_URL).
Only the Python standard library is used.
"""

import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information "
    "service. Explain health topics in short, simple sentences, ideally at a "
    "6th-grade reading level, and define any unavoidable medical term. "
    "You must NOT provide a diagnosis, a treatment plan, a drug dose, or a "
    "prediction about a specific person's condition. Do not claim to be a "
    "clinician. For anything urgent or personal, tell the user to contact a "
    "qualified clinician or local emergency service, and mention warning signs "
    "that require immediate care. If the question is outside health topics, "
    "say so briefly. Never invent sources or statistics."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for three days, starting two "
    "days after I returned from a trip abroad. I don't have a doctor yet. "
    "What should I know before I decide what to do next?"
)

SAFETY_NOTE = (
    "This is general health information, not medical advice. "
    "Please contact a qualified clinician for your own situation."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    history is an optional list of {"role": ..., "content": ...} turns.
    Unsupported roles and blank turns are ignored.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def _default_transport(url, data, timeout):
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_reply(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("malformed response from Spark-X2.6-4B") from exc
    text = (content or "").strip()
    if not text:
        raise RuntimeError("empty response from Spark-X2.6-4B")
    return text


def chat(question, history=None, base_url=None, timeout=120, transport=None):
    """Ask Spark-X2.6-4B a health question and return the answer text.

    transport is injectable for tests: transport(url, data, timeout) -> dict.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    payload = {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    sender = transport or _default_transport
    return extract_reply(sender(url, data, timeout))


def main():
    print("Spark-X2.6-4B health information demo")
    print("Model endpoint:", BASE_URL)
    print("Question:", DEMO_QUESTION)
    print("-" * 60)
    print(chat(DEMO_QUESTION))
    print("-" * 60)
    print(SAFETY_NOTE)


if __name__ == "__main__":
    main()

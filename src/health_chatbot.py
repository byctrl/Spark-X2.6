"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Run against a locally served, OpenAI-compatible Spark-X2.6-4B endpoint:

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py

Standard library only. The assistant answers in plain language, never
provides a diagnosis or a treatment plan, and always points the reader to a
qualified clinician. It is an education aid, not a medical device.
"""

import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
TIMEOUT = float(os.environ.get("SPARK_TIMEOUT", "120"))

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "simple, respectful language that a person without medical training can "
    "understand. You must not diagnose, prescribe, or replace a clinician: "
    "do not name a likely disease as a conclusion, do not give dosages, and "
    "do not tell the reader to stop a prescribed treatment. State clearly "
    "which symptoms need urgent in-person care, and always end by advising "
    "the reader to consult a qualified health professional. If the question "
    "is outside general health education, say so briefly and suggest a "
    "trusted source. Answer in the language of the question."
)

SAFETY_NOTE = (
    "This is general health education only, not a diagnosis or treatment "
    "advice. Please consult a qualified clinician about your own situation."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a mild fever in the evening. What could it be and what should I do?"
)


def build_messages(question):
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Build a bounded, non-streaming chat-completions request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def extract_text(response):
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "response does not contain choices[0].message.content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def generate(question, base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT):
    """Send one question to the local Spark-X2.6-4B server and return text."""
    payload = json.dumps(build_payload(build_messages(question))).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return extract_text(json.load(response))


def main():
    print("Model:", MODEL_ID)
    print("Endpoint:", BASE_URL)
    print("Question:", DEMO_QUESTION)
    try:
        answer = generate(DEMO_QUESTION)
    except Exception as exc:  # noqa: BLE001 - demo script reports the cause
        print("\nCould not get an answer from Spark-X2.6-4B.")
        print("Reason:", exc)
        print("Start a local OpenAI-compatible Spark-X2.6-4B server, then rerun.")
        return 1
    print("\nAnswer:\n" + answer)
    print("\n" + SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

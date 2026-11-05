"""Minimal Spark-X2.6-4B health-information chatbot.

Demo application for the HER Hack-Astron #7 "AI for Good" challenge.
The assistant explains general health topics in plain language and never
diagnoses or prescribes; every answer redirects the user to a qualified
clinician, and red-flag messages get an emergency notice.

Usage (requires a local OpenAI-compatible Spark-X2.6 server):

    python src/health_chatbot.py "I have had a cough for four days"
    python src/health_chatbot.py            # runs DEMO_QUESTION

The model is ``XHToken/Spark-X2.6-4B`` (text only). Serving examples:
SGLang or vLLM with an OpenAI-compatible endpoint on 127.0.0.1:30000.
"""

import argparse
import json
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant that helps people understand "
    "general health information in plain language.\n"
    "Rules you must follow:\n"
    "1. Explain concepts, common causes and when to seek care. Do not diagnose "
    "the user and do not name a specific disease as the user's condition.\n"
    "2. Do not prescribe medicines, doses or treatments, and do not tell the "
    "user to stop or change a prescribed treatment.\n"
    "3. Recommend consulting a qualified clinician or local health service for "
    "anything specific to the user's situation.\n"
    "4. Say clearly when information is general and may not apply locally.\n"
    "5. If the message suggests an emergency, tell the user to contact "
    "emergency services immediately.\n"
    "Keep answers short, calm and free of jargon."
)

DISCLAIMER = (
    "This is general health information for education, not medical advice, "
    "diagnosis or treatment. Please consult a qualified clinician about your "
    "own situation."
)

EMERGENCY_NOTICE = (
    "This may be an emergency. Contact your local emergency number or go to "
    "the nearest emergency department now. The information below is general "
    "and is not a substitute for emergency care."
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "stroke",
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days since returning from a "
    "trip. What should I know before I see a doctor?"
)


class ChatError(RuntimeError):
    """Raised when the model endpoint returns an unusable answer."""


def needs_emergency_notice(question):
    """Return True when the question contains a red-flag emergency phrase."""
    text = (question or "").lower()
    return any(keyword in text for keyword in EMERGENCY_KEYWORDS)


def build_messages(question, history=()):
    """Build the chat messages for one user question plus prior turns."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=(), max_new_tokens=MAX_NEW_TOKENS, temperature=0.2):
    """Build an OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }


def http_transport(url, payload, timeout):
    """POST the payload to an OpenAI-compatible endpoint and decode JSON."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(response):
    """Validate the endpoint response and return the model text."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatError(
            "malformed model response: missing choices[0].message.content"
        ) from exc
    text = (content or "").strip()
    if not text:
        raise ChatError("model returned an empty answer")
    return text


def compose_answer(question, model_text):
    """Attach the safety notices to the model text."""
    parts = []
    if needs_emergency_notice(question):
        parts.append(EMERGENCY_NOTICE)
    parts.append(model_text)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def ask(
    question,
    history=(),
    transport=http_transport,
    base_url=DEFAULT_BASE_URL,
    max_new_tokens=MAX_NEW_TOKENS,
):
    """Ask Spark-X2.6-4B one question and return a safety-wrapped answer."""
    payload = build_payload(question, history, max_new_tokens)
    url = base_url.rstrip("/") + "/chat/completions"
    response = transport(url, payload, REQUEST_TIMEOUT)
    return compose_answer(question, extract_answer(response))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-info chatbot")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    print(ask(args.question, base_url=args.base_url))


if __name__ == "__main__":
    main()

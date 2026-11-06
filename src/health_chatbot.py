"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest (health + accessibility) demo for HER Hack-Astron #7.
The model is used as a patient-education assistant: it explains general
health information in plain language, never diagnoses or prescribes, and
always points the person to a qualified clinician.

The model itself is text-only (see FAQ.md), so no image input is claimed
anywhere in this demo.

Run against a local OpenAI-compatible Spark endpoint::

    python src/health_chatbot.py
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py \
        --question "What should I note down before a doctor visit?"
"""

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
TIMEOUT_SECONDS = 120

DISCLAIMER = (
    "This is general health information for education only. It is not a "
    "diagnosis and does not replace advice from a qualified clinician. "
    "If symptoms are severe or getting worse, seek in-person care now."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You explain health information in "
    "plain, everyday language for people with no medical training.\n"
    "Rules you must follow:\n"
    "1. Do not diagnose the person and do not prescribe or recommend specific "
    "medicines, doses, or treatments.\n"
    "2. Do not state whether a symptom is definitely serious or definitely harmless.\n"
    "3. Encourage the person to contact a qualified clinician or local health "
    "service, and say that emergency symptoms need immediate in-person care.\n"
    "4. Offer general, well-established information: what a symptom can commonly "
    "mean, what to note down before an appointment, and which questions to ask.\n"
    "5. If the question is outside general health information, say so briefly and "
    "suggest a more suitable source.\n"
    "6. Be calm and non-judgemental. Keep the answer short and clearly structured."
)

EMERGENCY_NOTE = (
    "The person may be describing a possible emergency. Start your reply by "
    "telling them to contact local emergency services or go to the nearest "
    "emergency department now. Keep the rest of the answer brief."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and now I have a dry cough "
    "and a fever of 38.4 C. What should I know before I see a doctor?"
)

EMERGENCY_PATTERNS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "not breathing",
    "suicidal",
    "suicide",
    "overdose",
    "stroke",
    "seizure",
    "anaphylaxis",
)


def is_emergency(text):
    """Return True when the text mentions a possible emergency symptom."""
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in EMERGENCY_PATTERNS)


def build_messages(question, history=None, emergency=False):
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if emergency:
        messages.append({"role": "system", "content": EMERGENCY_NOTE})

    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError("history entries need role user/assistant and content")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Build an OpenAI-compatible request body for the Spark endpoint."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def request_completion(messages, base_url=None, max_new_tokens=MAX_NEW_TOKENS,
                       timeout=TIMEOUT_SECONDS):
    """POST the messages to a local Spark-X2.6-4B chat endpoint."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    body = json.dumps(build_payload(messages, max_new_tokens)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(response):
    """Pull the assistant text out of a chat-completion response."""
    choices = (response or {}).get("choices") or []
    if not choices:
        raise ValueError("model response contained no choices")
    content = ((choices[0] or {}).get("message") or {}).get("content") or ""
    content = content.strip()
    if not content:
        raise ValueError("model response contained an empty answer")
    return content


def with_disclaimer(answer):
    """Always attach the safety disclaimer to the model answer."""
    return (answer or "").strip() + "\n\n" + DISCLAIMER


def answer_question(question, history=None, base_url=None,
                    max_new_tokens=MAX_NEW_TOKENS):
    """Answer one health question using Spark-X2.6-4B."""
    emergency = is_emergency(question)
    messages = build_messages(question, history=history, emergency=emergency)
    response = request_completion(
        messages, base_url=base_url, max_new_tokens=max_new_tokens
    )
    return {
        "model": MODEL_ID,
        "emergency": emergency,
        "answer": with_disclaimer(extract_answer(response)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask the Spark-X2.6-4B health-information chatbot a question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    result = answer_question(args.question, base_url=args.base_url)
    if result["emergency"]:
        print("[possible emergency detected - seek in-person care now]\n")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

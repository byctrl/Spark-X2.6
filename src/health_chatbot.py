"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Scope, deliberately small:
  * plain-language patient *education* only, never diagnosis or prescriptions;
  * every answer is sent through :func:`with_disclaimer`;
  * possible emergencies are flagged before the model is even called;
  * one OpenAI-compatible HTTP call, standard library only (no new deps).

The model itself (``XHToken/Spark-X2.6-4B``) does the language work: it turns a
layperson question plus the fixed system prompt into a short, simple-language
explanation of what a symptom or term generally means and which kind of
professional to contact. Everything else (validation, emergency routing,
disclaimer, output limits) is application code, per the repo FAQ guidance that
a plausible-looking model answer is not by itself a correct one.

Run it against a local OpenAI-compatible server::

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py \
        "I have a cough and fever after travelling. What should I do?"

Without a server it prints the exact request it would send, so the demo is
still reproducible offline.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a health-education assistant for people who may have limited "
    "health literacy or limited access to a clinician.\n"
    "Rules you must follow:\n"
    "1. Use plain language, short sentences, and at most 6 bullet points.\n"
    "2. Explain what the described symptom or term generally means and what "
    "people commonly do about it.\n"
    "3. Do NOT diagnose, do NOT name a likely disease as a conclusion, and do "
    "NOT recommend prescription medicines or dosages.\n"
    "4. Always say which kind of professional or service to contact, and when "
    "to seek care urgently.\n"
    "5. If the question is outside health education, say so briefly.\n"
    "6. Never claim to replace a clinician. Do not invent sources."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days since I came back "
    "from a trip. I do not have a doctor. What should I do?"
)

DISCLAIMER = (
    "This is general health information, not medical advice or a diagnosis. "
    "Please contact a qualified health professional about your own situation."
)

EMERGENCY_NOTICE = (
    "This sounds like it could be an emergency. Contact your local emergency "
    "number or go to the nearest emergency department now. Do not wait for a "
    "chat answer."
)

EMERGENCY_FLAGS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "severe bleeding",
    "unconscious",
    "suicid",
    "胸痛",
    "呼吸困难",
    "大量出血",
    "自杀",
)


def build_messages(question: str, history=None):
    """Build an OpenAI-compatible message list for Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Wrap messages in the request body the local Spark server expects."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }


def extract_answer(body: dict) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "unexpected response from the Spark endpoint: %r" % (body,)
        ) from exc
    return (content or "").strip()


def emergency_notice(question: str) -> str:
    """Return an urgent-care notice when the text matches an emergency cue."""
    lowered = (question or "").lower()
    for flag in EMERGENCY_FLAGS:
        if flag in lowered:
            return EMERGENCY_NOTICE
    return ""


def with_disclaimer(answer: str, disclaimer: str = DISCLAIMER) -> str:
    """Append the safety disclaimer exactly once."""
    answer = (answer or "").strip()
    if disclaimer in answer:
        return answer
    if not answer:
        return disclaimer
    return answer + "\n\n" + disclaimer


def ask_spark(question: str, history=None, base_url: str = None, timeout: int = 120) -> str:
    """Send one bounded request to a local Spark-X2.6 OpenAI-compatible server."""
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(build_messages(question, history))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_answer(body)


def answer_question(question: str, history=None, generator=None) -> str:
    """Application entry point: safety routing first, then the model."""
    notice = emergency_notice(question)
    if notice:
        return notice
    if generator is None:
        generator = ask_spark
    return with_disclaimer(generator(question, history))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION

    notice = emergency_notice(question)
    if notice:
        print(notice)
        return 0

    try:
        print(with_disclaimer(ask_spark(question)))
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        print("Could not reach the Spark-X2.6-4B server (%s)." % (exc,))
        print("Start one, or inspect the request that would be sent:")
        print(json.dumps(build_payload(build_messages(question)), indent=2))
        print("")
        print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

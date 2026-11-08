"""Minimal Spark-X2.6-4B health-information chatbot.

Entry for HER Hack-Astron #7 (Spark-X2.6 Open Source AI for Good Challenge):
a plain-language health education assistant for people with limited access to
care. The model only explains and routes; it never diagnoses or prescribes.

Requires a locally served Spark-X2.6-4B OpenAI-compatible endpoint, e.g.

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py

No third-party dependencies are used on purpose: the demo must stay runnable
offline, next to the model, on modest hardware.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a health-education assistant for people with limited access to "
    "medical care. Explain in plain language, short paragraphs and short "
    "bullet points. Hard rules:\n"
    "- You do not diagnose, prescribe, or give medicine dosages. Never state "
    "or imply a definitive diagnosis.\n"
    "- Never tell the user to start, stop, or change any treatment or medicine.\n"
    "- Always name the kind of clinician or service to consult and what to say "
    "when they get there.\n"
    "- If the situation may be an emergency, say so first and tell the user to "
    "contact local emergency services now.\n"
    "- If key information is missing, ask at most two clarifying questions "
    "instead of guessing.\n"
    "- Say once, briefly, that you are not a substitute for a qualified "
    "professional.\n"
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, after returning from a "
    "trip. What should I do next, and what should I tell the clinic?"
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "severe bleeding",
    "heavy bleeding",
    "unconscious",
    "fainted",
    "seizure",
    "stroke",
    "suicid",
    "harm myself",
)

EMERGENCY_NOTICE = (
    "If this may be an emergency (for example chest pain, trouble breathing, "
    "heavy bleeding, fainting, or thoughts of harming yourself), contact your "
    "local emergency number right now. This assistant cannot handle "
    "emergencies."
)


def build_messages(question, history=None):
    """Return the chat messages sent to Spark for one question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def needs_emergency_notice(text):
    """Application-side red-flag check; model output is never the only gate."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def post_chat(
    messages,
    base_url=None,
    model=MODEL_ID,
    timeout=DEFAULT_TIMEOUT,
    api_key=None,
):
    """POST messages to the local OpenAI-compatible Spark endpoint."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.2,
            "stream": False,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("SPARK_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key

    request = urllib.request.Request(
        root + "/chat/completions", data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_reply(payload):
    """Extract the assistant text; fail loudly instead of returning junk."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from Spark: %r" % (payload,))
    content = (content or "").strip()
    if not content:
        raise ValueError("Spark returned an empty reply")
    return content


def answer(question, transport=None, history=None):
    """Build the prompt, call the model, and apply safety guards."""
    messages = build_messages(question, history)
    send = transport or post_chat
    reply = parse_reply(send(messages))
    if needs_emergency_notice(question) and EMERGENCY_NOTICE not in reply:
        reply = EMERGENCY_NOTICE + "\n\n" + reply
    return reply


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    try:
        print(answer(question))
    except (urllib.error.URLError, OSError) as exc:
        endpoint = os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
        print("Could not reach Spark-X2.6-4B at %s: %s" % (endpoint, exc), file=sys.stderr)
        print("Serve the model locally and retry (see README quickstart).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

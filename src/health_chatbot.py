"""Minimal health-education chatbot powered by Spark-X2.6-4B.

What the model does here: the app turns a user's plain-language health
question into a bounded chat request, sends it to a locally hosted
Spark-X2.6-4B OpenAI-compatible endpoint, and returns the model's
plain-language explanation. The prompt forbids diagnosis and treatment
advice, and the app adds an emergency notice when red-flag wording is
detected. A clinician remains the decision maker.

Usage:
    python src/health_chatbot.py                 # run the built-in demo question
    python src/health_chatbot.py "pertanyaan"    # ask your own question

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   optional, default "local"

This is an education aid, not a medical device.
"""

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You explain health information in "
    "plain language for people with no medical training, and you answer in the "
    "same language the user writes in.\n"
    "Rules you must follow:\n"
    "1. Never give a diagnosis and never name a medicine, dose or treatment plan.\n"
    "2. Explain in general terms what the described signs can mean, which warning "
    "signs need urgent care, and what a clinician will likely check.\n"
    "3. Always end by telling the user to see a qualified clinician for diagnosis "
    "and treatment. Never discourage someone from seeking care.\n"
    "4. If the description suggests an emergency, say clearly that emergency "
    "services should be contacted right now.\n"
    "5. Do not invent sources, statistics, phone numbers or local services.\n"
    "Keep the answer short (under 200 words) and use plain sentences."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, setelah pulang "
    "dari perjalanan. Apa yang sebaiknya saya perhatikan?"
)

EMERGENCY_KEYWORDS = (
    "nyeri dada",
    "sesak napas",
    "sesak nafas",
    "sulit bernapas",
    "pendarahan hebat",
    "kejang",
    "tidak sadar",
    "chest pain",
    "shortness of breath",
    "cannot breathe",
    "severe bleeding",
    "unconscious",
    "seizure",
)

EMERGENCY_NOTE = (
    "PERHATIAN: uraian Anda memuat tanda bahaya. Hubungi layanan gawat darurat "
    "atau klinik terdekat sekarang, jangan menunggu jawaban chatbot ini."
)


def build_messages(question):
    """Return the chat messages for one question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(messages):
    """Wrap messages in an OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": list(messages),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
        "stream": False,
    }


def needs_emergency_notice(text):
    """True when the text contains wording that needs an urgent-care notice."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def parse_response(data):
    """Extract the assistant text from a chat-completions response."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape: missing choices[0]") from exc

    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "model returned no content (finish_reason=%r)" % (choice.get("finish_reason"),)
        )
    return content.strip()


def chat(question, base_url=None, api_key=None, timeout=DEFAULT_TIMEOUT, opener=None):
    """Send one question to Spark-X2.6-4B and return the plain-text answer."""
    payload = build_payload(build_messages(question))
    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if api_key is None:
        api_key = os.environ.get("SPARK_API_KEY", "") or "local"

    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )

    open_request = opener or urllib.request.urlopen
    try:
        with open_request(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(
            "Spark endpoint returned HTTP %s: %s" % (exc.code, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach the Spark endpoint at %s (%s). Start the local "
            "server first." % (base_url, exc.reason)
        ) from exc

    return parse_response(json.loads(raw))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    if needs_emergency_notice(question):
        print(EMERGENCY_NOTE)
        print()
    print("Q:", question)
    print()
    try:
        print(chat(question))
    except (RuntimeError, ValueError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1
    print()
    print("Reminder: this is health education only, not a diagnosis. See a clinician.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

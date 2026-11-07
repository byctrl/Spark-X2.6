"""Minimal Spark-X2.6-4B health-information chatbot.

A small, dependency-free demo for the HER Hack-Astron #7 AI-for-Good challenge.
It sends plain-language health questions to a locally hosted Spark-X2.6-4B
OpenAI-compatible endpoint and returns an educational answer.

The assistant never diagnoses or prescribes. It explains a topic in simple
language, points out warning signs, and directs the user to a qualified
clinician. The application is an information aid only; it is not a medical
device and never replaces professional care.

Usage:
    python src/health_chatbot.py "my question"
    python src/health_chatbot.py            # runs the built-in demo question

Environment:
    SPARK_BASE_URL  OpenAI-compatible base URL of the local server.
                    Default http://127.0.0.1:30000/v1 (see README quickstart).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
SPARK_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest health information "
    "service. Explain health topics in simple, everyday language that a person "
    "without medical training can follow. You do not diagnose, you do not prescribe, "
    "and you never suggest medicine names or dosages. State clearly that your answer "
    "is general education and not a substitute for a qualified clinician. Always list "
    "the warning signs that mean the person should seek urgent care, and tell the user "
    "to consult a doctor, nurse, pharmacist, or community health worker about anything "
    "specific to their own body or situation. If the question suggests an emergency, "
    "say plainly that the person should contact local emergency services now. Do not "
    "ask for or repeat names, addresses, ID numbers, or other personal identifiers."
)

DISCLAIMER = (
    "This tool gives general health education only. It is not a diagnosis and not a "
    "prescription. For anything about your own health, contact a qualified clinician. "
    "In an emergency, call your local emergency number now."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago. Since yesterday I have a dry "
    "cough and a fever of 38.2 C. I feel tired but I can still eat. I do not have "
    "insurance. What should I know before I talk to a doctor, and what should make me "
    "seek help right away?"
)


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> list:
    """Build the chat messages sent to the model."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_payload(
    question: str,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Build the OpenAI-compatible request body."""
    return {
        "model": model_id,
        "messages": build_messages(question),
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def extract_answer(body: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        answer = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response from the Spark-X2.6-4B endpoint") from exc
    answer = (answer or "").strip()
    if not answer:
        raise ValueError("the Spark-X2.6-4B endpoint returned an empty answer")
    return answer


def generate(
    question: str,
    base_url: str = SPARK_BASE_URL,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Send one question to a local Spark-X2.6-4B server and return the answer."""
    payload = build_payload(question, model_id=model_id, max_new_tokens=max_new_tokens)
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Spark-X2.6-4B at {0}. Start a local OpenAI-compatible "
            "server such as SGLang, vLLM, or Spark-MLX-LLM (see README quickstart) "
            "and retry.".format(url)
        ) from exc
    except ValueError as exc:
        raise ValueError("the endpoint did not return valid JSON") from exc
    return extract_answer(body)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model: {0}".format(MODEL_ID))
    print("Question: {0}".format(question))
    print("")
    try:
        answer = generate(question)
    except (RuntimeError, ValueError) as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        return 1
    print(answer)
    print("")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

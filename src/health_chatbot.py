"""Minimal Spark-X2.6-4B health-information chatbot.

A small, dependency-free demo for the HER Hack-Astron #7 "AI for Good"
challenge: more inclusive access to plain-language health information.

The assistant is deliberately non-diagnostic. It explains in simple words what
a symptom or notice usually means, what general self-care is commonly
suggested, which warning signs need urgent care, and when to contact a
clinician. It never diagnoses and never prescribes medicine or doses.

Run against a local Spark-X2.6 OpenAI-compatible server (SGLang / vLLM / Ollama):

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py --demo

Or ask your own question:

    python src/health_chatbot.py "How do I read this clinic bill?"
"""

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest health "
    "accessibility project. Explain health topics in plain, everyday language "
    "for readers with no medical training.\n"
    "Rules you must always follow:\n"
    "1. Never give a diagnosis and never name a prescription medicine or a dose.\n"
    "2. Explain what the symptom or document usually means and what general "
    "self-care is commonly suggested.\n"
    "3. Always list clear warning signs: 'seek urgent care if ...'.\n"
    "4. Always tell the reader to confirm with a qualified clinician or "
    "pharmacist, and never delay care because of this chat.\n"
    "5. Do not ask for or repeat personal identifiers such as name, address, "
    "ID numbers or insurance numbers.\n"
    "6. Answer in the language the user writes in, under 250 words."
)

SAFETY_NOTICE = (
    "This is general health information, not a diagnosis or a prescription. "
    "Please confirm with a qualified clinician or pharmacist, and seek urgent "
    "care if symptoms are severe or getting worse."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for five days, since coming back from "
    "a trip. I have no health insurance and I am not sure whether I should go "
    "to a clinic. What should I do?"
)


def build_messages(question, history=None):
    """Build the chat messages for one Spark request.

    The system prompt is always first so the safety rules cannot be dropped by
    earlier conversation turns.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not content:
            raise ValueError(
                "history entries need a role of user/assistant and non-empty content"
            )
        messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_request(question, history=None):
    """Build the OpenAI-compatible request body sent to Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
        "stream": False,
    }


def extract_text(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the Spark endpoint") from exc

    text = (content or "").strip()
    if not text:
        raise ValueError("Spark returned an empty answer")
    return text


def chat_completion(question, history=None, base_url=None, timeout=120):
    """Send the question to a local Spark-X2.6-4B endpoint and return the text."""
    base_url = (
        base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    payload = json.dumps(build_request(question, history)).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_text(body)


def answer(question, history=None, base_url=None):
    """Return a plain-language answer with the safety notice guaranteed present."""
    reply = chat_completion(question, history=history, base_url=base_url)
    if SAFETY_NOTICE not in reply:
        reply = reply + "\n\n" + SAFETY_NOTICE
    return reply


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information demo"
    )
    parser.add_argument(
        "question", nargs="?", help="question to ask; defaults to the demo question"
    )
    parser.add_argument(
        "--demo", action="store_true", help="run the built-in demo scenario"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: SPARK_BASE_URL or %s)"
        % DEFAULT_BASE_URL,
    )
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if args.demo or not args.question else args.question
    print("Model:", MODEL_ID)
    print("Question:", question)
    print("-" * 60)
    try:
        reply = answer(question, base_url=args.base_url)
    except (urllib.error.URLError, OSError) as exc:
        print("Could not reach the Spark endpoint:", exc)
        print("Start a local OpenAI-compatible server and set SPARK_BASE_URL.")
        return 1
    except ValueError as exc:
        print("Unexpected model response:", exc)
        return 1

    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

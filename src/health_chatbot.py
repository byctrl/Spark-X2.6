"""Minimal health-education chatbot powered by Spark-X2.6-4B.

Public-interest demo for the HER Hack-Astron #7 / Spark-X2.6 Open Source
AI for Good Challenge: inclusive, plain-language health information.

The model is served by a local OpenAI-compatible endpoint (vLLM, SGLang,
Ollama or Spark-MLX-LLM). This module never diagnoses and never prescribes;
it only explains general health information and points people to a clinician.

Usage:
    python src/health_chatbot.py "I have had a cough for three days"
    python src/health_chatbot.py            # runs DEMO_QUESTION
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1 "..."
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
MAX_NEW_TOKENS = 1024
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health "
    "information in plain language at roughly a 6th-grade reading level. "
    "Use short sentences and briefly define medical terms.\n"
    "Rules you must always follow:\n"
    "1. Do not diagnose. Do not tell the user what disease or condition they "
    "have. Do not recommend prescription drugs, dosages, or treatments.\n"
    "2. Share only general, widely accepted health information.\n"
    "3. Always tell the user to confirm with a qualified clinician, and say "
    "clearly when a symptom needs urgent in-person care.\n"
    "4. Never ask for or repeat identifying details such as names, addresses, "
    "ID numbers, or insurance numbers.\n"
    "5. If the question is not about health education, say so and offer to "
    "explain a health topic instead.\n"
    "6. If you are unsure, say that you are unsure.\n"
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and a "
    "fever. What should I know before I see a doctor?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please confirm with a qualified clinician."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and the new question is always last,
    so the safety rules cannot be replaced by earlier conversation turns.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history turns must use role 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages):
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }


def extract_text(data):
    """Return the assistant text from an OpenAI-compatible response."""
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has empty content")
    return content.strip()


def _http_post(url, payload):
    """POST the payload to the local model server and return raw bytes."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def chat(question, base_url=DEFAULT_BASE_URL, history=None, transport=None):
    """Ask Spark-X2.6-4B one health-education question.

    ``transport`` is an optional ``callable(url, payload) -> bytes|str`` used
    by the tests so the demo can be verified without a running model server.
    """
    messages = build_messages(question, history)
    payload = build_payload(messages)
    post = transport or _http_post
    url = base_url.rstrip("/") + "/chat/completions"

    raw = post(url, payload)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)

    return {
        "answer": extract_text(data),
        "model": data.get("model") or MODEL_ID,
        "usage": data.get("usage") or {},
        "prompt_messages": messages,
        "disclaimer": DISCLAIMER,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health-education question."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)

    try:
        result = chat(args.question, base_url=args.base_url)
    except (urllib.error.URLError, OSError) as exc:
        print(
            "Could not reach the Spark-X2.6 endpoint at %s: %s"
            % (args.base_url, exc),
            file=sys.stderr,
        )
        print(
            "Start a local OpenAI-compatible server for %s first "
            "(see the model card and README quickstart)." % MODEL_ID,
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print("Model response could not be used: %s" % exc, file=sys.stderr)
        return 3

    print(result["answer"])
    print()
    print(result["disclaimer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

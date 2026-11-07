"""Minimal Spark-X2.6-4B health-information chatbot.

A small, safety-bounded public-interest demo for the HER Hack-Astron #7
"AI for Good" challenge: a chatbot that makes plain-language health
information easier to reach.

The model (``XHToken/Spark-X2.6-4B``) is loaded by a local
OpenAI-compatible server (SGLang / vLLM / llama.cpp / Spark-MLX-LLM).
This script is intentionally dependency-free: it only uses the Python
standard library and talks to the endpoint over HTTP.

Scope and limits (please read before using it with real people):

* The assistant gives general education only. It never diagnoses,
  prescribes, or triages an emergency.
* Every reply that does not already point at a clinician is followed by
  an explicit safety note.
* It is not a medical device and must not be presented as medical advice.

Run the demo (after starting a local Spark-X2.6-4B endpoint)::

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python health_chatbot.py
    python health_chatbot.py "How can I prepare for a heat wave?"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and have had a cough and a mild "
    "fever for three days. What should I watch for, and when should I see "
    "a doctor?"
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. You help people understand general health topics "
    "in plain language.\n"
    "Rules you must follow:\n"
    "1. Never diagnose, never prescribe, and never give dosages.\n"
    "2. Do not say or imply that your answer replaces a qualified clinician.\n"
    "3. Explain things in short, simple sentences and avoid unexplained "
    "medical jargon.\n"
    "4. If the question suggests an emergency (for example chest pain, "
    "difficulty breathing, fainting, heavy bleeding, signs of stroke, or "
    "thoughts of self-harm), tell the person to contact local emergency "
    "services or a crisis line right away.\n"
    "5. End by naming the kind of professional who can help further "
    "(doctor, nurse, pharmacist or local clinic).\n"
    "6. If information is missing, ask one short clarifying question "
    "instead of guessing."
)

CLINICIAN_KEYWORDS = (
    "doctor",
    "clinician",
    "physician",
    "nurse",
    "pharmacist",
    "hospital",
    "clinic",
    "emergency",
    "医生",
    "医师",
    "护士",
    "药师",
    "就医",
    "医院",
    "急诊",
)

SAFETY_NOTE = (
    "This is general health information, not a diagnosis or a prescription. "
    "Please talk to a qualified clinician (doctor, nurse or pharmacist) about "
    "your own situation, and contact local emergency services if symptoms are "
    "severe or getting worse."
)


def build_messages(question: str, history=None):
    """Build the OpenAI-style message list sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, max_new_tokens: int = MAX_NEW_TOKENS):
    """Build the request body for the OpenAI-compatible endpoint."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def build_request(base_url: str, payload):
    """Build the HTTP request without sending it (kept separate for tests)."""
    url = base_url.rstrip("/") + "/chat/completions"
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def extract_reply(response) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Unexpected response shape from the Spark endpoint")
    content = (content or "").strip()
    if not content:
        raise ValueError("Spark returned an empty reply")
    return content


def with_safety_note(reply: str) -> str:
    """Append the clinician referral note when the reply omits one."""
    lowered = (reply or "").lower()
    if any(keyword in lowered for keyword in CLINICIAN_KEYWORDS):
        return reply
    if SAFETY_NOTE in reply:
        return reply
    return reply.rstrip() + "\n\n" + SAFETY_NOTE


class SparkClient:
    """Tiny client for a locally hosted Spark-X2.6-4B endpoint."""

    def __init__(self, base_url: str = None, timeout: int = REQUEST_TIMEOUT):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def ask(self, question: str, history=None) -> str:
        payload = build_payload(build_messages(question, history))
        request = build_request(self.base_url, payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return with_safety_note(extract_reply(body))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    client = SparkClient()

    print("Spark-X2.6-4B health information demo")
    print("endpoint: %s" % client.base_url)
    print("question: %s" % question)
    print("-" * 60)

    try:
        answer = client.ask(question)
    except urllib.error.URLError as exc:
        print("Could not reach the Spark endpoint: %s" % exc)
        print("Start a local OpenAI-compatible Spark-X2.6-4B server and set "
              "SPARK_BASE_URL, then run this script again.")
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        print("Bad response from the Spark endpoint: %s" % exc)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

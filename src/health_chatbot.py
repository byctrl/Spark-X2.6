"""Minimal health-information chatbot on top of Spark-X2.6-4B.

Public-interest demo for the HER Hack-Astron #7 "AI for Good" challenge:
inclusive, plain-language patient education. The assistant only explains
general health information; it never diagnoses, prescribes, or replaces a
clinician. The model is reached through a local OpenAI-compatible
Spark-X2.6-4B endpoint (SGLang / vLLM / llama.cpp / Spark-MLX-LLM).
No third-party dependencies.

Usage:
    python src/health_chatbot.py                      # demo question
    python src/health_chatbot.py --question "..."     # custom question
    python src/health_chatbot.py --offline --json     # no model call

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   optional bearer token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "Anda adalah asisten edukasi pasien yang inklusif dan berbahasa sederhana. "
    "Tugas Anda: menjelaskan informasi kesehatan umum dengan kalimat pendek, "
    "mudah dipahami, dan bebas jargon; sebutkan kapan harus mencari bantuan. "
    "Batasan wajib: (1) JANGAN memberi diagnosis, resep, dosis obat, atau "
    "menggantikan klinisi; (2) JANGAN menebak hasil pemeriksaan; (3) selalu "
    "menganjurkan pengguna berkonsultasi dengan dokter atau tenaga kesehatan "
    "profesional; (4) untuk tanda bahaya (sesak napas berat, nyeri dada, "
    "penurunan kesadaran, perdarahan hebat) anjurkan mencari layanan darurat "
    "segera; (5) hormati privasi: jangan meminta data pribadi yang tidak perlu. "
    "Jawab dalam bahasa yang dipakai pengguna."
)

DEMO_QUESTION = (
    "Saya batuk dan demam sejak tiga hari lalu setelah pulang dari perjalanan. "
    "Apa yang sebaiknya saya perhatikan?"
)

OFFLINE_NOTE = (
    "[MODE OFFLINE] Spark-X2.6-4B tidak terhubung; teks di bawah ini adalah "
    "panduan umum statis, bukan keluaran model."
)

OFFLINE_ANSWER = (
    "Panduan umum: istirahat cukup, minum air, pantau suhu dan keluhan Anda, dan "
    "pakai masker bila batuk di dekat orang lain. Segera cari bantuan medis bila "
    "muncul sesak napas, nyeri dada, kebingungan, atau keluhan yang makin berat. "
    "Untuk penilaian dan pengobatan, hubungi dokter atau tenaga kesehatan "
    "profesional terdekat."
)


class ChatbotError(RuntimeError):
    """Raised when the Spark-X2.6-4B endpoint cannot be used."""


def build_messages(question, history=None):
    """Build the bounded chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need role user/assistant and non-empty content")
        messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None):
    """Build the OpenAI-compatible request body for the local Spark server."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_content(data):
    """Read and validate the assistant text from a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatbotError("unexpected response shape from model server") from exc
    if not isinstance(content, str) or not content.strip():
        raise ChatbotError("model returned an empty answer")
    return content.strip()


def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    api_key = os.environ.get("SPARK_API_KEY")
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(question, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT, requester=None):
    """Send one bounded question to Spark-X2.6-4B and return its answer."""
    requester = requester or _post_json
    url = base_url.rstrip("/") + "/chat/completions"
    payload = build_payload(question)
    try:
        data = requester(url, payload, timeout)
    except ChatbotError:
        raise
    except Exception as exc:  # network, timeout, bad JSON, ...
        raise ChatbotError("model server request failed: {}".format(exc)) from exc
    return extract_content(data)


def answer(question, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT, requester=None, offline=False):
    """Return a model answer, or the static safe fallback when unavailable."""
    if offline:
        return OFFLINE_NOTE + "\n\n" + OFFLINE_ANSWER
    try:
        return chat(question, base_url=base_url, timeout=timeout, requester=requester)
    except ChatbotError as exc:
        print("warning: {}".format(exc), file=sys.stderr)
        return OFFLINE_NOTE + "\n\n" + OFFLINE_ANSWER


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information demo"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the model call and print the static fallback",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON record")
    args = parser.parse_args(argv)

    text = answer(
        args.question,
        base_url=args.base_url,
        timeout=args.timeout,
        offline=args.offline,
    )
    if args.json:
        record = {"model": MODEL_ID, "question": args.question, "answer": text}
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

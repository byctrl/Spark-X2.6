"""Minimal Spark-X2.6-4B health-education chatbot.

Demo application for the HER Hack-Astron #7 AI for Good challenge.
It answers plain-language health questions with a locally served
Spark-X2.6-4B model exposed through an OpenAI-compatible endpoint.

The assistant only explains and orients: it never diagnoses, never
prescribes, and always points the user to a licensed clinician.

Usage (model already served locally, e.g. SGLang / vLLM / MLX server):
    python src/health_chatbot.py
    python src/health_chatbot.py --question "..." --base-url http://127.0.0.1:30000/v1
    python src/health_chatbot.py --show-prompt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = (
    "Anda adalah asisten edukasi kesehatan untuk masyarakat umum. "
    "Tugas Anda hanya menjelaskan informasi kesehatan dalam bahasa sederhana "
    "yang mudah dipahami orang tanpa latar belakang medis.\n"
    "Aturan wajib:\n"
    "1. Jangan pernah memberi diagnosis, resep, dosis, atau instruksi pengobatan.\n"
    "2. Jangan menyimpulkan bahwa gejala pengguna pasti suatu penyakit tertentu.\n"
    "3. Jelaskan penyebab umum secara umum, tanda bahaya (red flags) yang perlu "
    "diperiksa, dan langkah aman yang dapat dilakukan sendiri.\n"
    "4. Selalu arahkan pengguna ke dokter, klinik, atau tenaga kesehatan "
    "profesional untuk pemeriksaan dan keputusan medis.\n"
    "5. Jika muncul tanda darurat (sesak napas berat, nyeri dada, kebingungan, "
    "pendarahan hebat, kejang), sarankan segera menghubungi layanan darurat "
    "setempat.\n"
    "6. Jangan meminta data pribadi yang dapat mengidentifikasi pengguna.\n"
    "7. Jawab ringkas, maksimal sekitar 250 kata."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, setelah pulang "
    "dari perjalanan ke luar kota. Apa yang sebaiknya saya lakukan?"
)

SAFETY_FOOTER = (
    "Catatan: jawaban ini hanya untuk edukasi, bukan diagnosis. "
    "Konsultasikan ke dokter atau tenaga kesehatan profesional."
)


def build_messages(question: str, history=None) -> list:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and cannot be overridden by history.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need role user|assistant and non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question: str, history=None, model: str = MODEL_ID,
                  max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Build the OpenAI-compatible chat-completions request body."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }


def extract_reply(response_json) -> str:
    """Read the assistant text out of a chat-completions response."""
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected chat-completions response") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty reply")
    return content.strip()


def ask(question: str, base_url: str = DEFAULT_BASE_URL, history=None,
        timeout: int = TIMEOUT_SECONDS) -> str:
    """Send one question to the locally served Spark-X2.6-4B model."""
    payload = json.dumps(build_payload(question, history)).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return extract_reply(json.loads(body))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-education chatbot demo (HER Hack-Astron #7)"
    )
    parser.add_argument(
        "--question",
        default=DEMO_QUESTION,
        help="question to ask; defaults to the built-in demo question",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL of the local Spark-X2.6-4B server",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the system prompt and exit",
    )
    args = parser.parse_args(argv)

    if args.show_prompt:
        print(SYSTEM_PROMPT)
        return 0

    print("Asisten edukasi kesehatan Spark-X2.6-4B")
    print(f"Pertanyaan: {args.question}\n")

    try:
        answer = ask(args.question, base_url=args.base_url)
    except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
        print(f"Gagal menghubungi model di {args.base_url}: {exc}", file=sys.stderr)
        print(
            "Jalankan Spark-X2.6-4B dengan endpoint OpenAI-compatible "
            "(mis. SGLang/vLLM/MLX server) lalu coba lagi.",
            file=sys.stderr,
        )
        return 1

    print(answer)
    print()
    print(SAFETY_FOOTER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

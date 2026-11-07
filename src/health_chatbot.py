"""Minimal health-knowledge chatbot demo for HER Hack-Astron #7.

The application sends a plain-language question to Spark-X2.6-4B and returns a
patient-education style answer. It is deliberately bounded:

* it never diagnoses, prescribes or replaces a clinician;
* emergency wording short-circuits to a call-emergency-services notice;
* input length is capped so one question cannot dominate the context window.

Run locally (downloads the model on first use):

    python src/health_chatbot.py "Saya demam tiga hari, apa yang perlu saya pantau?"

Importing this module does not require ``transformers``: the model is loaded
lazily, so the unit tests run without a GPU or model weights.
"""

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_INPUT_CHARS = 2000

SYSTEM_PROMPT = (
    "Anda adalah asisten edukasi kesehatan untuk masyarakat umum. Jawab dalam bahasa yang "
    "sama dengan pertanyaan pengguna, dengan kalimat pendek dan sederhana.\n"
    "Aturan:\n"
    "1. Berikan informasi edukatif umum: tanda bahaya, langkah mandiri yang aman, dan jenis "
    "layanan kesehatan yang sesuai.\n"
    "2. JANGAN memberi diagnosis, resep, atau dosis obat tertentu.\n"
    "3. JANGAN menebak hasil pemeriksaan dan jangan menggantikan klinisi.\n"
    "4. Sarankan pengguna menghubungi dokter, bidan, perawat, atau fasilitas kesehatan untuk "
    "keputusan medis, dan sebutkan tanda bahaya yang perlu segera ditangani.\n"
    "5. Jika informasi kurang, ajukan satu atau dua pertanyaan lanjutan yang relevan.\n"
    "6. Jangan meminta nama, nomor telepon, NIK, atau data pribadi sensitif.\n"
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam 38.2 derajat sejak tiga hari setelah pulang dari luar kota. "
    "Saya tidak punya asuransi dan bingung harus ke mana. Informasi apa yang perlu saya "
    "siapkan sebelum menghubungi klinik?"
)

EMERGENCY_KEYWORDS = (
    "nyeri dada",
    "sesak napas",
    "sulit bernapas",
    "tidak sadarkan diri",
    "pingsan",
    "kejang",
    "perdarahan hebat",
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "unconscious",
    "seizure",
    "heavy bleeding",
)

EMERGENCY_NOTICE = (
    "Ini bisa jadi keadaan darurat. Hubungi layanan gawat darurat setempat (mis. 119 atau 112) "
    "atau segera menuju instalasi gawat darurat terdekat. / This may be an emergency: call your "
    "local emergency number or go to the nearest emergency department now."
)

FALLBACK_REPLY = (
    "Maaf, saya belum bisa menyusun jawaban yang aman untuk pertanyaan ini. Silakan hubungi "
    "tenaga kesehatan terdekat atau fasilitas kesehatan untuk mendapatkan arahan langsung."
)

_GENERATOR = None


def _normalize_question(question):
    """Collapse whitespace, reject empty input, and cap the question length."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    text = " ".join(question.split())
    if not text:
        raise ValueError("question must not be empty")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS].rstrip() + " ..."
    return text


def is_emergency(question):
    """Cheap keyword check so emergency wording never waits on the model."""
    lowered = question.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def build_messages(question):
    """Build the chat messages sent to Spark-X2.6-4B."""
    text = _normalize_question(question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _load_generator():
    """Load Spark-X2.6-4B locally and return a messages -> text callable."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )

    def _generate(messages):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


def get_generator():
    """Return the cached model callable, loading it on first use."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = _load_generator()
    return _GENERATOR


def generate_reply(question, generator=None):
    """Answer one health question.

    ``generator`` is a ``messages -> text`` callable; tests inject a stub so no
    model weights are needed. Emergency wording bypasses the model entirely.
    """
    text = _normalize_question(question)

    if is_emergency(text):
        return {
            "reply": EMERGENCY_NOTICE,
            "emergency": True,
            "used_model": False,
            "model": MODEL_ID,
        }

    messages = build_messages(text)
    generate = generator if generator is not None else get_generator()
    reply = (generate(messages) or "").strip() or FALLBACK_REPLY

    return {
        "reply": reply,
        "emergency": False,
        "used_model": True,
        "model": MODEL_ID,
    }


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args) if args else DEMO_QUESTION
    result = generate_reply(question)
    print("Q: " + question)
    print()
    print(result["reply"])
    if result["emergency"]:
        print()
        print("(tangga darurat ditampilkan tanpa memanggil model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

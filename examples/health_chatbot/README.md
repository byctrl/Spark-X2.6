# Health Information Chatbot - Spark-X2.6-4B

Minimal working entry for **HER Hack-Astron #7 | Spark-X2.6 Open Source AI for Good**.

**Public-interest need:** accessible health information. People often receive
instructions they cannot read or interpret; this prototype answers a plain
question in plain language and tells them what to prepare before seeing a
clinician.

## What Spark-X2.6-4B does here

The application sends a safety-bounded system prompt plus the user's question to
`XHToken/Spark-X2.6-4B` (served locally through an OpenAI-compatible endpoint)
and shows the model's plain-language answer. The model is the reasoning layer:
no lookup table or rules produce the explanation.

**Safety boundary:** the assistant does not diagnose, does not recommend
prescription drugs or dosages, flags emergency warning signs, and always
directs the person to a qualified professional. A human clinician stays in the
loop.

## Run it

Step 1 - serve the model locally with any OpenAI-compatible server, for example:

    python -m vllm.entrypoints.openai.api_server --model XHToken/Spark-X2.6-4B --port 30000

Step 2 - ask the demo question:

    python examples/health_chatbot/health_chatbot.py

Step 3 - ask your own question:

    python examples/health_chatbot/health_chatbot.py "What is a fever?" --base-url http://127.0.0.1:30000/v1

Configuration:

| Setting | Default |
| --- | --- |
| `SPARK_BASE_URL` or `--base-url` | `http://127.0.0.1:30000/v1` |
| `--model` | `XHToken/Spark-X2.6-4B` |
| `MAX_NEW_TOKENS` | `1024` |

No third-party Python packages are required (standard library only).

## Demo scenario

Input (`DEMO_QUESTION`):

> I came back from a trip two weeks ago and now I have a dry cough and a mild
> fever in the evening. What should I know before I talk to a doctor?

Expected output shape: a short general explanation, a list of warning signs
that need urgent care, which professional to contact, and a reminder that this
is general information rather than a medical assessment.

## Tests

    python -m unittest discover -s examples/health_chatbot -p "test_*.py"

## Limitations

- Not a medical device and not a substitute for clinical advice.
- Quality depends on the served quantisation and on the endpoint configuration.
- The prompt is English-only in this minimal prototype.

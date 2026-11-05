'''Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: plain-language patient education for people who cannot
easily reach a clinician. It never diagnoses or prescribes; it explains and
points the person to professional care.

Run against a local OpenAI-compatible Spark server:

    python src/health_chatbot.py I have had a cough and fever for 3 days

Environment:
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    default EMPTY
    SPARK_TIMEOUT    default 120 (seconds)
'''

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = 'XHToken/Spark-X2.6-4B'
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_BASE_URL = 'http://127.0.0.1:30000/v1'

SYSTEM_PROMPT = (
    'You are a patient-education assistant. Explain health information in plain, '
    'respectful language that a person with no medical training can understand.\n'
    'Rules you must follow:\n'
    '1. Do not diagnose, do not name a likely disease as a conclusion, and do not '
    'prescribe medicines, doses, or treatments.\n'
    '2. Explain what the described signs usually mean in general terms, and which '
    'warning signs mean the person should seek care urgently.\n'
    '3. Ask at most two short clarifying questions if key details are missing.\n'
    '4. Always tell the person to confirm with a qualified clinician, and mention '
    'that this is general education, not medical advice.\n'
    '5. Never ask for names, ID numbers, addresses, or other identifying data.\n'
    '6. Reply in the language the person used.'
)

DEMO_QUESTION = (
    'I came back from a trip 5 days ago and now I have a dry cough and a fever '
    'of 38.2 C. What should I pay attention to?'
)

DISCLAIMER = (
    'This is general health education, not medical advice. '
    'Please confirm with a qualified clinician.'
)


def build_messages(question, history=None):
    '''Return the chat payload: system prompt first, current question last.'''
    question = (question or '').strip()
    if not question:
        raise ValueError('question must not be empty')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get('role')
        content = (turn.get('content') or '').strip()
        if role in ('user', 'assistant') and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': question})
    return messages


def request_payload(messages):
    return {
        'model': MODEL_ID,
        'messages': messages,
        'max_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
    }


def parse_reply(payload):
    '''Extract the assistant text from an OpenAI-compatible response.'''
    try:
        content = payload['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('unexpected response shape: %r' % (payload,)) from exc
    text = (content or '').strip()
    if not text:
        raise ValueError('model returned an empty answer')
    return text


def with_disclaimer(text):
    text = (text or '').strip()
    if DISCLAIMER.lower() in text.lower():
        return text
    return text + '\n\n' + DISCLAIMER


def chat(question, history=None, base_url=None, api_key=None, timeout=None):
    '''Send one question to Spark-X2.6-4B and return text plus disclaimer.'''
    base_url = (base_url or os.environ.get('SPARK_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
    api_key = api_key or os.environ.get('SPARK_API_KEY') or 'EMPTY'
    if timeout is None:
        timeout = float(os.environ.get('SPARK_TIMEOUT') or 120)
    else:
        timeout = float(timeout)
    messages = build_messages(question, history)
    body = json.dumps(request_payload(messages)).encode('utf-8')
    request = urllib.request.Request(
        base_url + '/chat/completions',
        data=body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + api_key,
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:500]
        raise RuntimeError(
            'Spark endpoint returned HTTP %s: %s' % (exc.code, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            'Cannot reach Spark-X2.6-4B at %s. Start an OpenAI-compatible server '
            'or set SPARK_BASE_URL.' % base_url
        ) from exc
    return with_disclaimer(parse_reply(payload))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = ' '.join(argv).strip() or DEMO_QUESTION
    print('Q: ' + question)
    print()
    print(chat(question))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Optional B13 live probe: does a bare /api/chat/completions persist a chat?

This calls a *real* LLM on the configured Open WebUI instance, so is opt-in.
Only use a synthetic test user/account and test model. It reports differences
in that user's saved chat IDs, not the contents of any other user's chats.
"""
import argparse
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import uuid


def req(base, token, method, path, payload=None):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = Request(base.rstrip('/') + path, data=data, method=method, headers={
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    })
    with urlopen(request, timeout=60) as resp:
        return json.load(resp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:3000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--confirm-model-call', action='store_true')
    args = parser.parse_args()
    if not args.confirm_model_call:
        parser.error('Explicitly pass --confirm-model-call: this may incur LLM API costs')
    token = os.getenv('AVATAR_STUDENT_API_KEY', '')
    if not token:
        parser.error('AVATAR_STUDENT_API_KEY must contain a synthetic student token')
    marker = 'B13-synthetic-probe-' + str(uuid.uuid4())
    try:
        before = req(args.base_url, token, 'GET', '/api/v1/chats/')
        pre = {r['id'] for r in before if isinstance(r, dict) and 'id' in r}
        result = req(args.base_url, token, 'POST', '/api/chat/completions', {
            'model': args.model,
            'stream': False,
            'messages': [{'role': 'user', 'content': marker + ' Please reply with OK.'}],
        })
        after = req(args.base_url, token, 'GET', '/api/v1/chats/')
        post = {r['id'] for r in after if isinstance(r, dict) and 'id' in r}
        print('Completion returned:', bool(result.get('choices')))
        print('New chats in same student account:', sorted(post-pre))
        print('If empty, bare completions did not create a saved chat in this test.')
        print('If not empty, inspect the new chat manually before concluding it came from this request.')
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        print(f'Probe failed: {type(exc).__name__} (check model and student credentials)', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

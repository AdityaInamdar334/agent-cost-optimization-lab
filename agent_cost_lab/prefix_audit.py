"""Audit canonical prefix fingerprints without echoing sensitive prompt text.

Sample JSON should contain `messages` and optional `tools`. Supply the number
of leading messages that are intentionally stable. These are exact-serialization
fingerprints in this lab, not the provider's true token-prefix hash.
"""
import argparse
import json
from pathlib import Path
from .cache_experiment import prefix_key


def inspect_request(request: dict, stable_messages: int, tenant: str,
                    model: str) -> dict:
    messages = request.get('messages', [])
    if not isinstance(messages, list) or stable_messages < 0 or stable_messages > len(messages):
        raise ValueError('Invalid stable_messages')
    tools = request.get('tools', [])
    stable = {'tools': tools, 'messages': messages[:stable_messages]}
    full = {'tools': tools, 'messages': messages}
    return {'stable_fingerprint': prefix_key(tenant, model, stable).split(':')[-1],
            'full_fingerprint': prefix_key(tenant, model, full).split(':')[-1],
            'stable_messages': stable_messages,
            'total_messages': len(messages),
            'warning': 'Hashes are only local audit signals; no provider cache hit is guaranteed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('requests', nargs='+', type=Path)
    parser.add_argument('--stable-messages', type=int, default=1)
    parser.add_argument('--tenant', default='local')
    parser.add_argument('--model', default='example-model')
    args = parser.parse_args()
    for path in args.requests:
        request = json.loads(path.read_text(encoding='utf-8'))
        print(json.dumps({'file': str(path), **inspect_request(
            request, args.stable_messages, args.tenant, args.model)}, indent=2))


if __name__ == '__main__':
    main()

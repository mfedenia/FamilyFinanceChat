"""Require a fresh chat response-time push after a smoke test began."""

import argparse
import json
import math
import re
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\[\\"n])*)"(?:,|$)')
SAMPLE = re.compile(r'(openwebui_chat_completion_seconds|push_time_seconds)\{(.*)\}\s+(\S+)(?:\s+\S+)?\s*')


def fresh_chat_metric(exposition, job, since, model=None):
    """Match a finite chat sample to a fresh push in the same grouping key.

    Pushgateway adds job/instance to every sample; other push_time labels are
    grouping labels too. This is group freshness, not per-request attribution.
    """
    chats = []
    pushes = []
    for line in exposition.splitlines():
        sample = SAMPLE.fullmatch(line)
        if sample is None:
            continue
        name, raw_labels, raw_value = sample.groups()
        labels = {}
        pos = 0
        for label in LABEL.finditer(raw_labels):
            if label.start() != pos or label[1] in labels:
                break
            labels[label[1]] = json.loads('"' + label[2] + '"')
            pos = label.end()
        if pos != len(raw_labels) or labels.get("job") != job or "instance" not in labels:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if name == "push_time_seconds" and value > since:
            pushes.append(labels)
        elif name == "openwebui_chat_completion_seconds" and value >= 0:
            if model is None or labels.get("model") == model:
                chats.append(labels)
    return any(all(chat.get(key) == value for key, value in group.items())
               for group in pushes for chat in chats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Pushgateway base URL")
    parser.add_argument("--job", default="openwebui_chat_metrics")
    parser.add_argument("--model", help="Require this model label on the chat sample")
    parser.add_argument("--since", required=True, type=float, help="Unix time recorded before the chat")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(args.url.rstrip("/") + "/metrics", timeout=5) as response:
                body = response.read().decode("utf-8")
            if fresh_chat_metric(body, args.job, args.since, args.model):
                print("Fresh chat response-time metric found")
                return 0
        except (OSError, URLError, UnicodeDecodeError):
            pass
        time.sleep(2)
    print("No fresh chat response-time metric appeared in Pushgateway", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

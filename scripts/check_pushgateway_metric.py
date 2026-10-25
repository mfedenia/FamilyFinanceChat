"""Require a fresh chat response-time push after a smoke test began."""

import argparse
import re
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def fresh_chat_metric(exposition, job, since):
    pushed_at = None
    chat_metric = False
    for line in exposition.splitlines():
        if line.startswith("openwebui_chat_completion_seconds{"):
            chat_metric = True
        if not line.startswith("push_time_seconds{"):
            continue
        labels, _, value = line.partition("} ")
        if re.search(r'(?:^|,)job="' + re.escape(job) + r'"(?:,|$)', labels.split("{", 1)[1]):
            try:
                pushed_at = float(value)
            except ValueError:
                pass
    return chat_metric and pushed_at is not None and pushed_at > since


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Pushgateway base URL")
    parser.add_argument("--job", default="openwebui_chat_metrics")
    parser.add_argument("--since", required=True, type=float, help="Unix time recorded before the chat")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(args.url.rstrip("/") + "/metrics", timeout=5) as response:
                body = response.read().decode("utf-8")
            if fresh_chat_metric(body, args.job, args.since):
                print("Fresh chat response-time metric found")
                return 0
        except (OSError, URLError, UnicodeDecodeError):
            pass
        time.sleep(2)
    print("No fresh chat response-time metric appeared in Pushgateway", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

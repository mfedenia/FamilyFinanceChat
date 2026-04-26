#!/usr/bin/env python3
"""
Convert openwebui_history.json (Prometheus range-query matrix result) to OpenMetrics
text format for backfilling into Prometheus TSDB via:

  promtool tsdb create-blocks-from openmetrics <input> --output-dir <dir>

Output: backfill_history.openmetrics (written next to this script's repo root)
"""

import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, "openwebui_history.json")
OUTPUT_FILE = os.path.join(REPO_ROOT, "backfill_history.openmetrics")


def escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {INPUT_FILE} ...")
    with open(INPUT_FILE) as f:
        data = json.load(f)

    if data.get("status") != "success":
        print("ERROR: JSON status is not 'success'", file=sys.stderr)
        sys.exit(1)

    results = data["data"]["result"]
    print(f"  {len(results)} time series found")

    # Group by metric name so each family is written contiguously — required by
    # the OpenMetrics spec and expected by promtool.
    metric_families: dict[str, list] = defaultdict(list)
    for series in results:
        name = series["metric"].get("__name__", "__unknown__")
        labels = {k: v for k, v in series["metric"].items() if k != "__name__"}
        label_str = ""
        if labels:
            pairs = ",".join(
                f'{k}="{escape_label_value(v)}"' for k, v in sorted(labels.items())
            )
            label_str = f"{{{pairs}}}"
        metric_families[name].append((label_str, series["values"]))

    print(f"  {len(metric_families)} unique metric names")
    print(f"Writing OpenMetrics to {OUTPUT_FILE} ...")

    total = 0
    # 4 MB write buffer — reduces syscall overhead on 5.8 M samples
    with open(OUTPUT_FILE, "w", buffering=4 * 1024 * 1024) as out:
        for metric_name in sorted(metric_families):
            # Use 'unknown' type for every metric so we avoid OpenMetrics counter
            # naming rules (counter families must NOT have _total in the family name,
            # but our __name__ values already include it).
            out.write(f"# HELP {metric_name} .\n")
            out.write(f"# TYPE {metric_name} unknown\n")
            for label_str, values in metric_families[metric_name]:
                for ts, val in values:
                    out.write(f"{metric_name}{label_str} {val} {float(ts):.3f}\n")
                    total += 1
        out.write("# EOF\n")

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"Done: {total:,} samples written ({size_mb:.1f} MB)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

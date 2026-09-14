"""Summarise one GKE fine-tuning Job as a timeline: cluster overhead first, then every command.

Called by finetune.sh with the Job creation time, the GPU node's creation time, the pod log (PHASE lines) and the
pod's events. Standard library only. Prints JSON; no project, node or pod names are included.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def pull_seconds(message: str) -> float | None:
    """ "Successfully pulled image ... in 1m2.345s (1m2.345s including waiting)" -> 62.345"""
    match = re.search(r" in ((?:(\d+)h)?(?:(\d+)m)?([\d.]+)(ms|s))", message)
    if not match:
        return None
    hours, minutes, value, unit = match.group(2), match.group(3), float(match.group(4)), match.group(5)
    seconds = value / 1000 if unit == "ms" else value
    return seconds + 60 * int(minutes or 0) + 3600 * int(hours or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--created", required=True)
    ap.add_argument("--node-created", required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--commands-file", type=Path, required=True, help="one ltm-ft argument string per line")
    args = ap.parse_args()

    created = parse(args.created)
    phases = {
        m.group(1): parse(m.group(2))
        for m in re.finditer(r"^PHASE (\S+) (\S+)$", args.log.read_text(), flags=re.MULTILINE)
    }
    events = json.loads(args.events.read_text())["items"]

    def first(reason: str) -> dict[str, object] | None:
        matching = [e for e in events if e.get("reason") == reason]
        return min(matching, key=lambda e: e.get("firstTimestamp") or e["eventTime"]) if matching else None

    def at(event: dict[str, object] | None) -> datetime | None:
        if event is None:
            return None
        return parse(str(event.get("firstTimestamp") or event.get("eventTime")))

    def since(t: datetime | None) -> float | None:
        return None if t is None else round((t - created).total_seconds(), 1)

    pulled = first("Pulled")
    started = phases.get("container_started")
    commands = []
    for i, command in enumerate(args.commands_file.read_text().splitlines(), start=1):
        start, end = phases.get(f"command_{i}_start"), phases.get(f"command_{i}_end")
        seconds = None if start is None or end is None else round((end - start).total_seconds(), 1)
        commands.append({"command": f"ltm-ft {command}", "seconds": seconds})

    timeline = {
        "job": args.job,
        "gpu": "1x NVIDIA L4 (g2-standard-8, GKE node pool scaling from zero)",
        "seconds_since_job_created": {
            "scale_up_triggered": since(at(first("TriggeredScaleUp"))),
            "gpu_node_created": since(parse(args.node_created)),
            "pod_scheduled": since(at(first("Scheduled"))),
            "image_pull_started": since(at(first("Pulling"))),
            "image_pulled": since(at(pulled)),
            "container_started": since(started),
            "all_commands_done": since(phases.get("all_done")),
        },
        "image_pull_seconds": pull_seconds(str(pulled.get("message", ""))) if pulled else None,
        "commands": commands,
    }
    print(json.dumps(timeline, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


UPSTREAM = {
    "repo": "https://github.com/zhehangdu/Newton-Muon",
    "sha": "df78af0db523d8bceb25af4919a3e3e7082b80f3",
}


def cmd_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args()

    record = {
        "run_id": args.run_id,
        "method": args.method,
        "mode": args.mode,
        "launcher": args.launcher,
        "source": args.source,
        "git_sha": args.git_sha,
        "started_at": args.started_at,
        "upstream": UPSTREAM,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": cmd_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()


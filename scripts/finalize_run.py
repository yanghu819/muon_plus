#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PAPER = {
    "adam1": {"loss": 3.3801, "time_s": 7228.4, "label": "Newton-Muon-1 AdamW"},
    "muon1": {"loss": 3.2793, "time_s": 7314.1, "label": "Newton-Muon-1 Muon baseline"},
    "newton_muon1": {"loss": 3.2611, "time_s": 7443.3, "label": "Newton-Muon-1 Newton-Muon"},
    "adam2": {"loss": 3.4628, "time_s": 4272.0, "label": "Newton-Muon-2 AdamW avg"},
    "muon2": {"loss": 3.2787, "time_s": 4305.9, "label": "Newton-Muon-2 Muon baseline avg"},
    "newton_muon2": {"loss": 3.2739, "time_s": 4342.4, "label": "Newton-Muon-2 Newton-Muon avg"},
}


VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) train_time:([0-9.]+)ms")
PEAK_RE = re.compile(r"peak memory consumption: ([0-9]+) MiB")


def parse_stdout(path: Path) -> dict[str, float | int | None]:
    final_val = None
    final_step = None
    total_steps = None
    train_time_ms = None
    peak_mib = None
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if match := VAL_RE.search(line):
            final_step = int(match.group(1))
            total_steps = int(match.group(2))
            final_val = float(match.group(3))
            train_time_ms = float(match.group(4))
        if match := PEAK_RE.search(line):
            peak_mib = int(match.group(1))
    return {
        "final_step": final_step,
        "total_steps": total_steps,
        "final_val_loss": final_val,
        "train_time_s": None if train_time_ms is None else train_time_ms / 1000.0,
        "peak_memory_mib": peak_mib,
    }


def append_leaderboard(repo: Path, record: dict) -> None:
    path = repo / "leaderboard.csv"
    fields = [
        "run_id",
        "method",
        "mode",
        "status",
        "git_sha",
        "final_val_loss",
        "paper_loss",
        "loss_delta",
        "train_time_s",
        "paper_time_s",
        "started_at",
        "finished_at",
    ]
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("run_id") != record["run_id"]]
    rows.append({key: record.get(key, "") for key in fields})
    rows.sort(key=lambda row: row.get("started_at", ""))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(repo: Path, record: dict) -> None:
    exp_dir = repo / "experiments"
    exp_dir.mkdir(exist_ok=True)
    paper_loss = record.get("paper_loss")
    loss_delta = record.get("loss_delta")
    mode = record.get("mode")
    status = record.get("status")
    final_loss = record.get("final_val_loss")
    if mode == "smoke":
        decision = "Use this run only as an infrastructure check; tiny-data losses are not comparable to the paper."
    elif status != "completed":
        decision = "Do not compare this run to the paper until the full training path completes successfully."
    elif final_loss is not None and paper_loss is not None and loss_delta is not None:
        if abs(float(loss_delta)) <= 0.003:
            decision = "Treat this full run as reproduced on validation loss. Wall time remains hardware-sensitive and secondary."
        else:
            decision = "This full run completed but did not reproduce the paper validation loss; investigate optimizer, resume, or environment differences before using it as a trusted baseline."
    else:
        decision = "Use full runs for paper comparison; timing is hardware-sensitive, so validation loss is the primary comparison target."
    lines = [
        f"# {record['run_id']}",
        "",
        f"- method: {record['method']}",
        f"- mode: {record['mode']}",
        f"- status: {record['status']}",
        f"- git_sha: {record['git_sha']}",
        f"- source: {record['source']}",
        f"- launcher: {record['launcher']}",
        f"- started_at: {record['started_at']}",
        f"- finished_at: {record['finished_at']}",
        "",
        "## Hypothesis",
        "",
        "This run checks whether the imported Newton-Muon baseline code can reproduce the paper-reported single-GPU validation loss under the same script-level hyperparameters.",
        "",
        "## Result",
        "",
        f"- final_val_loss: {record.get('final_val_loss')}",
        f"- paper_loss: {paper_loss}",
        f"- loss_delta: {loss_delta}",
        f"- train_time_s: {record.get('train_time_s')}",
        f"- paper_time_s: {record.get('paper_time_s')}",
        f"- peak_memory_mib: {record.get('peak_memory_mib')}",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Artifacts",
        "",
        f"- raw_run_dir: runs/{record['run_id']}",
        f"- stdout_log: runs/{record['run_id']}/stdout.log",
        f"- source_snapshot: runs/{record['run_id']}/source_snapshot",
    ]
    (exp_dir / f"{record['run_id']}.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--status-code", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo = run_dir.parent.parent
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    parsed = parse_stdout(run_dir / "stdout.log")

    method = metadata["method"]
    paper = PAPER.get(method, {})
    final_loss = parsed.get("final_val_loss")
    paper_loss = paper.get("loss")

    record = dict(metadata)
    record.update(parsed)
    record["status_code"] = args.status_code
    record["status"] = "completed" if args.status_code == 0 else "failed"
    record["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["paper_label"] = paper.get("label")
    record["paper_loss"] = paper_loss
    record["paper_time_s"] = paper.get("time_s")
    record["loss_delta"] = None if final_loss is None or paper_loss is None else final_loss - paper_loss

    metadata_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    append_leaderboard(repo, record)
    write_report(repo, record)
    print(json.dumps({k: record.get(k) for k in ["run_id", "status", "final_val_loss", "loss_delta", "train_time_s"]}, sort_keys=True))


if __name__ == "__main__":
    main()

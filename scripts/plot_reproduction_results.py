#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
LEADERBOARD = REPO / "leaderboard.csv"
OUT = REPO / "figures" / "newton_muon_reproduction_summary.png"

PAPER = {
    "muon1": 3.2793,
    "newton_muon1": 3.2611,
}

RUN_LABELS = {
    "2026-06-29T14-59-07Z-muon1-full-0c5a7ea0": "Muon ours",
    "2026-06-29T23-07-57Z-newton_muon1-full-9c87b12e": "Newton broken resume",
    "2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9": "Newton view fix",
    "2026-07-01T03-02-06Z-newton_muon1-full-resume-final-7d5202c": "Newton offset fix",
}


def load_leaderboard() -> dict[str, dict[str, str]]:
    with LEADERBOARD.open(newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def load_curve(run_id: str) -> list[tuple[int, float]]:
    path = REPO / "runs" / run_id / "stdout.log"
    if not path.exists():
        return []
    vals: list[tuple[int, float]] = []
    pattern = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+)")
    for line in path.read_text(errors="replace").splitlines():
        if match := pattern.search(line):
            vals.append((int(match.group(1)), float(match.group(3))))
    return vals


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    rows = load_leaderboard()
    offset_run = "2026-07-01T03-02-06Z-newton_muon1-full-resume-final-7d5202c"
    muon_run = "2026-06-29T14-59-07Z-muon1-full-0c5a7ea0"

    bars = [
        ("Muon paper", PAPER["muon1"], "#6b7280"),
        (RUN_LABELS[muon_run], float(rows[muon_run]["final_val_loss"]), "#2563eb"),
        ("Newton paper", PAPER["newton_muon1"], "#111827"),
        (RUN_LABELS["2026-06-29T23-07-57Z-newton_muon1-full-9c87b12e"], float(rows["2026-06-29T23-07-57Z-newton_muon1-full-9c87b12e"]["final_val_loss"]), "#d97706"),
        (RUN_LABELS["2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9"], float(rows["2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9"]["final_val_loss"]), "#059669"),
        (RUN_LABELS[offset_run], float(rows[offset_run]["final_val_loss"]), "#dc2626"),
    ]

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(14, 5.6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    labels = [x[0] for x in bars]
    values = [x[1] for x in bars]
    colors = [x[2] for x in bars]
    bar_artists = ax0.bar(range(len(labels)), values, color=colors)
    annotate_bars(ax0, bar_artists)
    ax0.set_title("Final validation loss: paper vs reproduced", fontsize=13, pad=12)
    ax0.set_ylabel("Validation loss")
    ax0.set_ylim(3.255, 3.286)
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, rotation=30, ha="right")
    ax0.axhline(PAPER["newton_muon1"], color="#111827", linestyle="--", linewidth=1, alpha=0.7)
    ax0.grid(axis="y", alpha=0.25)

    muon_ours = float(rows[muon_run]["final_val_loss"])
    newton_ours = float(rows[offset_run]["final_val_loss"])
    ax0.text(
        0.01,
        0.02,
        f"Paper Newton gain vs Muon: {PAPER['newton_muon1'] - PAPER['muon1']:+.4f}\n"
        f"Reproduced offset-fix gain vs Muon: {newton_ours - muon_ours:+.4f}",
        transform=ax0.transAxes,
        fontsize=10,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )

    curve = load_curve(offset_run)
    if curve:
        xs, ys = zip(*curve)
        ax1.plot(xs, ys, marker="o", color="#dc2626", linewidth=2.2, label="Newton offset-fix run")
    ax1.axhline(PAPER["newton_muon1"], color="#111827", linestyle="--", linewidth=1.2, label="Newton paper")
    ax1.axhline(muon_ours, color="#2563eb", linestyle=":", linewidth=1.6, label="Muon ours final")
    ax1.set_title("Last resume segment validation curve", fontsize=13, pad=12)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Validation loss")
    ax1.set_xlim(5450, 6250)
    ax1.set_ylim(3.255, 3.325)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")

    fig.suptitle("Newton-Muon reproduction status on AIStation GPU2", fontsize=15, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()

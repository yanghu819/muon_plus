#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "cosine",
    "norm_ratio",
    "alpha",
    "cov_trace_mean",
    "cov_diag_cond_proxy",
    "inv_diag_mean",
    "inv_diag_max",
]


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, round(q * (len(ys) - 1))))
    return ys[idx]


def fmt(x: float) -> str:
    if x != x:
        return "nan"
    if abs(x) >= 1000 or (abs(x) < 0.001 and x != 0):
        return f"{x:.3e}"
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    path = Path(args.telemetry)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("kind", "unknown")].append(row)

    summary = {}
    for kind, items in sorted(groups.items()):
        summary[kind] = {"count": len(items)}
        for field in FIELDS:
            vals = [float(row[field]) for row in items if field in row]
            if vals:
                summary[kind][field] = {
                    "mean": mean(vals),
                    "p10": quantile(vals, 0.10),
                    "p50": quantile(vals, 0.50),
                    "p90": quantile(vals, 0.90),
                    "min": min(vals),
                    "max": max(vals),
                }

    print(f"rows={len(rows)} path={path}")
    header = "kind count cos_p50 cos_p10 norm_p50 norm_p90 cov_p50 inv_p50 cond_p90"
    print(header)
    for kind, stats in summary.items():
        cosine = stats.get("cosine", {})
        norm = stats.get("norm_ratio", {})
        cov = stats.get("cov_trace_mean", {})
        inv = stats.get("inv_diag_mean", {})
        cond = stats.get("cov_diag_cond_proxy", {})
        print(
            kind,
            stats["count"],
            fmt(cosine.get("p50", float("nan"))),
            fmt(cosine.get("p10", float("nan"))),
            fmt(norm.get("p50", float("nan"))),
            fmt(norm.get("p90", float("nan"))),
            fmt(cov.get("p50", float("nan"))),
            fmt(inv.get("p50", float("nan"))),
            fmt(cond.get("p90", float("nan"))),
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

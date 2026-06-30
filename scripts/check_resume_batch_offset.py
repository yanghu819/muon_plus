#!/usr/bin/env python3
from __future__ import annotations

import argparse


def upstream_state_after_steps(steps: int, accumulation_steps: int) -> tuple[int, int]:
    """Return (prefetched_batch, loader_next_batch) at checkpoint step=N.

    The upstream scripts prefetch one batch, then call train_loader.reset()
    before entering the loop. A resumable materialization has to preserve that
    exact quirk to match an uninterrupted run.
    """

    loader_next = 0
    prefetched = 0
    for _step in range(steps):
        for _micro in range(accumulation_steps):
            prefetched = loader_next
            loader_next += 1
    return prefetched, loader_next


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    args = parser.parse_args()

    expected_prefetch, expected_loader_next = upstream_state_after_steps(
        args.steps, args.accumulation_steps
    )
    seek_index = args.steps * args.accumulation_steps - 1
    if seek_index != expected_prefetch:
        raise SystemExit(
            f"bad resume seek index {seek_index}; expected {expected_prefetch}"
        )
    print(
        "ok: resume seek index "
        f"{seek_index} matches upstream checkpoint step {args.steps}; "
        f"loader next batch is {expected_loader_next}"
    )


if __name__ == "__main__":
    main()

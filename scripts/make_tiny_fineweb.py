#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MAGIC = 20240520


def write_shard(path: Path, ntok: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = ntok
    tokens = rng.integers(0, 50257, size=ntok, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-tokens", type=int, default=16384)
    parser.add_argument("--val-tokens", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    write_shard(out_dir / "fineweb_train_000001.bin", args.train_tokens, 1)
    write_shard(out_dir / "fineweb_val_000000.bin", args.val_tokens, 2)
    print(f"wrote tiny FineWeb shards under {out_dir}")


if __name__ == "__main__":
    main()


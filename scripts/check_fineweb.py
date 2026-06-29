#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MAGIC = 20240520


def read_ntok(path: Path) -> int:
    with path.open("rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if len(header) != 256 or int(header[0]) != MAGIC or int(header[1]) != 1:
        raise SystemExit(f"bad FineWeb shard header: {path}")
    return int(header[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--chunks", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    required = [data_dir / "fineweb_val_000000.bin"]
    required += [data_dir / f"fineweb_train_{i:06d}.bin" for i in range(1, args.chunks + 1)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing FineWeb shards:\n" + "\n".join(missing[:20]))

    total = 0
    for path in required:
        total += read_ntok(path)
    print(f"verified {len(required)} shards in {data_dir} with {total:,} tokens")


if __name__ == "__main__":
    main()


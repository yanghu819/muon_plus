#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


UPSTREAM_SHA = "df78af0db523d8bceb25af4919a3e3e7082b80f3"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def patch_common_paths(text: str, train_pattern: str, val_pattern: str) -> str:
    replacements = [
        ("input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'", f"input_bin : str = {train_pattern!r}"),
        ("input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'", f"input_val_bin : str = {val_pattern!r}"),
        ('train_files = "data/fineweb10B/fineweb_train_*.bin"', f"train_files = {train_pattern!r}"),
        ('val_files = "data/fineweb10B/fineweb_val_*.bin"', f"val_files = {val_pattern!r}"),
    ]
    for old, new in replacements:
        if old in text:
            text = replace_once(text, old, new)
    return text


def patch_smoke_1(text: str) -> str:
    replacements = [
        ("batch_size : int = 8*64", "batch_size : int = 4"),
        ("device_batch_size : int = 64", "device_batch_size : int = 4"),
        ("sequence_length : int = 1024", "sequence_length : int = 128"),
        ("num_iterations : int = 6200", "num_iterations : int = 2"),
        ("warmdown_iters : int = 1800", "warmdown_iters : int = 1"),
        ("val_loss_every : int = 100", "val_loss_every : int = 1"),
        ("val_tokens : int = 10485760", "val_tokens : int = 4096"),
        (
            "GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))",
            "GPT(GPTConfig(vocab_size=num_vocab, n_layer=2, n_head=2, n_embd=128))",
        ),
        ("model = torch.compile(model)", "model = model  # smoke: skip torch.compile startup cost"),
    ]
    for old, new in replacements:
        text = replace_once(text, old, new)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    text = source.read_text()

    if args.mode == "smoke":
        if not args.method.endswith("1"):
            raise SystemExit("smoke mode currently supports the *_1 single-GPU scripts")
        data_root = Path(args.run_dir) / "data" / "fineweb10B"
    else:
        data_root = Path(args.repo_root) / "data" / "fineweb10B"

    text = patch_common_paths(
        text,
        str(data_root / "fineweb_train_*.bin"),
        str(data_root / "fineweb_val_*.bin"),
    )
    if args.mode == "smoke":
        text = patch_smoke_1(text)

    banner = (
        f"# Materialized by scripts/materialize_train.py from {source.name}\n"
        f"# Upstream Newton-Muon SHA: {UPSTREAM_SHA}\n"
        f"# Mode: {args.mode}; method: {args.method}\n\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(banner + text)


if __name__ == "__main__":
    main()

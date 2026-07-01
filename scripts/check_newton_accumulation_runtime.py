#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "train_gpt_newton_muon_1.py"
sys.path.insert(0, str(REPO))


def load_training_defs() -> dict:
    src = SOURCE.read_text()
    end = src.index("# -----------------------------------------------------------------------------\n# Our own simple Distributed Data Loader")
    ns: dict = {"__name__": "newton_accumulation_runtime"}
    exec(src[:end], ns)
    return ns


def xtx_reference(x: torch.Tensor) -> torch.Tensor:
    x2d = x.detach().float().flatten(0, -2)
    return x2d.T @ x2d / float(x2d.size(0))


def xtx_blocks4_reference(x: torch.Tensor) -> torch.Tensor:
    x2d = x.detach().float().flatten(0, -2)
    n, four_d = x2d.shape
    d = four_d // 4
    blocks = x2d.view(n, 4, d).permute(1, 2, 0)
    return torch.bmm(blocks, blocks.transpose(1, 2)) / float(n)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.08)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Newton-Muon accumulation runtime check")

    ns = load_training_defs()
    GPT = ns["GPT"]
    GPTConfig = ns["GPTConfig"]

    torch.manual_seed(1234)
    raw_model = GPT(GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=1,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )).cuda()
    raw_model.train()

    block = raw_model.transformer.h[0]
    captures: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, inputs):
            captures[name] = inputs[0].detach().float()
        return hook

    handles = [
        block.attn.register_forward_pre_hook(capture("qkv")),
        block.attn.c_proj.register_forward_pre_hook(capture("o")),
        block.mlp.c_fc.register_forward_pre_hook(capture("c_fc")),
        block.mlp.c_proj.register_forward_pre_hook(capture("c_proj")),
    ]

    model = torch.compile(raw_model) if args.compiled else raw_model
    idx = torch.randint(0, args.vocab_size, (args.batch, args.seq_len), device="cuda")
    targets = torch.randint(0, args.vocab_size, (args.batch, args.seq_len), device="cuda")
    _, loss = model(idx, targets, return_logits=False, precond_flag=True)
    loss.backward()
    torch.cuda.synchronize()

    for handle in handles:
        handle.remove()

    checks = {
        "qkv": max_abs(block.attn.qkv_xtx_accum, xtx_reference(captures["qkv"])),
        "o": max_abs(block.attn.o_xtx_accum, xtx_reference(captures["o"])),
        "c_fc": max_abs(block.mlp.fc_xtx_accum, xtx_reference(captures["c_fc"])),
        "c_proj": max_abs(block.mlp.proj_xtx_accum, xtx_blocks4_reference(captures["c_proj"])),
    }
    counts = {
        "qkv": float(block.attn.qkv_xtx_count.item()),
        "o": float(block.attn.o_xtx_count.item()),
        "c_fc": float(block.mlp.fc_xtx_count.item()),
        "c_proj": float(block.mlp.proj_xtx_count.item()),
    }

    print(f"compiled={bool(args.compiled)}")
    for key in ("qkv", "o", "c_fc", "c_proj"):
        print(f"{key}_count={counts[key]:.0f} {key}_max_abs_err={checks[key]:.6g}")
        if counts[key] != 1.0:
            raise SystemExit(f"{key} accumulation count mismatch")
        if checks[key] > args.tolerance:
            raise SystemExit(f"{key} accumulation mismatch")


if __name__ == "__main__":
    main()

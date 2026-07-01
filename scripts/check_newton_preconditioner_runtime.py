#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "train_gpt_newton_muon_1.py"


def load_muon_class():
    src = SOURCE.read_text()
    start = src.index("class Muon(torch.optim.Optimizer):")
    end = src.index("# -----------------------------------------------------------------------------\n# PyTorch nn.Module definitions")
    ns = {"torch": torch, "Tensor": Tensor}
    exec(src[start:end], ns)
    return ns["Muon"]


def make_param(kind: str, d: int, device: torch.device) -> torch.nn.Parameter:
    shapes = {
        "qkv": (3 * d, d),
        "o": (d, d),
        "c_fc": (4 * d, d),
        "c_proj": (d, 4 * d),
    }
    p = torch.nn.Parameter(torch.zeros(shapes[kind], device=device))
    if kind == "c_proj":
        accum = torch.zeros((4, d, d), device=device, dtype=torch.float32)
    else:
        accum = torch.zeros((d, d), device=device, dtype=torch.float32)
    p._stats_ref = {
        "kind": kind,
        "d": d,
        "accum": accum,
        "count": torch.zeros((), device=device, dtype=torch.float32),
    }
    return p


def build_optimizer(Muon, d: int, device: torch.device):
    params = [make_param(kind, d, device) for kind in ("qkv", "o", "c_fc", "c_proj")]
    opt = Muon(params, lr=1.0, momentum=0.0)
    opt.attach_preconditioner()
    return opt, params


def set_nonidentity_inverses(opt, params: list[torch.nn.Parameter], d: int) -> list[torch.Tensor]:
    expected: list[torch.Tensor] = []
    for idx, p in enumerate(params):
        st = opt.state[p]
        inv = st["precond_inv_apply"]
        inv.zero_()
        if inv.ndim == 2:
            inv.diagonal().copy_(torch.linspace(1.25 + idx, 2.0 + idx, d, device=inv.device))
        else:
            for block in range(inv.size(0)):
                inv[block].diagonal().copy_(
                    torch.linspace(1.1 + idx + block, 1.8 + idx + block, d, device=inv.device)
                )
        expected.append(inv.detach().clone())
    return expected


def fill_grads(params: list[torch.nn.Parameter]) -> dict[torch.nn.Parameter, torch.Tensor]:
    before: dict[torch.nn.Parameter, torch.Tensor] = {}
    for i, p in enumerate(params):
        values = torch.arange(p.numel(), device=p.device, dtype=torch.float32).reshape_as(p)
        p.grad = (values / 100.0 + 0.5 + i).to(p.dtype)
        before[p] = p.grad.detach().clone().float()
    return before


def manual_apply(grad: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    grad = grad.float()
    inv = inv.float()
    if inv.ndim == 2:
        return grad @ inv
    d = inv.size(-1)
    blocks = grad.view(d, 4, d).permute(1, 0, 2)
    out = torch.bmm(blocks, inv).permute(1, 0, 2).reshape_as(grad)
    return out


def max_error_after_apply(opt, params, expected_inv: list[torch.Tensor]) -> float:
    before = fill_grads(params)
    opt._apply_precond_all_grads_batched_()
    err = 0.0
    for p, inv in zip(params, expected_inv):
        want = manual_apply(before[p], inv)
        got = p.grad.detach().float()
        err = max(err, float((got - want).abs().max().item()))
    return err


def rebuild_preconditioner_views_after_load(opt) -> None:
    saved = []
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            inv = st.get("precond_inv_apply")
            saved.append((p, None if inv is None else inv.detach().clone()))
    opt._precond_ready = False
    opt._refresh_map = []
    opt._refresh_K = None
    opt._apply_plan = None
    opt._finalize_precond_buffers_()
    for p, inv in saved:
        if inv is not None and "precond_inv_apply" in opt.state[p]:
            opt.state[p]["precond_inv_apply"].copy_(inv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    Muon = load_muon_class()

    opt, params = build_optimizer(Muon, args.d, device)
    expected = set_nonidentity_inverses(opt, params, args.d)
    direct_err = max_error_after_apply(opt, params, expected)

    state = opt.state_dict()
    loaded, loaded_params = build_optimizer(Muon, args.d, device)
    loaded.load_state_dict(state)

    stale_err = max_error_after_apply(loaded, loaded_params, expected)
    rebuild_preconditioner_views_after_load(loaded)
    fixed_err = max_error_after_apply(loaded, loaded_params, expected)

    print(f"device={device}")
    print(f"direct_apply_max_err={direct_err:.6g}")
    print(f"post_load_without_rebuild_max_err={stale_err:.6g}")
    print(f"post_load_with_rebuild_max_err={fixed_err:.6g}")

    if device.type != "cuda":
        if direct_err > 1e-5:
            print(
                "warning=cpu bmm(out=input) is not training-parity for these batched shapes; "
                "rerun with --device cuda on GPU2 for the optimizer runtime check"
            )
        return

    if direct_err > 1e-5:
        raise SystemExit("direct preconditioner apply mismatch")
    if stale_err <= 1e-3:
        raise SystemExit("checkpoint roundtrip did not expose stale apply-plan mismatch")
    if fixed_err > 1e-5:
        raise SystemExit("rebuilt preconditioner apply mismatch")


if __name__ == "__main__":
    main()

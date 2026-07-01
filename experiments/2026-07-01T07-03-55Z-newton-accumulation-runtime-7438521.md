# 2026-07-01T07-03-55Z-newton-accumulation-runtime-7438521

- method: newton_muon1
- mode: runtime_diagnostic
- status: completed
- git_sha: 7438521412eb495a6d9b3b1a0149843c84435b94
- machine: AIStation GPU2
- device: NVIDIA A800-SXM4-80GB
- started_at: 2026-07-01T07:03:55Z
- finished_at: 2026-07-01T07:05Z

## Hypothesis

The remaining Newton-Muon reproduction gap may be caused by a fallback or
mis-wired activation covariance accumulation path, rather than by the Cholesky
inverse or gradient apply path already checked in the CUDA preconditioner
diagnostic.

## Command

```bash
cd /huyang2/muon_plus
git checkout --detach 7438521412eb495a6d9b3b1a0149843c84435b94
.venv/bin/python scripts/check_newton_accumulation_runtime.py \
  --compiled --batch 2 --seq-len 64 --n-embd 128 --n-head 2
```

## Result

```text
compiled=True
qkv_count=1 qkv_max_abs_err=0.000859499
o_count=1 o_max_abs_err=4.92185e-05
c_fc_count=1 c_fc_max_abs_err=0.00178623
c_proj_count=1 c_proj_max_abs_err=0.000390291
```

## Decision

The compiled hook path is active and numerically close to the torch reference
for all instrumented module families. This rules out a gross activation-side
fallback as the reason the repaired full run reached only `3.2785` versus the
paper's `3.2611`.

Next target: log live `C/P/gP` telemetry at refresh boundaries and choose the
algorithmic branch from those signals.

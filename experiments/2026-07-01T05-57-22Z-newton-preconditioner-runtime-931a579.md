# 2026-07-01T05-57-22Z-newton-preconditioner-runtime-931a579

- method: newton_muon1
- mode: runtime_diagnostic
- status: completed
- git_sha: 931a5796360687a6aedad372fec84072f8f99926
- machine: AIStation GPU2
- device: NVIDIA A800-SXM4-80GB
- started_at: 2026-07-01T05:57:22Z
- finished_at: 2026-07-01T05:57:22Z

## Hypothesis

The repaired Newton-Muon full run may still miss the paper result if the
batched right-preconditioner apply path is numerically wrong on CUDA, or if
checkpoint restore still leaves `_apply_plan` disconnected from the loaded
`precond_inv_apply` tensors.

## Command

```bash
cd /huyang2/muon_plus
git checkout --detach 931a5796360687a6aedad372fec84072f8f99926
.venv/bin/python scripts/check_newton_preconditioner_runtime.py --device cuda
```

## Result

```text
device=cuda
direct_apply_max_err=0
post_load_without_rebuild_max_err=41.14
post_load_with_rebuild_max_err=0
```

## Decision

The stale `_apply_plan` issue is confirmed as a real checkpoint bug, and the
current rebuild fix exactly restores the active CUDA preconditioner apply path.
This rules out the latest repaired run failing because preconditioned gradients
are still disconnected after resume.

Next target: activation covariance accumulation and refresh scheduling. That
has higher information value than another full run because the full
offset-preserving run already completed `6200/6200` and stayed at `3.2785`.

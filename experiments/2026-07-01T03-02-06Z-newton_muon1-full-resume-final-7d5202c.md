# 2026-07-01T03-02-06Z-newton_muon1-full-resume-final-7d5202c

- method: newton_muon1
- mode: full
- status: completed
- git_sha: 7d5202ca79359c2bb3796aad99149fb064dc6fe9
- source: train_gpt_newton_muon_1.py
- launcher: python
- started_at: 2026-07-01T02:59:11Z
- finished_at: 2026-07-01T03:51:22Z

## Hypothesis

This run tests the last high-value resume hypothesis: after preserving the
upstream DataLoader prefetch offset and rebuilding Newton preconditioner apply
views after optimizer state restore, Newton-Muon-1 should recover the
paper-reported single-GPU validation loss if the prior gap was caused by our
resume machinery.

## Result

- final_val_loss: 3.2785
- paper_loss: 3.2611
- loss_delta: 0.017400000000000304
- train_time_s: 15830.292
- paper_time_s: 7443.3
- peak_memory_mib: 39789

## Decision

This full Newton-Muon run completed to `6200/6200`, but it did not reproduce
the paper optimizer delta. The result matches the previous repaired run
(`3.2785`) and remains only `0.0028` better than the reproduced Muon baseline
(`3.2813`), while the paper claims a Newton-Muon-1 loss of `3.2611`.

Conclusion: the remaining gap is not explained by training length, GPU-only
wall-time differences, smoke-mode compile fallback, TensorDescriptor fallback,
or the audited resume offset. Another blind full rerun has low information
value. The next experiment should be a targeted Newton-specific diagnostic of
the active right-preconditioned gradient path, refresh boundaries, and
optimizer source parity.

## Artifacts

- raw_run_dir: runs/2026-07-01T03-02-06Z-newton_muon1-full-resume-final-7d5202c
- stdout_log: runs/2026-07-01T03-02-06Z-newton_muon1-full-resume-final-7d5202c/stdout.log
- source_snapshot: runs/2026-07-01T03-02-06Z-newton_muon1-full-resume-final-7d5202c/source_snapshot

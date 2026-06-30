# 2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9

- method: newton_muon1
- mode: full
- status: completed
- git_sha: 46d9db9a9ec51979c6c771655fde44ef63834943
- source: train_gpt_newton_muon_1.py
- launcher: python
- started_at: 2026-06-30T03:51:05Z
- finished_at: 2026-06-30T05:45:23Z

## Hypothesis

If the earlier Newton-Muon miss was caused by resume corruption, then resuming
from the last pre-corruption checkpoint with fixed preconditioner apply-plan
views and exact DataLoader seek should recover the paper-scale Newton-Muon
improvement over Muon.

## Result

- final_val_loss: 3.2785
- paper_loss: 3.2611
- loss_delta: 0.017400000000000304
- train_time_s: 15875.037
- paper_time_s: 7443.3
- peak_memory_mib: 39789
- reproduced_muon1_loss: 3.2813
- improvement_vs_reproduced_muon1: 0.0028
- improvement_vs_broken_newton_run: 0.0021

## Decision

The resume repair helped, but Newton-Muon-1 is still not reproduced. The result
beats the reproduced Muon-1 baseline by only `0.0028`, while the paper reports
a `0.0182` Newton-Muon gain over Muon. Do not tag this as an excellent result.

## Lesson

The preconditioner-view bug was real because the corrected resume improved the
final loss from `3.2806` to `3.2785`, but it explains only a small fraction of
the missing optimizer delta. The next high-ROI step is a targeted audit of the
materialized Newton-Muon source, optimizer-state serialization, and exact
upstream/runtime differences; a broad AdamW/Muon/Newton table is low value
until the Newton-specific gap has a sharper mechanism.

## Artifacts

- raw_run_dir: runs/2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9
- stdout_log: runs/2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9/stdout.log
- source_snapshot: runs/2026-06-30T03-51-05Z-newton_muon1-full-resume-fix3-46d9db9/source_snapshot

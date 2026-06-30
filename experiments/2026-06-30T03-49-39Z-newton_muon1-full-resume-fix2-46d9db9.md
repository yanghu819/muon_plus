# 2026-06-30T03-49-39Z-newton_muon1-full-resume-fix2-46d9db9

- method: newton_muon1
- mode: full
- status: failed
- git_sha: 46d9db9a9ec51979c6c771655fde44ef63834943
- source: train_gpt_newton_muon_1.py
- launcher: python
- started_at: 2026-06-30T03:49:40Z
- finished_at: 2026-06-30T03:50:03Z

## Hypothesis

Resume the corrected Newton-Muon-1 run from the latest good checkpoint after
the preconditioner-view and DataLoader-seek fixes, preserving the already-paid
early trajectory instead of restarting blindly.

## Result

- final_val_loss: None
- paper_loss: 3.2611
- loss_delta: None
- train_time_s: None
- paper_time_s: 7443.3
- peak_memory_mib: None

## Decision

Do not compare this run to the paper. The launch failed before training because
`RESUME_CHECKPOINT` was passed as a relative path while `run.sh` executes the
materialized training script from `runs/<run_id>/source_snapshot`.

## Lesson

Resume checkpoints must be absolute `/huyang2/muon_plus/...` paths when a run
is launched through `run.sh`. This is provenance, not a model result, and it is
kept so future resumes do not waste GPU2 lease time on the same path bug.

## Artifacts

- raw_run_dir: runs/2026-06-30T03-49-39Z-newton_muon1-full-resume-fix2-46d9db9
- stdout_log: runs/2026-06-30T03-49-39Z-newton_muon1-full-resume-fix2-46d9db9/stdout.log
- source_snapshot: runs/2026-06-30T03-49-39Z-newton_muon1-full-resume-fix2-46d9db9/source_snapshot

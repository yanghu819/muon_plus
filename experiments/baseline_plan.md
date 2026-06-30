# Baseline Reproduction Plan

## Target

Reproduce the upstream Newton-Muon single-GPU Muon baseline first, then compare the final validation loss with the paper/README value.

## Upstream Snapshot

- repo: https://github.com/zhehangdu/Newton-Muon
- commit: df78af0db523d8bceb25af4919a3e3e7082b80f3
- imported_at: 2026-06-29T06:00:00Z

## Paper Reference Values

| Method | Loss | Time (s) | Notes |
| --- | ---: | ---: | --- |
| adam1 | 3.3801 | 7228.4 | Newton-Muon-1, single H100 |
| muon1 | 3.2793 | 7314.1 | Newton-Muon-1 Muon baseline, single H100 |
| newton_muon1 | 3.2611 | 7443.3 | Newton-Muon-1, single H100 |
| adam2 | 3.4628 | 4272.0 | Newton-Muon-2 average |
| muon2 | 3.2787 | 4305.9 | Newton-Muon-2 Muon baseline average |
| newton_muon2 | 3.2739 | 4342.4 | Newton-Muon-2 average |

## Experiment Choice

Insight: the fastest decision-value anchor is not a broad optimizer grid. It is one exact Muon-1 baseline run because the user wants a reproduction base for future Newton-Muon changes, and Muon-1 is the README's named baseline with the same single-GPU memory requirement as GPU2's A100 80GB.

Prediction: on A100 80GB, the final validation loss should be close to 3.2793 if the data, code, and dependency stack are correct. Wall time is expected to differ from the H100 paper number and should be treated as secondary.

Kill criteria: stop before a full run if smoke fails, FineWeb shard verification fails, CUDA is unavailable, or the current AIStation GPU2 instance cannot stay alive long enough to finish a baseline segment safely.

Next decision unlocked: once muon1 is reproduced, run newton_muon1 under the same harness and compare loss delta against the reported 0.0182 improvement over Muon.

## 2026-06-29 Smoke Lesson

The AIStation GPU2 base image provides PyTorch 2.7.0+cu126 with Triton 3.3.0.
That Triton build does not expose `triton.tools.tensor_descriptor`, while the
upstream `triton_kernels.py` imported it at module import time. The Muon-1 and
Newton-Muon-1 scripts only need `XXT` and `ba_plus_cAA`, so the safe fix is to
delay the TensorDescriptor import until `linear_relu_square()` is actually used.
This keeps the Newton-Muon-1 baseline path unchanged and avoids downloading a
large or mismatched torch/triton stack into the project environment.

The second smoke attempt showed that tiny-shape `torch.compile` spends minutes
in TorchInductor/PTX compilation before producing one training step. That is not
a useful smoke signal and does not reuse enough work for the full 12-layer
baseline. Smoke materialization now skips `torch.compile`; full mode still uses
the upstream compile path unchanged.

The first exact Muon-1 full attempt on GPU2 reached step 41 with a stable
post-warmup step average around 2523 ms. At that rate, 6200 training steps alone
need about 4.3 hours before validation overhead, which exceeds a single
AIStation GPU2 lease. Full-mode materialization now saves checkpoints every 100
steps by default and supports `RESUME_CHECKPOINT`/`RESUME_STEP`; this preserves
the baseline training path while making completion across GPU2 restarts
practical.

The first resume attempt from `state_step002100.pt` exposed a hidden recovery
cost: iterating `resume_step * train_accumulation_steps` batches burned more
than 15 minutes of CPU time with the GPU idle. The loader is deterministic and
has explicit `current_shard/current_position` state, so resume now seeks
directly to the target batch and checkpoints carry `training_time_ms`. For old
checkpoints without that field, pass `RESUME_TRAIN_TIME_MS` from the previous
segment log.

## Next Experiment: Newton-Muon-1

Insight: after Muon-1 reproduced within `+0.0020` loss of the paper, the
highest-value next check is the paper's actual optimizer delta, not an AdamW
grid or formatting ablation.

Mechanism: run `newton_muon1` with the same FineWeb shards, harness, checkpoint
resume, and single GPU2 environment. This isolates the right-preconditioner
change against the now-validated Muon baseline path.

Prediction: final validation loss should land near the paper value `3.2611`,
roughly `0.0182` below the Muon paper baseline and about `0.0202` below this
GPU2 Muon reproduction.

Expected upside: if it matches, the repo is ready as a reliable baseline for
optimizer changes. If it misses, the next investigation should focus on
Newton-Muon-specific preconditioner state, kernel behavior, or checkpoint
resume state rather than data/dependency issues.

Budget: one full `_1` run, expected to need multiple AIStation lease segments
at about Muon-1 speed. Use checkpoint resume; do not run AdamW unless the
Newton-Muon result creates a specific diagnostic need.

Kill criteria: stop before full if smoke fails, CUDA/kernel registration fails,
FineWeb verification fails, or resume cannot preserve optimizer/preconditioner
state.

## 2026-06-30 Newton-Muon-1 Result

The Newton-Muon-1 full run completed on GPU2 at git SHA
`9c87b12eee49dc98c7b7f3141299f43093b4b5c0` with final validation loss
`3.2806`. This is only `0.0007` better than the reproduced Muon-1 loss
`3.2813` and remains `+0.0195` above the paper Newton-Muon-1 value `3.2611`.
Conclusion: the run is complete, but the paper optimizer improvement was not
reproduced.

The highest-value diagnostic is checkpoint resume correctness, not another
optimizer grid. The Newton-Muon optimizer keeps batched preconditioner apply
buffers outside the serialized optimizer state graph. After
`optimizer.load_state_dict()`, the loaded `precond_inv_apply` tensors no longer
point at the batched `_apply_plan` buffers created before loading. That means a
resumed run can keep updating checkpoint tensors while applying stale identity
preconditioners in the batched gradient path. Full-mode materialization now
rewires those views immediately after loading optimizer state.

The same resume audit found a one-batch DataLoader seek offset. A checkpoint
at `step=N` is saved before training step `N`, with the next batch already
positioned at `N * train_accumulation_steps`. The old seek used
`N * train_accumulation_steps - 1`, repeating the previous batch on resume.
Full-mode materialization now seeks to the exact next batch index.

Next decision: rerun Newton-Muon-1 from the last checkpoint produced before the
first resume boundary, using the fixed materialization. A blind from-scratch
rerun is lower ROI while the pre-resume checkpoint can preserve the valid early
trajectory and directly test whether the resume fix recovers the reported
Newton-Muon gap.

## 2026-06-30 Corrected Resume Result

The corrected Newton-Muon-1 resume completed on GPU2 at git SHA
`46d9db9a9ec51979c6c771655fde44ef63834943` with final validation loss
`3.2785`. That is better than the broken resumed Newton-Muon run (`3.2806`) and
slightly better than the reproduced Muon-1 baseline (`3.2813`), but it remains
`+0.0174` above the paper Newton-Muon-1 loss `3.2611`.

Conclusion: Muon-1 is reproduced; Newton-Muon-1 is not reproduced yet. The
resume repair recovered only `0.0021` loss versus the broken Newton-Muon run,
while the paper-level Newton-Muon delta should be about `0.0182` versus Muon.
This rules out using the corrected run as a trusted baseline for new optimizer
ideas.

Operational lesson: `run.sh` launches from each run's `source_snapshot`, so
`RESUME_CHECKPOINT` must be an absolute `/huyang2/muon_plus/...` path. The
failed `2026-06-30T03-49-39Z-newton_muon1-full-resume-fix2-46d9db9` run is
kept in the ledger as a provenance and lease-time lesson.

Next decision: do not run a broad comparison table or blind from-scratch
rerun. The next experiment must isolate a Newton-specific mechanism: compare
the materialized training source against the imported upstream file, audit
preconditioner refresh/apply tensors across a checkpoint round trip on a tiny
deterministic batch, and verify whether the expected right-preconditioned
gradient path is active before spending another full GPU2 lease.

## 2026-06-30 Resume Offset Re-Audit

The corrected-resume run still used one materialization change that does not
match an uninterrupted upstream training stream. The upstream script prefetches
`x, y = train_loader.next_batch()` and then calls `train_loader.reset()` before
the training loop. Because of that quirk, checkpoint step `N` is positioned with
the next prefetched training batch at `N * train_accumulation_steps - 1`, not
`N * train_accumulation_steps`.

The prior change to seek exactly `N * train_accumulation_steps` was therefore
not an upstream-preserving fix. Full materialization now restores the
`-1` offset and keeps the preconditioner view rebuild. It also fast-forwards
the LR scheduler if a legacy checkpoint lacks serialized scheduler state,
because missing scheduler state would delay the warmdown and answer the wrong
"trained enough" question.

Next decision: rerun Newton-Muon-1 from the same clean step-2100 checkpoint
with both fixes active. This run has high information value: if it recovers the
paper loss, the miss was implementation/resume-induced; if it remains near
`3.278`, the next target is the Newton preconditioner math/runtime itself, not
training length or GPU model.

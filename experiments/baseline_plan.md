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

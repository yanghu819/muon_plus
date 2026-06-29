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


# 2026-07-02T05-20-30Z-newton_full_trust-1200-control-3fb689d

- method: newton_full_trust
- mode: same-horizon 1200-step control
- git_sha: 3fb689df25faec35cce414fdb494150a362d4caa
- machine: AIStation GPU2, NVIDIA A800-SXM4-80GB
- seed: 1337
- status: in_progress
- worktree: `/huyang2/muon_plus/_worktrees/hybrid_3fb689d`

## Hypothesis

The lite256 schedule's 900-step advantage over fixed-seed full Trust is only
about `0.001` and the 1200-step final loss has no same-horizon full Trust
control. A single 1200-step full Trust run answers the promotion question: is
Lite256->full Trust a real temporal preconditioning improvement, or did it only
look better because the comparison stopped at 900?

## Configuration

- optimizer path: `newton_muon1`
- mode: `full`
- iterations: `1200`
- validation cadence: every `300` steps
- validation tokens: `1048576`
- warmdown: disabled with `WARMDOWN_ITERS=0`
- flags:
  - `NEWMUON_TRUST=1`
  - `NEWMUON_TELEMETRY_MAX_STEP=1200`

## Command

```bash
RUN_ID=2026-07-02T05-20-30Z-newton_full_trust-1200-control-3fb689d \
SEED=1337 \
NUM_ITERATIONS=1200 \
WARMDOWN_ITERS=0 \
VAL_LOSS_EVERY=300 \
VAL_TOKENS=1048576 \
SAVE_EVERY=0 \
NEWMUON_TRUST=1 \
NEWMUON_TELEMETRY_PATH=/huyang2/muon_plus/runs/${RUN_ID}/newton_telemetry.jsonl \
NEWMUON_TELEMETRY_MAX_STEP=1200 \
./run.sh full newton_muon1
```

## Results

- pending

## Decision

Prune interpretation is strict. If full Trust is `<= 3.6895` at step 1200, the
temporal schedule is not worth a full 6200-step promotion. If full Trust is
worse than `3.6895` while preserving the known 900-step gap, promote one
lite256 full-length run and use telemetry for the paper mechanism story.

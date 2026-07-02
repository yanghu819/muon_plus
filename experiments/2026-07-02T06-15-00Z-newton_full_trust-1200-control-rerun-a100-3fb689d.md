# 2026-07-02T06-15-00Z-newton_full_trust-1200-control-rerun-a100-3fb689d

- method: newton_full_trust
- mode: same-horizon 1200-step control rerun after AIStation halt
- git_sha: 3fb689df25faec35cce414fdb494150a362d4caa
- machine: AIStation GPU2, NVIDIA A100-SXM4-80GB
- seed: 1337
- status: in_progress
- worktree: `/huyang2/muon_plus/_worktrees/hybrid_3fb689d`
- actual_started_at: `2026-07-02T06:14:46Z`
- pid: `165`
- launcher_log: `/huyang2/muon_plus/artifacts/2026-07-02T06-15-00Z-newton_full_trust-1200-control-rerun-a100-3fb689d.launcher.log`
- hardware_note: restarted GPU2 instance reports `NVIDIA A100-SXM4-80GB`; torch emitted a TF32 availability warning, but no matmul precision setting was changed.

## Hypothesis

The first same-horizon control was interrupted after step 623 by the AIStation
environment halt, but its step-300/600 points already trailed Lite256. A fresh
uninterrupted 1200-step full Trust control is still the blocking evidence for
whether Lite256->full Trust deserves a 6200-step promotion.

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
RUN_ID=2026-07-02T06-15-00Z-newton_full_trust-1200-control-rerun-a100-3fb689d \
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

- step 300 val_loss: `4.3811`

## Decision

Step 300 reproduces full Trust's known early weakness despite the restarted
A100 instance: it matches the prior full Trust 900-run step-300 value (`4.3810`)
and the interrupted control (`4.3795`), while trailing Lite256 (`4.3690` in the
1200 extension, `4.3667` in the 300 gate). Continue to 600/900/1200. If final
full Trust is `<= 3.6895`, stop the temporal schedule. If full Trust is worse
than `3.6895`, promote one full-length Lite256->full Trust run, while noting
that the rerun used the restarted GPU2 A100 instance.

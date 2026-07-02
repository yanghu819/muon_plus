# 2026-07-02T06-15-00Z-newton_full_trust-1200-control-rerun-a100-3fb689d

- method: newton_full_trust
- mode: same-horizon 1200-step control rerun after AIStation halt
- git_sha: 3fb689df25faec35cce414fdb494150a362d4caa
- machine: AIStation GPU2, NVIDIA A100-SXM4-80GB
- seed: 1337
- status: completed
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
- step 600 val_loss: `3.9177`
- step 900 val_loss: `3.7756`
- step 1200 val_loss: `3.6919`
- train_time_s: `2975.242`
- peak_memory_mib: `39448`
- finished_at: `2026-07-02T07:08:47Z`

## Decision

Completed. Step 900 reversed the early Lite256 advantage. Step 300 reproduced full
Trust's known early weakness despite the restarted A100 instance: it matched the
prior full Trust 900-run step-300 value (`4.3810`) and the interrupted control
(`4.3795`), while trailing Lite256 (`4.3690` in the 1200 extension, `4.3667` in
the 300 gate). Step 600 was also behind Lite256 (`3.9177` vs `3.9162`), but only
by `0.0015`. By step 900, full Trust is better than the lite256 1200 extension
(`3.7756` vs `3.7767`) and slightly better than the prior lite256 900 run
(`3.7762`). At step 1200, full Trust ends at `3.6919`, worse than the A800
Lite256 extension final `3.6895` by `0.0024`. That is a real promotion signal if
hardware is ignored, but the margin is small and this control ran on the
restarted A100 instance. Do not promote directly to 6200 yet. Launch one
A100-side Lite256 1200 mirror; if it beats this full Trust control on the same
hardware, promote a single 6200-step Lite256 run.

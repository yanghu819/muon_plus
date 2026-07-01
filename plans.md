# Experiment Plans

This ledger tracks mechanism-driven experiments only. Broad table-filling
ablations are intentionally excluded.

| id | status | hypothesis | evidence | next decision |
| --- | --- | --- | --- | --- |
| newton-accumulation-runtime-7438521 | done | Check whether compiled activation covariance accumulation is wired correctly. | qkv/o/c_fc/c_proj max absolute errors were all below `0.002` against torch reference. | Move from fallback suspicion to live optimizer telemetry. |
| newton-telemetry-gate-7438521 | done | Determine whether Newton-Muon is weak, too strong, noisy, or module-dependent. | `o/c_proj` had low cosine and extreme norm ratios, while `qkv` was moderate. | Prefer Trust-Region Newton-Muon over low-rank or structure sweeps. |
| trust-newton-muon1-600-7438521 | done | Downweight low-trust Newton directions while preserving reliable module families. | 600-step run completed with val `3.9154`; alpha split matched the telemetry prediction. | Promote to a 2100-step gate before any full 6200-step run. |
| trust-newton-muon1-2100-7438521 | in_progress | Verify whether the Trust mechanism has early-run advantage strong enough to justify a full run. | Launched on GPU2 at 2026-07-01T07:45:43Z. | If 2100 trend is not competitive, pivot to Scale-Invariant Newton-Muon; otherwise promote to 6200. |
| scale-invariant-newton-muon | pending | Separate coordinate correction from hidden global learning-rate scaling. | Pending; only worth running if Trust 2100 fails to show enough gain or telemetry shows scale remains the dominant issue. | Decide after Trust 2100. |
| lagged-newton-muon | pending | Remove current-batch coupling at refresh boundaries. | Pending; only worth running if refresh-boundary instability remains after Trust. | Decide after Trust 2100 telemetry. |

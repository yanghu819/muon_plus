# Experiment Plans

This ledger tracks mechanism-driven experiments only. Broad table-filling
ablations are intentionally excluded.

| id | status | hypothesis | evidence | next decision |
| --- | --- | --- | --- | --- |
| newton-accumulation-runtime-7438521 | done | Check whether compiled activation covariance accumulation is wired correctly. | qkv/o/c_fc/c_proj max absolute errors were all below `0.002` against torch reference. | Move from fallback suspicion to live optimizer telemetry. |
| newton-telemetry-gate-7438521 | done | Determine whether Newton-Muon is weak, too strong, noisy, or module-dependent. | `o/c_proj` had low cosine and extreme norm ratios, while `qkv` was moderate. | Prefer Trust-Region Newton-Muon over low-rank or structure sweeps. |
| trust-newton-muon1-600-7438521 | done | Downweight low-trust Newton directions while preserving reliable module families. | 600-step run completed with val `3.9154`; alpha split matched the telemetry prediction. | Promote to a 2100-step gate before any full 6200-step run. |
| trust-newton-muon1-2100-7438521 | discarded | Verify whether the Trust mechanism has early-run advantage strong enough to justify a full run. | Stopped by strategy change after step-900 val `3.7713`; user requested fast idea screening instead of one variant running long. | Keep partial trend only as evidence that Trust is safe, not as a promotion decision. |
| fast-idea-sweep-128-04a83d36 | done | Before spending 300+ steps, ask which activation-side preconditioner family has any early signal. | 128-step vals: Muon `5.2253`, Newton base `5.4443`, Trust `5.2058`, Scale `5.2676`, Lagged `5.4418`, Lite diag `5.1874`. | Prune full covariance, lagged, and scale-only; keep Trust and Lite diag as positive directions. |
| newton-lite-scale-128-04a83d36 | discarded | Determine whether Lite diagonal's early win is merely hidden global learning-rate scaling. | Lite+scale final val `5.2051`, worse than pure Lite diag `5.1874`; scale normalization removed useful early acceleration. | Do not promote scale normalization alone; test norm control through adaptive or trust gates instead. |
| fast-variants-300-05138f72 | done | Quickly cover the remaining top-level ideas with 300-step gates, then prune. | Best was Lite+Layer-Adaptive `4.3658`; Lite+Trust `4.3677`; Lite diag `4.3811`; qkv/c_fc mask `4.4118`; low-rank16 `4.4062`; sketch64 `4.4392`. | Promote Lite+Layer-Adaptive to the next longer gate; prune hard module masking, low-rank, and random sketch for now. |

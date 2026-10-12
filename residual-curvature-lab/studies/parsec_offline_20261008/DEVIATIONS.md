# Deviations from PROTOCOL.md

| # | Time (UTC) | Section | Departure | Reason |
| --- | --- | --- | --- | --- |
| 1 | 2026-10-08T16:27Z | Execution | The first B1 unit (`main__searched_soap-0010`) ran directly, not through the queue, to check the capacity-screen gate before launching the rest. | Smoke test of the gate; identical code path and outputs. |
| 2 | 2026-10-08T16:55Z | Analysis (exploratory additions) | Three analyses beyond the protocol, labelled exploratory in RESULTS.md: (a) PARSEC-L's share of exact tensor-GGN headroom on the 69 states shared with gap-audit 2c/4b; (b) the C reading with the degenerate MLP states excluded; (c) GELU activity of the SN-study MLP states. | (a) puts B1's ratio on the same scale as earlier families; (b, c) the MLP's no-SN and c = 6 runs had dead residual branches by step 200 (zero curvature), which the protocol did not anticipate. The pre-declared readings are reported unchanged. |

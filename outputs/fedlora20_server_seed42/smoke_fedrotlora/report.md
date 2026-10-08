# FedRot-LoRA continuation

Local mean / global Macro use identical full original tests. SMOKE ONLY: train8/client, not formal results.

| Round | Local own-domain mean (%) | Global Macro (%) | Local − Global (pp) |
|---:|---:|---:|---:|
| 1 | 68.68 | 53.47 | +15.21 |
| 2 | 70.38 | 60.95 | +9.44 |
| 3 | 71.30 | 64.21 | +7.09 |
| 4 | 73.25 | 65.85 | +7.40 |
| 5 | 73.04 | 66.60 | +6.44 |
| 6 | 73.93 | 67.27 | +6.65 |
| 7 | 73.15 | 68.45 | +4.70 |
| 8 | 73.59 | 68.63 | +4.96 |
| 9 | 73.10 | 69.00 | +4.10 |
| 10 | 73.51 | 68.95 | +4.56 |
| 11 | 69.30 | 68.98 | +0.32 |
| 12 | 69.43 | 69.08 | +0.35 |

Best shared round: 12; Macro 69.08%.

Original algorithm implementation warnings remain unchanged; see source protocol warnings. Imported old rounds have provenance in the transfer manifest.

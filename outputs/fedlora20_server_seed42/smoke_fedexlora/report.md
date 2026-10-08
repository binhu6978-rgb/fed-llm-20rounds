# FedEx-LoRA continuation

Local mean / global Macro use identical full original tests. SMOKE ONLY: train8/client, not formal results.

| Round | Local own-domain mean (%) | Global Macro (%) | Local − Global (pp) |
|---:|---:|---:|---:|
| 1 | 68.68 | 54.21 | +14.47 |
| 2 | 70.83 | 61.25 | +9.58 |
| 3 | 70.93 | 64.39 | +6.54 |
| 4 | 73.29 | 65.69 | +7.61 |
| 5 | 73.20 | 66.72 | +6.48 |
| 6 | 73.99 | 67.44 | +6.54 |
| 7 | 73.70 | 68.27 | +5.43 |
| 8 | 73.20 | 68.93 | +4.27 |
| 9 | 72.85 | 68.73 | +4.12 |
| 10 | 72.93 | 69.07 | +3.86 |
| 11 | 69.40 | 69.12 | +0.28 |
| 12 | 69.34 | 69.17 | +0.17 |

Best shared round: 12; Macro 69.17%.

Original algorithm implementation warnings remain unchanged; see source protocol warnings. Imported old rounds have provenance in the transfer manifest.

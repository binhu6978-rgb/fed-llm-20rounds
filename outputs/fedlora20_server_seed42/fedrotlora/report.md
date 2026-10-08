# FedRot-LoRA continuation

Local mean / global Macro use identical full original tests. Round1–10 reused; new full rounds11–20. Total training budget doubles.

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
| 11 | 72.85 | 69.48 | +3.37 |
| 12 | 73.21 | 69.39 | +3.82 |
| 13 | 73.17 | 70.14 | +3.03 |
| 14 | 73.23 | 69.83 | +3.40 |
| 15 | 73.21 | 70.08 | +3.13 |
| 16 | 73.27 | 70.15 | +3.12 |
| 17 | 73.10 | 70.24 | +2.86 |
| 18 | 73.26 | 70.10 | +3.16 |
| 19 | 72.99 | 70.34 | +2.65 |
| 20 | 72.32 | 70.22 | +2.11 |

Best shared round: 19; Macro 70.34%.

Original algorithm implementation warnings remain unchanged; see source protocol warnings. Imported old rounds have provenance in the transfer manifest.

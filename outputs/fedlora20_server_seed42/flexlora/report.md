# FlexLoRA continuation

Local mean / global Macro use identical full original tests. Round1–10 reused; new full rounds11–20. Total training budget doubles.

| Round | Local own-domain mean (%) | Global Macro (%) | Local − Global (pp) |
|---:|---:|---:|---:|
| 1 | 68.68 | 60.78 | +7.90 |
| 2 | 71.65 | 62.44 | +9.21 |
| 3 | 71.72 | 64.50 | +7.23 |
| 4 | 71.95 | 65.65 | +6.30 |
| 5 | 70.70 | 66.38 | +4.32 |
| 6 | 70.36 | 66.81 | +3.55 |
| 7 | 69.77 | 67.39 | +2.37 |
| 8 | 70.66 | 67.81 | +2.86 |
| 9 | 70.11 | 67.90 | +2.21 |
| 10 | 69.90 | 68.36 | +1.54 |
| 11 | 69.74 | 68.19 | +1.55 |
| 12 | 69.47 | 68.01 | +1.46 |
| 13 | 69.48 | 68.61 | +0.87 |
| 14 | 69.95 | 68.49 | +1.46 |
| 15 | 69.72 | 68.25 | +1.47 |
| 16 | 70.04 | 68.71 | +1.33 |
| 17 | 69.61 | 68.78 | +0.83 |
| 18 | 69.94 | 68.78 | +1.15 |
| 19 | 69.85 | 68.71 | +1.14 |
| 20 | 69.66 | 68.75 | +0.91 |

Best shared round: 18; Macro 68.78%.

Original algorithm implementation warnings remain unchanged; see source protocol warnings. Imported old rounds have provenance in the transfer manifest.

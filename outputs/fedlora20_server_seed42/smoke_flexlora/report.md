# FlexLoRA continuation

Local mean / global Macro use identical full original tests. SMOKE ONLY: train8/client, not formal results.

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
| 11 | 68.34 | 68.20 | +0.14 |
| 12 | 68.21 | 68.35 | -0.14 |

Best shared round: 10; Macro 68.36%.

Original algorithm implementation warnings remain unchanged; see source protocol warnings. Imported old rounds have provenance in the transfer manifest.

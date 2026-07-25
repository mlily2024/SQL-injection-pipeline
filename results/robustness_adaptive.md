# Robustness test 4: adaptive (best-of-N) evasion attack

For a sample of 400 malicious test queries, an adaptive attacker tries a library of 7 WAF-bypass transforms and keeps whichever evades the detector. A query counts as evaded if any transform makes the model predict benign. The best-of-N evasion rate is a stronger robustness measure than any single transform; lower is more robust.

| Transform | Structure-GNN evasion (%) | BERT-only evasion (%) |
|---|---|---|
| url_encode | 6.25 | 8.00 |
| inline | 0.00 | 0.00 |
| case_mix | 0.00 | 0.00 |
| kw_split | 0.00 | 0.00 |
| mysql | 0.00 | 0.00 |
| double_url | 0.00 | 0.00 |
| combined | 0.00 | 0.00 |

**Adaptive best-of-7 evasion rate: structure-GNN 6.25%, BERT-only 8.00%.** The structure-aware model is evaded less often by an adaptive attacker.

Reproduce with `python robustness_4_adaptive.py`.

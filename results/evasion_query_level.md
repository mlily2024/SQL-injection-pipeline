# Query-level evasion robustness (DistilBERT-Stacked Ensemble vs SVM)

Recall (% of 2291 malicious test queries still detected) after applying WAF-bypass obfuscation transforms to the raw query, then re-embedding with DistilBERT. Unlike the meta-feature FGSM, this measures resistance to realistic query-level evasion.

| Transform | Ensemble recall (%) | SVM recall (%) |
|---|---|---|
| clean | 99.74 | 99.83 |
| url_encode | 99.17 | 99.61 |
| inline_comment | 96.38 | 97.21 |
| case_random | 99.74 | 99.83 |
| tab_whitespace | 99.74 | 99.83 |
| combined_url_case | 99.17 | 99.61 |
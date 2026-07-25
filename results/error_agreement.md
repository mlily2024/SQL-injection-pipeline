# Error agreement + ensemble (BERT-only [CLS] vs structure-GNN mean-token)

Test set n=9276. Do the two models err on the *same* queries? If errors are complementary, an ensemble can beat either alone.

## Error overlap
| | count |
|---|---|
| Both correct | 9231 |
| Both wrong | 16 |
| GNN correct, BERT-only wrong | 9 |
| BERT-only correct, GNN wrong | 20 |

Of the 45 queries at least one model gets wrong, only 16 are missed by BOTH; 29 are 'saved' by one model. Errors are largely complementary.

## Accuracy / detection
| Model | Accuracy (%) | Mal. recall (%) | FPR (%) |
|---|---|---|---|
| BERT-only ([CLS]+MLP) | 99.73 | 99.48 | 0.120 |
| Structure-GNN | 99.61 | 99.39 | 0.257 |
| Ensemble (prob average) | 99.70 | 99.45 | 0.154 |
| Ensemble (OR / flag-if-either) | 99.66 | 99.62 | 0.326 |
| Ensemble (AND / flag-if-both) | 99.69 | 99.25 | 0.051 |

Best single = 99.73%; best ensemble (avg) = 99.70%. Ensemble does NOT beat the best single model. NOTE: the GNN is seed-noisy; treat a single-run ensemble gain as provisional pending multi-seed.

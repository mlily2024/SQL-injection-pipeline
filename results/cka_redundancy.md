# Representation redundancy (linear CKA)

How much is the structure-GNN's representation a re-encoding of BERT's [CLS]? CKA in [0,1]; 1 = identical geometry. A high value + a high linear-probe R^2 means the GNN adds no representation BERT did not already provide.

| Comparison | Value |
|---|---|
| CKA(GNN penultimate, BERT[CLS]) | 0.001 |
| CKA(GNN pooled,      BERT[CLS]) | 0.001 |
| CKA(GNN pooled, mean BERT token) | 0.532 |
| CKA(BERT[CLS], mean BERT token) | 0.002 |
| CKA(GNN pooled, RANDOM) [floor] | 0.004 |
| Linear probe R^2: [CLS] -> GNN rep | 0.170 |

Interpretation: CKA(GNN, [CLS]) = 0.001 (random floor 0.004); [CLS] linearly reconstructs the GNN representation with R^2 = 0.170.

# Low-data learning curves: structure-GNN vs BERT-only (MLP)

Trained on stratified fractions of the training set (3 seeds each), tested on the full 9,276-query test set. If the GNN exceeds BERT-only at small fractions, structure buys sample efficiency.

| Train fraction | GNN acc (%) | BERT-only acc (%) | GNN recall (%) | BERT-only recall (%) |
|---|---|---|---|---|
| 1% | 98.22 | 98.08 | 96.87 | 96.12 |
| 2% | 98.64 | 98.84 | 97.73 | 97.64 |
| 5% | 99.01 | 99.25 | 98.38 | 98.36 |
| 10% | 99.26 | 99.43 | 98.87 | 98.79 |
| 25% | 99.43 | 99.53 | 99.09 | 98.98 |
| 50% | 99.59 | 99.64 | 99.42 | 99.31 |
| 100% | 99.65 | 99.68 | 99.52 | 99.29 |
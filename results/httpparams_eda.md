# HttpParams (Morzeux) SQLi dataset - cleaning, feature engineering, EDA

Source `payload_full.csv`: 31,067 rows, columns ['payload', 'length', 'attack_type', 'label'].
attack_type counts:

```
attack_type
norm              19304
sqli              10852
xss                 532
path-traversal      290
cmdi                 89
```

**Task filter** attack_type in {sqli, norm}: 30,156 rows (10,852 sqli, 19,304 norm).

## Cleaning
- Lowercased + stripped whitespace (identical to the Kaggle-set preprocessing).
- Removed 0 empty/missing payloads.
- Label-conflicting duplicates (same string labelled both sqli and norm): 0 distinct strings -> dropped (ambiguous).
- Exact duplicate strings removed: 0.
- **Clean dataset: 30,156 rows** (10,852 SQLi, 19,304 benign; 36.0% malicious).

## Feature engineering
- `Query_Length` (word count) and `char_len` (character count).
- Special-character counts (quote, =, parenthesis, %, dash) as EDA descriptors.

## Exploratory data analysis
Query length (words) by class:

```
           mean  50%   max
Label                     
0      1.467312  1.0  11.0
1      8.651124  7.0  29.0
```
99th-percentile query length: 20 words; max 29 words, 280 chars.
Mean special chars (SQLi vs benign): quote 1.24 vs 0.00; = 1.20 vs 0.00; ( 3.90 vs 0.00.

Figures: `httpparams_01_class_distribution.png`, `httpparams_02_length_by_class.png`, `httpparams_03_charlen_by_class.png`.

## Class-imbalance handling
The set is 64.0% benign / 36.0% malicious (mild imbalance). We handle it with **cost-sensitive learning (balanced class weights in the loss)**, not resampling, and we judge models on **imbalance-robust metrics** (malicious recall, false-positive rate, F1, MCC, balanced accuracy) rather than raw accuracy.
- *Technique:* `compute_class_weight('balanced')` weights the minority (SQLi) class inversely to its frequency inside the GNN's class-weighted cross-entropy and the logistic-regression head; the MLP head, which sklearn cannot class-weight directly, is handled at the evaluation level.
- *Why not resampling:* SMOTE interpolates in feature space, but there is no meaningful interpolation between two SQL query strings, and a synthetic BERT embedding between two queries corresponds to no real query - it injects artefacts, not signal. Random oversampling merely duplicates rows (overfitting risk); undersampling would discard about 40% of real benign data for a mild imbalance. Class weighting keeps every real example and fixes the loss asymmetry directly.
- *Why these metrics:* at 64/36 a trivial all-benign classifier already scores ~64% accuracy, so accuracy is misleading; malicious recall (attack detection), FPR (false alarms), F1, MCC and balanced accuracy are where imbalance actually bites, and they are the fair basis for comparing the GNN against BERT-only.

## Output
Clean modelling dataset written to `httpparams_sqli.csv` (30,156 rows, columns Query/Label).
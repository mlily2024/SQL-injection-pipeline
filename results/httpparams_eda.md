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

## Output
Clean modelling dataset written to `httpparams_sqli.csv` (30,156 rows, columns Query/Label).
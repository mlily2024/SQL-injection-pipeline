# Edge ablation: does graph structure contribute?

Structure-GNN (config {'hid': 169, 'drop': 0.15579754426081674, 'lr': 7.52374288453485e-05, 'bs': 67}) retrained with original / no / random edges, 3 seeds each. Clean test accuracy (%). If none and random match original, the graph topology is inert.

| Edge condition | Mean acc (%) | Per-seed |
|---|---|---|
| original | 99.67 +/- 0.03 | [99.64, 99.67, 99.69] |
| none | 99.65 +/- 0.03 | [99.64, 99.68, 99.62] |
| random | 99.59 +/- 0.02 | [99.57, 99.61, 99.58] |
# Experiment 1: Boost=true

**Metadata**
- CPU Hz: 480000000
- Frame/Hop: 512 / 128
- STFT bins/frames: 257 / 5
- Feature channels: 9
- Model params: 859053
- Weight bytes: 3436212
- Scratch bytes: 1472676
- Static total bytes: 4908888
- Warmup/bench: 5 / 20
- Compiler Optimizations: -O2
- Weights Location: QSPI

**Metrics**
|Case|min cyc|avg cyc|max cyc|min us|avg us|max us|avg hop budget|max abs err|rmse|
|---|---|---|---|---|---|---|---|---|---|
|Zero| 134344255 | 135168699 | 135575007 | 279883.864 | 281601.456 | 282447.931 | 10560.05% | 0.0 | 0.0 |
|Impulse| 133524175 | 134290033 | 134982663 | 278175.364 | 279770.902 | 281213.881 | 10560.05% | 0.0 | 0.0 |
|Random| 136208827 | 137029543 | 138208495 | 283768.389 | 285478.214 | 287934.364 | 10705.43% | 0.0 | 0.0 |

# Experiment 2: -O3 -ffast-math
**Metadata**
- CPU Hz: 480000000
- Frame/Hop: 512 / 128
- STFT bins/frames: 257 / 5
- Feature chans: 9
- Model params: 859053
- Weight bytes: 3436212
- Scratch bytes: 1472676
- Static total bytes: 4908888
- Compiler Optimizations: -O3 -ffast-math
- Weights Location: QSPI

|Case|min cyc|avg cyc|max cyc|min us|avg us|max us|avg hop budget|max abs err|rmse|
|---|---|---|---|---|---|---|---|---|---|
|Zero| 105296319 | 105869290 | 106648751 | 219367.331 | 220561.020 | 222184.897 | 8271.03% | 0.0 | 0.0 |
|Impulse| 103999127 | 104848747 | 105502279 | 216664.847 | 218434.889 | 219796.414 | 8191.30% | 0.0 | 0.0 |
|Random| 107231577 | 107787570  | 108203207 | 223399.118 | 224557.437 | 225423.347 | 8420.90% | 0.0 | 0.0 |


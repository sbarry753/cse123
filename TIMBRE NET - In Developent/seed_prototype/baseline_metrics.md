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

# Experiment 3: -O3 -ffast-math SDRAM
**Metadata**
- CPU Hz: 480000000
- Frame/Hop: 512 / 128
- STFT bins/frames: 257 / 5
- Feature channels: 9
- Model params: 859053
- Weight bytes: 3436212
- Scratch bytes: 4910936
- Static total bytes: 8347148
- Compiler Optimizations: -O3 -ffast-math
- Weights Location: SDRAM

|Case|min cyc|avg cyc|max cyc|min us|avg us|max us|avg hop budget|max abs err|rmse|
|---|---|---|---|---|---|---|---|---|---|
|Zero| 161249769 | 162130716 | 162673937 | 335937.018 | 337772.325 | 338904.035 | 12666.46% | 0.0 | 0.0 |
|Impulse| 160754913 | 161140860 | 161877181 | 334906.068 | 335710.125 | 337244.127 | 12589.12% | 0.0 | 0.0 |
|Random| 163274432 | 163776221 | 164525021 |340155.066 | 341200.460 | 342760.460 | 12795.01% | 0.0 | 0.0 |

# Experiment 4: -O3 -ffast-math int8 Quantization TFLM
**Metadata**
- CPU Hz: 480000000
- Frame/Hop: 512 / 128
- STFT bins/frames: 257 / 5
- Feature channels: 9
- Activation note: quick-gelu approx in exported TFLite mirror
- UNet model bytes: 932584
- Transient model bytes: 29912
- DSP scratch bytes: 157132
- Reserved arena bytes: 3407872
- Static reserved total: 4527500
- UNet arena used/reserved: 193628 / 3145728
- Transient arena used/reserved: 54916 / 262144
- Compiler Optimizations: -O3 -ffast-math
- Weights Location: QSPI
- int8 quantization

|Case|min cyc|avg cyc|max cyc|min us|avg us|max us|avg hop budget|max abs err|rmse|
|---|---|---|---|---|---|---|---|---|---|
|Zero| 1327903565 | 1328184965 | 1328621069 | 2766465.760 | 2767052.010 | 767960.560 | 103764.45% | 1.336782 | 0.913601 |
|Impulse| 1327792401 | 1328205556 | 1328664525 | 2766234.168 | 2767094.908 | 2768051.093 | 103766.05% | 0.215599 | 0.029597
|Random| 1327907661 | 1328198052 | 1328492277 | 2766474.293 | 2767079.275 | 2767692.243| 103765.47% | 0.119147 | 0.035504

**Case: zero**

avg stage us: pre=9427.447 unet=2671944.408 recon=3663.522 transient=81983.093

**Case: impulse**

avg stage us: pre=9388.445 unet=2672816.268 recon=3724.139 transient=81132.537

**Case: random**

avg stage us: pre=9444.110 unet=2672239.333 recon=3741.095 transient=81620.352

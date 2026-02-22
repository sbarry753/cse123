# Benchmarking

## Overview
This directory contains the strategy and code for benchmarking the Daisy Seed. The benchmarks will test multiple effects, each requiring a different computational demand. Different block sizes will be tested for each effect.

## Tests to Run
For each effect + block size, the min, max, and avg CPU load in the callback will be measured. The CPU load is the ratio between time spent in the callback by the maximum time available per block. As the audio callback must finish processing a block before the next set of samples are ready to be processed, there is a hard deadline for the processing time of the callback. This test determines if a given effect + block size is able to process an entire block before the next block is ready. 

Estimating the CPU load for an effect is given by:

- $f_s$ : Sample rate
- $t_{per\ sample}$ : Processing time per sample
- $t_{overhead}$ : Callback overhead
- $N$: Number of samples (block size)

$$
CPU\ Load \approx f_s \cdot t_{per\ sample} \ + \ \frac{f_s \cdot t_{overhead}}{N}
$$

From this model, we can see that there is a fixed cost as block size increases ($f_s \cdot t_{per\ sample}$) for an effect. The right side of the equation tells us how block size scales the CPU load. As the block size increases, the CPU load decreases. However, this would increase latency. 

Simple effects have a low fixed cost because ($t_{per sample}$) is low. For increasingly complex effects, this cost is greater. Therefore, for a given CPU load, simpler effects can have a smaller block size than more computationally expensive effects. However, higher block sizes results in higher latency, so we must find an optimal block size that preserves the target latency while still being able to fully process all samples through the effect. 

After all tests have been run and data has been collected, we should have a good idea on the maximum complexity of an effect such that the latency is in an acceptable range and no audio errors occur.


Additionally, for each experiment, the memory usage of the all memory regions will be documented. With this information, we can see if used RAM or Flash memory can bottleneck the complexity of the pedal. 

### Daisy Seed Memory Regions
| Memory Region | Region Size |
|--------------|-------------|
| FLASH        | 128 KB      |
| DTCMRAM      | 128 KB      |
| SRAM         | 512 KB      |
| RAM_D2       | 288 KB      |
| RAM_D3       | 64 KB       |
| BACKUP_SRAM  | 4 KB        |
| ITCMRAM      | 64 KB       |
| SDRAM        | 64 MB       |
| QSPIFLASH    | 8 MB        | 

## Effects
- Passthrough - Baseline
    - Audio remains unchanged, directly passing input to output
- Simple 
    - Distortion
    - Synthesizer (sine wave)
- Medium
    - Phaser
    - Chorus
- Complex
    - ReverbSc
    - PitchShifter
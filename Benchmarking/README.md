# Benchmarking

## Overview
This directory contains the code for benchmarking the Daisy Seed. The benchmarks will test multiple effects, each requiring a different computational demand. Additionally, different block sizes and samples rates will be tested for each effect.

## Tests to Run
For each effect + sample rate + block size, the min, max, and avg CPU load in the callback will be measured. The CPU load is the ratio between time spent in the callback by the maximum time available per block. As the audio callback much finish processing a block before the next set of samples have finished being buffered, there is a hard deadline for the processing time of the callback. This test determines if a given effect + sample rate + block size is able to process an entire block before the next block is ready. After all tests have been run and data has been collected, we should have a good idea on the maximum complexity of an effect such that the latency is in an acceptable range and no audio errors occur.

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
- Passthrough
    - Audio remains unchanged, directly passing input to output

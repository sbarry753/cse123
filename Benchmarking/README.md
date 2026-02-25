# Benchmarking

## Overview
This directory contains the strategy and code for benchmarking the Daisy Seed. The benchmarks will test multiple effects, each requiring a different computational demand. Different block sizes will be tested for each effect.

For each effect + block size, the min, max, and avg CPU load in the callback will be measured. The CPU load is the ratio between time spent in the callback by the maximum time available per block. As the audio callback must finish processing a block before the next set of samples are ready to be processed, there is a hard deadline for the processing time of the callback. This test determines if a given effect + block size is able to process an entire of samples before the next samples are ready. 

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

## Effects
The effects were mostly sourced from *DaisyExamples* with slight modifications.

- Passthrough - Baseline
- Simple 
    - Distortion
    - Bitcrush
- Medium
    - Delay
    - Chorus
- Complex
    - Reverb
    - PitchShifter

## Block Sizes
- 16
- 32
- 48
- 64
- 96
- 128
- 256

Note that the max block size supported by the Daisy seed is 256.

## Experiment Strategy
The mentioned block sizes will be tested for each effect. The sample rate across all experiments is *48 kHz*. Each effect processes the same input in its *process_audio()* method, a 220 Hz since wave with an amplitude of 0.4. Note that this signal is generated within the audio callback function, which differs from how the input is handled in the finished pedal. For the finished pedal, the MCU will process an instrument's input via *I2S*, then send it to the audio callback for processing. However, the *Passthrough* experiment only shows a maximum of approximately 1% CPU Load across all block sizes, so the extra processing load is minimal. Regardless, the callback needs to leave some headroom (max $\approx$ 70-80%) to protect against latency spikes so the MCU can still process the buffer in time before the next samples are ready. Essentially, generating input inside the callback is justified because we will cap CPU load at a specific point and can subtract out the minimal computation the generation introduces. The *CPULoadMeter cpu_load* object calculates the min, max, and avg CPU load between the object's *OnBlockStart()* and *OnBlockEnd()* methods. The benchmark also processes ADC input from one of the potentiometers, simulating a musician tuning their effect to get their desired output sound. Each experiment will track the CPU load metrics in a spreadsheet. 

## Results
[in progress]
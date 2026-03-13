# Status Report 4

**Authors:** Yasmeen Amani, Zion, Kai, Barry, Nick, Aaryan <!-- yall want to insert your last names ? -->

**Date:** February 2026

---

## Collaboration System Links

Github: https://github.com/sbarry753/cse123

---

## Checklist

(-) Not Done (+) Done (=) In Progress ~ Personas

\+ Need Statement

\+ Goal Statement

\= Design Objective

\= Conceptualisations

\= Gantt Chart

\= Aesthetic and Functional Prototypes

\- Design for Manufacture and Assembly

\= Test Plan

\- Test Report

\- Life Cycle Assessment

\- Ethics Statement

\- 3D CAD of final deliverable

---

## Timeline

We have been revising and finalizing our timeline to have a working prototype by week 11. Each team have made lists of key requirements to finish by specified weeks. We are prioritizing functionality within our respective teams and collaborating on aesthetic choices together.

---

## Weekly Progress Summary

### Benchmarking

This week, we started benchmarking the Daisy Pod only since the Teensy boards appear to have been lost in transit. As the only difference between the Daisy Seed and Teensy 4.0 is the Teensy has a marginally faster clock speed, we believe the Daisy board will be capable to handle the complexity of our effect.

Now, our main benchmarking attention is spent on testing the Daisy seed with multiple effects, varying in computational demand, with different block sizes and sample rates. For each effect + sample rate + block size, the min, max, and avg CPU load in the callback will be measured. These tests should indicate the maximum complexity of an effect the Daisy seed can support such that the latency is within the acceptable range (8ms) and no audio errors occur. Additionally, for each experiment, the memory usage of the all memory regions (SRAM, flash, etc). will be documented. With this information, we can see if used RAM or Flash memory can bottleneck the complexity of the pedal effect. This weekend, Kai will benchmark the Daisy pod with these strategies in mind.

### Analog effects

Zion has completed a functioning prototype of an analog overdrive effect, based on the Tascam 424 preamp. The effect is intended to be used on the output of the digital synth, or on the guitar signal. Once the synthesizer portion is completed, the EQ section, input gain, and op-amp selection of the analog effect could be tweaked to be more synergistic, and pleasing to listen to.

---

## Current Personas + Clientele

People who play guitar and want to expand the tonal capablilities of their instrument.

**Current Need Statement**
"Articulate the need as an expression of dissatisfaction with the current situation"

A guitar only makes a guitar sound, and not the sound of other instruments or synthesizers. A musician may have developed intuitive control of the guitar, but cannot easily transfer it to a keyboard or other instrument.

---

## Current Goal Statement

"A brief, general, and ideal response to the need statement"

Design an enviable guitar pedal that detects the notes a guitarist is playing, and plays the same notes through a digital synthesizer.

---

## Decisions Made This Week

- Moving forward with building around daisy seed because Teensy is delayed in shipping
- Chose to make device stand-alone and not a smart updatable product

---

## Test Plan

![Team Testing](zion.png)
*Figure 1: Team Testing*

We made lists of different tests we would perform to make sure all bases were covered. There are plans to test not only functionality of software, hardware, and analog, but also UX, error handling, and setup instructions. Tests will be ran with special attention to our approach, execution, and documentation.

![Chalkboard Test Plans](test.png)
*Figure 2: Chalkboard Test Plans*

---

## References

[1] Electrosmith, "Daisy Programmer," 2022. [Online] https://flash.daisy.audio/.

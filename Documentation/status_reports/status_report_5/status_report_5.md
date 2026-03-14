# Status Report 5

**Authors:** Yasmeen Amani, Zion, Kai, Barry, Nick, Aaryan
<!-- yall want to insert your last names ? -->

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

\- Design document draft

---

## Weekly Progress Summary

### Benchmarking

The Daisy Seed effect benchmarking was completed this week with the goal determining how complex of an effect is possible and at which block size to keep latency within the acceptable range. In the experiments, we found that the MCU was more than capable of handling effects inspired by the DaisyExamples repository. The greatest Maximum CPU Load measured occurred on a Reverb effect with a block size of 8 at approximately 14\%. This shows how well the MCU can support DSP-related workflows and how optimized the DaisySP library is for audio effects. The full results are documented in this [spreadsheet](https://docs.google.com/spreadsheets/d/1QgfL2nDbECkWdAaCYBFUneOOQhCu9Kp85AZZbPZRD8g/edit?usp=sharing). However, we still want to determine the limits of the Daisy Seed, so we plan on stress-testing it with matrix multiplication. Matrix multiplication is a particularly relevant benchmark because it is computationally demanding and it is used heavily in neural networks, which we are implementing for note/chord detection. We will test different matrix/block sizes and different matrix multiplication algorithms (differing by optimization level).

Furthermore, the memory usage tracking across different effects (mentioned in the last status report) were removed from benchmarking because the choice of effect had no effect on the binary size. The effects and code infrastructure were relatively simple, so the memory usage was trivial. The initial prototype software suite will be much larger.

### Analog effects

Both overdrive circuits were tested with an electric guitar. There was a preference among the group for the sound overdrive circuit based on the Tascam 424 pedal. This week we also began making a schematic and PCB for the overdrive circuit based on a Wampler design. It is uploaded on our Github. We have decided what we are going to present for the initial demo and are working to have the preamp, daisy pod, and overdrive circuit.

### Software: Daisy Pod Effects and Neural Net

---

## Current Personas + Clientele

People who play guitar and want to expand the tonal capabilities of their instrument.

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
- Chose to make device stand-alone and not a product that has an app or receives updates after purchase
- Plan to order a industry standard metal pedal enclosure from Love My Switches

---

## Test Plan

### Overdrive Clipping

We plan to test that our overdrive circuit is indeed clipping with an oscilloscope and waveform generator. We want to have a soft-clipping effect that pairs well with the effects coming from the synth.

### Overdrive Sound

The overdrive circuit also has to sound good to our ears and the ears of select musicians testing our pedal. We are currently going through this stage of testing with just our guitarist team members and their various guitars and amps. We have multiple op amps and other analog components to test and create an overdrive circuit that excites us and our clientele.

### Embedded System Testing

Currently working on a test plan for the Daisy Seed to make sure it can handle real-time audio processing. Setting tests to check buffer size, polyphonic, sample rate vs latency, and CPU usage. Wrote a program to test full signal chain while monitoring CPU load. Will test by this week when guitar is acquired.

### Demo Prep Test

We plan to meet again this weekend to wire up all sub assemblies of the pedal and test it with a guitar and amp.

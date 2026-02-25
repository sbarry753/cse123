#### Collaboration System Links
Github: [https://github.com/sbarry753/cse123](https://github.com/sbarry753/cse123) 

#### Weekly Progress Summary
This week we….

Meeting 1 (2/2/26)
- Discussed muti-effect vs single-effect pedal
- Formulated roles and went over skill sets
- Roles decided: Analog, Embedded Systems, Artistic Director

Meeting 2 (2/3/26)
- Went over status report submission guidelines with the Karan. He suggested recording: How much time was spent working and which person did which aspect? 
- TA proposed questions to consider; What has been done? What is in progress? What work needs to be completed in the future?

Meeting 3 (2/4/26)
- Barry and Nick met to deliver the first Daisy Pod to begin designing the initial prototype of the digital effect 

Meeting 4 (2/5/26)
- Spent time with Prof Harrison to talk about team management strategies
- The sign of a healthy team is one that assigns tasks to themselves
- Keep the kanban board simple with 4 categories maximum (backlog, todo, in progress, done)
- Showed off Nick’s demo of the digital effect so far
- Confirmed that Prof Harrison is a musician and specifically plays guitar, bass and piano
- He doesn’t use effects but he likes the sound of amps from British brands like Vox

#### Checklist 
(-) Not Done (+) Done (~) In Progress

~ Personas

\+ Need Statement

\+ Goal Statement

\- Design Objective

\- Conceptualisations

~ Decision Table

~ Basic Plan - Gantt Chart

\- Aesthetic & Functional Prototypes

\- Design for Manufacture & Assembly

\- Test Plan

\- Test Report

\- Life Cycle Assessment

\- Ethics Statement


#### Current Personas
People who play guitar but want common synth effects like wave folding specifically made for guitars in a guitar pedal form factor
#### Current Need Statement
There exists no guitar pedal or it is uncommon that guitar pedals exist that either replicates a specific cool vintage pedal/amp/speaker or does this certain effect using digital and analog components
#### Current Goal Statement
Design an enviable guitar pedal that digitally (or analog + digital) recreates a vintage effect or rare/new/uncommon effect with low enough latency that the effect can be used for live performance and fits within a standard guitar pedal enclosure
#### Decisions Made This Week
- Purchased Daisy Seed + Daisy Pod and Teensy 4.0 + Teensy Audio Shield to test out these platforms for digital effects creation
- Began to outline roles and assign them amongst the team members most interested
- Started a github repo and github project

#### Design Plan Decision Defense
We chose the Daisy Seed and Teensy 4.0 platforms because they match our need for processing guitar effects. The Daisy Seed’s purpose is to be used in audio hardware devices and is open-source so that was a clear option for us to try. Teensy 4.0 has an audio library and audio board. This is a fast microcontroller with low latency and real time digital signal processing. From here we will evaluate each microcontroller and compare which choice is superior for our project's needs. Specifically focusing on comparing latencies.

#### Morphological Chart Rough Draft
![](morphological_chart.png)

#### Current Research Explored
This week we looked at…..

**Different Effects**
- Wave folding
- distortion
- fuzz
- overdrive

Through our research with Daisy Seed, in performing a wavefolding sound effect for the pedal, there's a few concerns we have to consider. The general flow of our design would be something like this:

Guitar Sound --> Input Buffer --> Preamp Gain --> ADC --> DAC --> Reconstruction Filter --> Output Buffer --> Amp (as seen in Status Report #1)

The Daisy Seed microcontroller provides built-in ADC-DAC control, however everything else would be connected to the PCB and may require analog controls to interact with the Daisy Pod. We may have to add or remove some things during the prototyping process as well.

Some concerns with the wavefolding effect we are aiming for is that transmitting the guitar signal, since it's a high-impedance source, needs to cleanly be transmitted and attunated through the Daisy Seed.  

**Trying to find acceptable latency (extremely important design constraint)**
-  How to score latency (likely a nonlinear response curve of latency -> “perceived quality”)
- Designing an experiment to measure latency (in progress)
- Requires a Y-splitter cable or stereo out device, audio interface with 2+ inputs, digital audio workstation… all of which are on hand
- Jitter (variation in latency) is an additional concern
- Lower is better, 10ms (w/~1ms jitter) is described as a target for digital musical instruments
- Sources
	- R. H. Jack, A. Mehrabi, T. Stockman, and A. McPherson, “Action-sound Latency and the Perceived Quality of Digital Musical Instruments,” Music Perception, vol. 36, no. 1, pp. 109–128, Sep. 2018, doi: [10.1525/mp.2018.36.1.109](https://doi.org/10.1525/mp.2018.36.1.109).
	- A. Schmid, M. Ambros, J. Bogon, and R. Wimmer, “Measuring the Just Noticeable Difference for Audio Latency,” in Audio Mostly 2024 - Explorations in Sonic Cultures, Milan Italy: ACM, Sep. 2024, pp. 325–331. doi: [10.1145/3678299.3678331](https://doi.org/10.1145/3678299.3678331).

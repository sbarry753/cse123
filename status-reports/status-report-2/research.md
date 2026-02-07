## Research

Through my research with the Daisy Seed, in performing a wavefolding sound effect for the pedal, there's a few concerns we have to consider. The general flow of our design would be something like this:

This flow was seen in Status Report #1
Guitar Sound --> Input Buffer --> Preamp Gain --> ADC --> DAC --> Reconstruction Filter --> Output Buffer --> Amp

The Daisy Seed microcontroller provides built-in ADC-DAC control, however everything else would be connected to the PCB and may require analog controls to interact with the Daisy Pod. We ay have to add or remove some things during the prototyping process as well.

Some concerns with the wavefolding effect we are aiming for is that transmitting the guitar signal, since it's a high-impedence source, needs to cleanly be transmitted and attunated through the Daisy Seed.

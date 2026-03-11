Guitar is a polyphonic instrument, but accurate, generalized polyphonic note detection does not yet exist. We hope to design accurate digital polyphonic note detection for the restricted input space of the clean direct input of an electric guitar.

For an intuitive and pleasurable experience, the pedal must have a consistent low latency (<10ms). The musician should not notice a delay between playing a note on the guitar, and the synthesized sound coming out of their guitar amplifier. This places a demanding performance requirement on the final product.

For our effect to fit in a guitar pedal form factor, we will use an embedded microcontroller to run the note detection software. Therefore, all software functions (note detection, synthesizer, digital audio effects) must be computationally lightweight and memory efficient.

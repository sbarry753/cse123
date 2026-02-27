#Organizational

Files we will be using:  
1) Makefile  
2) main.cpp  
3) {effect}.cpp // Deprecated since we can use inline in .h to optimize
4) {effect}.h  

Makefile for main.cpp, which takes in the cpp effects and header

#ToDo

Bool logic on unity-gain to prevent unnecessary attunation if not needed.

When porting to daisy seed, use numerical/enum for analog inputs to special_effects

#Tests

DryGuitar.wav : Duration, 1:24
real	0m1.330s
user	0m1.278s
sys	0m0.046s

#Guidelines

0 - distortion
1 - fuzz
2 - wavefolding

#Get Started!

./effect_processor /app/special_effects/test_files/DryGuitar.wav

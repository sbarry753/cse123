# Model Distillation

This file outlines the process of distilling the model from `TIMBRE NET - In Developent`, partially using the framework provided by **Analog Devices** from training and synthesis *(see [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) and [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis))*

## Distilled Model Architecture
The small student model is defined in `TIMBRE NET - In Developent/model_distilled.py`. Currently, it is a simple 5-layer CNN. Each layer has a 3x3 kernel with a stride and padding of 1. 

## TODO Steps
1. Train model with ai8x-training train.py
2. Quantize with ai8x-synthesis quantize.py
3. Generate code with ai8x-syntheses ai8xize.py


python train.py --device MAX78000 --model timbrestudent --dataset MAXGuitarPiano --data ../../TIMBRE\ NET\ -\ In\ Developent/data --regression --use-bias -j 0 
# Model Distillation

This file outlines the process of distilling the model from `TIMBRE NET - In Developent`, partially using the framework provided by **Analog Devices** from training and synthesis *(see [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) and [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis))*

python train.py --device MAX78000 --model timbrestudent --dataset MAXGuitarPiano --data ../../TIMBRE\ NET\ -\ In\ Developent/data --regression --use-bias -j 0 
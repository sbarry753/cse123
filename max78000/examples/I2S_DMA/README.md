## Description

This application demonstrates receiving data from the microphone on the MAX78000 EV Kits using the I2S and DMA modules. Once valid data is received, a status message is printed confirming the example succeeded.

## Software

When testing harware, will add audio output, change sample rate, config line input and add passthrough. 

## Setup

If using the MAX78000FTHR (FTHR_RevA)
-   Connect a USB cable between the PC and the CN1 (USB/PWR) connector.
-   Open a terminal application on the PC and connect to the EV kit's console UART at 115200, 8-N-1.

## Expected Output

The Console UART of the device will output these messages:

```
***** I2S Receiver Example *****

Microphone enabled!
Receiving microphone data!
```


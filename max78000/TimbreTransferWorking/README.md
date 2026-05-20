# TimbreTranser - MAX78000 FreeRTOS Audio Processing 

## Current functionality
- Audio passthrough, line in/out working on MAX78000FTHR
- FreeRTOS circular buffer integrated and tested
- Has audio processing and buffer demo running simultaneous
- !! NEURAL NETROK INTERFACE PENDING !!

## Hardwware Requirements
- MAX78000FTHR board
- (2) 3.5 mm auxiliary cables
- audio source and audio output device

## Setup development environment

### 1. Update MAX78000 debugger firmware 
 following the instructions [here](https://github.com/analogdevicesinc/MaximAI_Documentation/blob/main/MAX78000_Feather/README.md)

### 2. Install MaxiSDK
- [Install MaximSDK](https://analogdevicesinc.github.io/msdk/USERGUIDE/#installation)
- [setup with Visual Studio Code](https://analogdevicesinc.github.io/msdk/USERGUIDE/#getting-started-with-visual-studio-code)
- Install [Serial Monitor](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-serial-monitor) extension

### 3. **CRITICAL: Replace the MAX9867 driver**
    - Backup `max9867.c/h` in `Libraries/MiscDrivers/CODEC` in the MaximSDK installation folder.
    - Replace `max9867.c/h` with the versions in the `driver` folder contained within *this* folder. Each time we update the driver code, this process must be repeated. (Probably we want to not perform this process later on)

### 4. Build and Flash 
Connect MAX78000FTHR board, and monitor it using the Serial Monitor extension; build and flash using [this](https://analogdevicesinc.github.io/msdk/USERGUIDE/#visual-studio-code) guide.

## Testing Audio Passthrough
1. Connect audio source (computer/phone) → J5 (line-in jack)
2. Connect speaker/headphones → J7 (line-out jack)
3. Play audio - you should hear it through the speaker
4. LED2 should blink at ~1Hz (indicates DMA running)

## Circular Buffer
The project includes a FreeRTOS-compatible circular buffer (`buffer.c`, `buffer.h`) ported from POSIX semaphores written by David Harrison. Demo tasks (producer/consumer) run alongside audio passthrough to verify functionality.
 
## SDK Bugs
1. **Codec driver broken** - Fixed by patched driver (see setup step 3)
2. **I2C not initialized** - Driver assumes I2C already initialized ([Issue #762](https://github.com/analogdevicesinc/msdk/issues/762))

## Debugging in the command line
To step through the program in GDB, follow this sequence:
- Connect the MAX78000FTHR board, and monitor it using the Serial Monitor extension.
- Open a terminal window and enter *this* directory.
- Run `make clean`, then `make all`, then `make flash-hold`.
- Open a new terminal window, without closing the previous one, and enter this directory. In the new terminal window, run `make gdb-debug`. You should now be in GDB, with the program paused at `main()`. [Reference for GDB commands](https://visualgdb.com/gdbreference/commands/)

## Contributors
- Zion - Initial audio passthrough setup
- Yasmeen - Cicular buffer port, SDK bugs, documentation

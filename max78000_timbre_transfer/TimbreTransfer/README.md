## Current functionality
At the moment, the program passes audio straight through without affecting it.

## Setup development environment
1. Update MAX78000 debugger firmware by following the instructions [here](https://github.com/analogdevicesinc/MaximAI_Documentation/blob/main/MAX78000_Feather/README.md)
2. [Install MaximSDK](https://analogdevicesinc.github.io/msdk/USERGUIDE/#installation), and [setup with Visual Studio Code](https://analogdevicesinc.github.io/msdk/USERGUIDE/#getting-started-with-visual-studio-code)
3. Install [Serial Monitor](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-serial-monitor) extension
4. Replace the MAX9867 driver
    - Backup `max9867.c/h` in `Libraries/MiscDrivers/CODEC` in the MaximSDK installation folder.
    - Replace `max9867.c/h` with the versions in the `driver` folder contained within *this* folder. Each time we update the driver code, this process must be repeated. (Probably we want to not perform this process later on)
5. Connect MAX78000FTHR board, and monitor it using the Serial Monitor extension; build and flash using [this](https://analogdevicesinc.github.io/msdk/USERGUIDE/#visual-studio-code) guide.

## Debugging in the command line
To step through the program in GDB, follow this sequence:
- Connect the MAX78000FTHR board, and monitor it using the Serial Monitor extension.
- Open a terminal window and enter *this* directory.
- Run `make clean`, then `make all`, then `make flash-hold`.
- Open a new terminal window, without closing the previous one, and enter this directory. In the new terminal window, run `make gdb-debug`. You should now be in GDB, with the program paused at `main()`. [Reference for GDB commands](https://visualgdb.com/gdbreference/commands/)
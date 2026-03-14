# Daisy Seed/Pod 

## Description
Daisy Skeleton Code

## Prerequisites
- VS Code or [Daisy Web Programmer](https://flash.daisy.audio/)
- [Toolchain/DaisyExamples Installation](https://daisy.audio/tutorials/cpp-dev-env/)
    - Follow the Getting Started tutorial
- *libDaisy* and *DaisySP* are submodules of the project
    - If this is your first time cloning, you must run it with the **--recurse-submodules** flag to install the libraries
    - If you have already cloned and need to pull, to install the libraries you must run 
    ```bash
    git submodule update --init --recursive
    ```
    - After installing the submodules, head to each library in *lib/* and run *make* to build them
    ```bash
    cd src/daisy/lib/libDaisy && make
    cd src/daisy/lib/DaisySP && make
    ```


## Usage
- If on Pod, set *C_DEFS* in the *Makefile* to *POD*
- If on Seed, set *C_DEFS* in the *Makefile* to *SEED*
- Build: **make build**
- Flash: **make program-dfu**
    - When flashing, make sure to put the Daisy in **bootloader mode**
        - Connect the Seed to your computer with the USB Micro cable.
        - Press and hold the Boot button
        - Press and hold the Reset button
        - Let go of the Reset button
        - Let go of the Boot button

- Clean: **make clean**


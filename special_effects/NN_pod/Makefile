TARGET = daisy

# Sources
CPP_SOURCES = main.cpp

# Target hardware
C_DEFS = -DPOD -DUSE_DAISY_POD

# Put the program in external QSPI (8MB)
APP_TYPE = BOOT_QSPI

LIBDAISY_DIR = ../../lib/libDaisy/
DAISYSP_DIR  = ../../lib/DaisySP/

# ---- CMSIS-DSP (build FFT from source; no prebuilt lib needed) ----
CFLAGS   += -DARM_MATH_CM7 -DUSE_ARM_MATH
CXXFLAGS += -DARM_MATH_CM7 -DUSE_ARM_MATH

C_INCLUDES += -I../../lib/libDaisy/Drivers/CMSIS-DSP/Include
C_INCLUDES += -I../../lib/libDaisy/Drivers/CMSIS-DSP/PrivateInclude

# FFT/RFFT implementation
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_cfft_f32.c
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_cfft_init_f32.c
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/TransformFunctions/arm_bitreversal2.c

# Constant structs + required twiddle/bitrev tables
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/CommonTables/arm_const_structs.c
C_SOURCES += ../../lib/libDaisy/Drivers/CMSIS-DSP/Source/CommonTables/arm_common_tables.c
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

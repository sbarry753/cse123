TARGET = daisy

# Sources
CPP_SOURCES = main.cpp tflite_onset_runner.cpp
CPP_STANDARD = -std=gnu++17

# Target hardware
C_DEFS = -DPOD -DUSE_DAISY_POD -DARM_MATH_LOOPUNROLL -DTF_LITE_STATIC_MEMORY

OPT = -O3

# Put the program in external QSPI (8MB)
APP_TYPE = BOOT_QSPI

LIBDAISY_DIR = ../../lib/libDaisy/
DAISYSP_DIR  = ../../lib/DaisySP/
TFLM_DIR = ../../lib/tflite-micro
TFLM_BUILD_DIR = $(TFLM_DIR)/gen/cortex_m_generic_cortex-m7+fp_release_with_logs_cmsis_nn_gcc
TFLM_LIB = $(TFLM_BUILD_DIR)/lib/libtensorflow-microlite.a

# ---- CMSIS-DSP (build FFT from source; no prebuilt lib needed) ----
CFLAGS   += -DARM_MATH_CM7 -DUSE_ARM_MATH
CXXFLAGS += -DARM_MATH_CM7 -DUSE_ARM_MATH

C_INCLUDES += -I../../lib/libDaisy/Drivers/CMSIS-DSP/Include
C_INCLUDES += -I../../lib/libDaisy/Drivers/CMSIS-DSP/PrivateInclude
C_INCLUDES += -I$(TFLM_DIR)
C_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
C_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp
C_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy
C_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/kissfft
C_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Include
C_INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/cmsis_nn/Include

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

LIBS += $(TFLM_LIB)
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

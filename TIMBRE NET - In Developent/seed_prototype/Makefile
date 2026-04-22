# Project Name
BENCHMARK ?= baseline
TARGET = nn_daisy_$(BENCHMARK)

# Sources
CPP_SOURCES = $(BENCHMARK).cpp

# Optimization level
OPT = -O3

# Put the program in external QSPI (8MB)
APP_TYPE = BOOT_QSPI

# For weights to be stored in SDRAM instead of QSPI for raw-C baseline:
#   make BENCHMARK=baseline USE_SDRAM_WEIGHTS=1
USE_SDRAM_WEIGHTS ?= 0

LIBDAISY_DIR = ../../lib/libDaisy/
DAISYSP_DIR = ../../lib/DaisySP/
TFLM_DIR = ../../lib/tflite-micro
TFLM_GEN_DIR = $(TFLM_DIR)/gen/cortex_m_generic_cortex-m7+fp_release_with_logs_cmsis_nn_gcc
TFLM_DOWNLOADS_DIR = $(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# Faster float operations
CFLAGS += -ffast-math
CPPFLAGS += -ffast-math

CFLAGS   += -DARM_MATH_CM7 -DUSE_ARM_MATH
CXXFLAGS += -DARM_MATH_CM7 -DUSE_ARM_MATH

ifeq ($(BENCHMARK),baseline)
ifeq ($(USE_SDRAM_WEIGHTS),1)
CFLAGS += -DBASELINE_USE_SDRAM_WEIGHTS
CXXFLAGS += -DBASELINE_USE_SDRAM_WEIGHTS
endif
endif

ifeq ($(BENCHMARK),tflm_benchmark)
CPPFLAGS += -DTF_LITE_STATIC_MEMORY
CFLAGS += -DTF_LITE_STATIC_MEMORY

C_INCLUDES += \
	-I$(TFLM_DIR) \
	-I$(TFLM_DOWNLOADS_DIR)/flatbuffers/include \
	-I$(TFLM_DOWNLOADS_DIR)/gemmlowp \
	-I$(TFLM_DOWNLOADS_DIR)/ruy \
	-I$(TFLM_DOWNLOADS_DIR)/cmsis/CMSIS/Core/Include \
	-I$(TFLM_DOWNLOADS_DIR)/cmsis/CMSIS/NN/Include

LIBDIR += -L$(TFLM_GEN_DIR)/lib
LIBS += -ltensorflow-microlite
endif

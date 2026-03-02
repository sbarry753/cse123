# Project Name
TARGET = Benchmarking

# Sources
CPP_SOURCES = daisy_benchmark.cpp

# Needed for reverb
USE_DAISYSP_LGPL = 1

# Enable Matrix Multiplication Stress-test
C_DEFS = -DMAT_MUL

# Optimization Level
OPT = -O3

# Library Locations
LIBDAISY_DIR = ../lib/libDaisy
DAISYSP_DIR = ../lib/DaisySP

# CMSIS DSP
C_SOURCES = $(LIBDAISY_DIR)/Drivers/CMSIS-DSP/Source/MatrixFunctions/arm_mat_mult_f32.c \
			$(LIBDAISY_DIR)/Drivers/CMSIS-DSP/Source/MatrixFunctions/arm_mat_init_f32.c
# Core location, and generic Makefile.
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

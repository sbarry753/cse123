<<<<<<< HEAD
TARGET = effect_processor
CPP_SOURCES = main.cpp

LIBDAISY_DIR = libDaisy
DAISYSP_DIR = DaisySP
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core

include $(SYSTEM_FILES_DIR)/Makefile
=======
# The compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall

# Target executable name
TARGET = effect_processor

# All source files to compile
SRCS = main.cpp fuzz_effect.cpp distortion_effect.cpp wavefolding_effect.cpp

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean up compiled files
clean:
	rm -f $(TARGET) output.wav
>>>>>>> b786ae4c5ff541eba33bb3089403792ad699f197

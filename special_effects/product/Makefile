# The compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall

# Target executable name
TARGET = effect_processor

# All source files to compile
SRCS = main.cpp fuzz_effect.cpp

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean up compiled files
clean:
	rm -f $(TARGET) output.wav

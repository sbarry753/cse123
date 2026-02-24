# Default target: compile the program
fuzz_tester: main.cpp
	g++ main.cpp -o fuzz_tester -std=c++17

# Type 'make run' to compile (if needed) and then execute
run: fuzz_tester
	./fuzz_tester

# Type 'make clean' to delete the compiled executable
clean:
	rm -f fuzz_tester

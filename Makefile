# Compiler and Flags
CXX = g++
# Standard production flags (optimized)
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

# Debugging flags (comprehensive error checking)
# -g: Include debug symbols
# -O0: Disable optimizations for accurate stack traces
# -fsanitize=address: Detect memory errors (out-of-bounds, use-after-free, etc.)
# -fsanitize=undefined: Detect undefined behavior (null pointers, division by zero, etc.)
# -D_GLIBCXX_DEBUG: Enable bounds checking in STL containers (std::vector, etc.)
# -fstack-protector-all: Add stack smash protection
DEBUG_FLAGS = -std=c++17 -g -O0 -Wall -Wextra -Wpedantic \
              -fsanitize=address -fsanitize=undefined \
              -D_GLIBCXX_DEBUG -fstack-protector-all

# Target Executables
TARGET = fr_solver
TEST_TARGET = unit_tests

# Source and Header Files
SRC = main.cpp
TEST_SRC = test_main.cpp
HEADERS = basis.hpp parameters.hpp solver.hpp state.hpp

# Default Target
all: $(TARGET)

# Build the main executable (optimized)
$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Build the main executable with comprehensive debugging checks
debug: clean $(SRC) $(HEADERS)
	$(CXX) $(DEBUG_FLAGS) $(SRC) -o $(TARGET)
	@echo "Build complete: $(TARGET) compiled with ASan, UBSan, and STL Debugging enabled."

# Build and run the unit tests
test: $(TEST_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET)
	./$(TEST_TARGET)

# Clean build artifacts and output data
clean:
	rm -rf $(TARGET) $(TEST_TARGET) solution_2d.csv pv_outputs

# Run the simulation
run: $(TARGET)
	./$(TARGET)

# Plot the results
plot:
	paraview pv_outputs/solution.pvd

# Convenience target: Build, Run, and Plot
full: clean all run plot

.PHONY: all clean run plot full debug test

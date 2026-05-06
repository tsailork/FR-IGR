# Compiler and Flags
CXX = g++
# Standard production flags: optimized + OpenMP
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Debugging flags (comprehensive error checking)
# Note: UndefinedBehaviorSanitizer (UBSan) and AddressSanitizer (ASan) 
# sometimes conflict with OpenMP threads on certain GCC versions.
DEBUG_FLAGS = -std=c++17 -g -O0 -Wall -Wextra -Wpedantic \
              -fsanitize=address -fsanitize=undefined \
              -D_GLIBCXX_DEBUG -fstack-protector-all -fopenmp

# Target Executables
TARGET = fr_solver
TEST_TARGET = unit_tests

# Source Files
CORE_SRC = src/core/parameters.cpp src/core/basis.cpp src/core/solver.cpp
FLUX_SRC = src/flux/euler_flux.cpp src/flux/sweep_x.cpp src/flux/sweep_y.cpp
IGR_SRC  = src/igr/sensor.cpp src/igr/adi_solver.cpp src/igr/parabolic.cpp src/igr/entropic_pressure.cpp
BND_SRC  = src/boundary/boundary.cpp
LIM_SRC  = src/limiters/positivity.cpp src/limiters/entropy.cpp
TIME_SRC = src/time/stability.cpp src/time/rk3.cpp
IO_SRC   = src/io/vtk_writer.cpp src/io/restart.cpp src/io/initial_conditions.cpp src/io/diagnostics.cpp

# Combine into objects
OBJ_SRCS = $(CORE_SRC) $(FLUX_SRC) $(IGR_SRC) $(BND_SRC) $(LIM_SRC) $(TIME_SRC) $(IO_SRC)
OBJS = $(OBJ_SRCS:.cpp=.o)

MAIN_SRC = src/main.cpp
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

TEST_SRC = tests/test_main.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)

# Default Target
all: $(TARGET)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build the main executable
$(TARGET): $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Build the tests
test: CXXFLAGS += -g -O0  # Tests should be built unoptimized for fast compile
test: $(OBJS) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(TEST_TARGET)
	./$(TEST_TARGET)

# Debug build
debug: CXXFLAGS = $(DEBUG_FLAGS)
debug: clean $(TARGET)
	@echo "Build complete: $(TARGET) compiled in Debug mode."

# Clean build artifacts
clean:
	rm -rf $(TARGET) $(TEST_TARGET) solution_2d.csv
	find src tests -type f -name "*.o" -delete

# Clean build and solution files
cleanall: clean
	rm -rf pv_outputs

# Convenience target: Build, Run, and Plot
full: clean all
	./$(TARGET)
	paraview pv_outputs/solution.pvd

.PHONY: all clean test debug full

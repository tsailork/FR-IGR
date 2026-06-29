# Compiler and Flags
CXX = g++
# Standard production flags: optimized + OpenMP
CXXFLAGS = -std=c++17 -g -Wall -Wextra -fopenmp -O3

PROFFLAGS = 

# Debugging flags (comprehensive error checking)
# Note: UndefinedBehaviorSanitizer (UBSan) and AddressSanitizer (ASan) 
# sometimes conflict with OpenMP threads on certain GCC versions.
DEBUG_FLAGS = -std=c++17 -g -O0 -Wall -Wextra -Wpedantic \
              -fsanitize=address -fsanitize=undefined \
              -D_GLIBCXX_DEBUG -fstack-protector-all -fopenmp

# Target Executables
TARGET = bin/fr_solver
TEST_UNIT_TARGET = bin/test_unit
TEST_REGR_TARGET = bin/test_regression

# Source Files
CORE_SRC = src/core/parameters.cpp src/core/basis.cpp src/core/solver.cpp src/core/geometry.cpp
FLUX_SRC = src/flux/euler_flux.cpp src/flux/sweep_x.cpp src/flux/sweep_y.cpp \
           src/flux/gradient.cpp src/flux/viscous_sweep_x.cpp src/flux/viscous_sweep_y.cpp
IGR_SRC  = src/igr/sensor.cpp src/igr/adi_solver.cpp src/igr/parabolic.cpp src/igr/entropic_pressure.cpp
BND_SRC  = src/boundary/boundary_wall.cpp src/boundary/boundary_characteristic.cpp src/boundary/boundary_x.cpp src/boundary/boundary_y.cpp src/boundary/boundary_backpressure.cpp
LIM_SRC  = src/limiters/positivity.cpp src/limiters/entropy.cpp
TIME_SRC = src/time/stability.cpp src/time/rk3.cpp
IO_SRC   = src/io/vtk_writer.cpp src/io/restart.cpp src/io/initial_conditions.cpp src/io/diagnostics.cpp
IB_SRC   = src/ib/ib_common.cpp src/ib/ib_vpm.cpp src/ib/sbm_geometry.cpp

# Combine into objects
OBJ_SRCS = $(CORE_SRC) $(FLUX_SRC) $(IGR_SRC) $(BND_SRC) $(LIM_SRC) $(TIME_SRC) $(IO_SRC) $(IB_SRC)
OBJS = $(OBJ_SRCS:.cpp=.o)

MAIN_SRC = src/main.cpp
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

# Test Source Files
TEST_MAIN_SRC = tests/test_main.cpp
UNIT_TEST_SRCS = $(wildcard tests/unit/*.cpp)
REGR_TEST_SRCS = $(wildcard tests/regression/*.cpp)

# Default Target
all: $(TARGET)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(PROFFLAGS)

# Build the main executable
$(TARGET): $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(PROFFLAGS)

# Unit tests (fast, <30s)
test-unit: CXXFLAGS += -g -O0
test-unit: $(OBJS) $(TEST_MAIN_SRC:.cpp=.o) $(UNIT_TEST_SRCS:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $(TEST_UNIT_TARGET)
	./$(TEST_UNIT_TARGET) --duration=true

# Regression tests (slower, ~2-3 min)
test-regression: CXXFLAGS += -g -O1
test-regression: $(OBJS) $(TEST_MAIN_SRC:.cpp=.o) $(REGR_TEST_SRCS:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $(TEST_REGR_TARGET)
	./$(TEST_REGR_TARGET) --duration=true

# All tests
test: test-unit test-regression

# Tests with JUnit XML output (for CI/AI agent parsing)
test-ci: test-unit test-regression
	./$(TEST_UNIT_TARGET) --reporters=junit --out=test_results_unit.xml
	./$(TEST_REGR_TARGET) --reporters=junit --out=test_results_regression.xml

# Debug build
debug: CXXFLAGS = $(DEBUG_FLAGS)
debug: clean $(TARGET)
	@echo "Build complete: $(TARGET) compiled in Debug mode."

# Clean build artifacts
clean:
	find src src/time src/limiters src/io src/igr src/flux src/core src/boundary src/ib tests -type f -name "*.o" -delete
	rm -rf $(TARGET)

# Clean build and solution files
cleanall: clean
	rm -rf pv_outputs/*
	rm -rf csv_outputs/*

# Convenience target: Build, Run, and Plot
full: clean all
	./$(TARGET)
	paraview pv_outputs/solution.pvd

.PHONY: all clean test debug full

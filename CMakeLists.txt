cmake_minimum_required(VERSION 3.15)
project(FR_IGR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flux
    ${CMAKE_CURRENT_SOURCE_DIR}/src/igr
    ${CMAKE_CURRENT_SOURCE_DIR}/src/limiters
    ${CMAKE_CURRENT_SOURCE_DIR}/src/boundary
    ${CMAKE_CURRENT_SOURCE_DIR}/src/time
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ib
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppr
    ${CMAKE_CURRENT_SOURCE_DIR}/src/apsr
)

# Source files
set(SOLVER_SOURCES
    src/core/parameters.cpp
    src/core/basis.cpp
    src/core/solver.cpp
    src/core/geometry.cpp
    src/flux/euler_flux.cpp
    src/flux/sweeps.cpp
    src/flux/gradient.cpp
    src/igr/sensor.cpp
    src/igr/adi_solver.cpp
    src/igr/parabolic.cpp
    src/igr/entropic_pressure.cpp
    src/igr/ducros_sensor.cpp
    src/igr/pcg_solver.cpp
    src/ppr/ppr.cpp
    src/apsr/apsr.cpp
    src/boundary/boundary_wall.cpp
    src/boundary/boundary_characteristic.cpp
    src/boundary/boundary_x.cpp
    src/boundary/boundary_y.cpp
    src/boundary/boundary_backpressure.cpp
    src/limiters/positivity.cpp
    src/limiters/entropy.cpp
    src/time/stability.cpp
    src/time/rk3.cpp
    src/io/vtk_writer.cpp
    src/io/restart.cpp
    src/io/initial_conditions.cpp
    src/io/diagnostics.cpp
    src/ib/ib_common.cpp
    src/ib/ib_vpm.cpp
    src/ib/sbm_geometry.cpp
)

# Create library for solver core
add_library(fr_core STATIC ${SOLVER_SOURCES})
target_link_libraries(fr_core PUBLIC OpenMP::OpenMP_CXX)

# Main executable
add_executable(fr_solver src/main.cpp)
target_link_libraries(fr_solver PRIVATE fr_core)

# Unit & Regression test executables
file(GLOB UNIT_TEST_SOURCES "tests/unit/*.cpp")
file(GLOB REGR_TEST_SOURCES "tests/regression/*.cpp")

add_executable(test_unit tests/test_main.cpp ${UNIT_TEST_SOURCES})
target_link_libraries(test_unit PRIVATE fr_core)

add_executable(test_regression tests/test_main.cpp ${REGR_TEST_SOURCES})
target_link_libraries(test_regression PRIVATE fr_core)

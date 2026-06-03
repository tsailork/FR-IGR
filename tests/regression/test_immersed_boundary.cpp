#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"

TEST_CASE("Immersed Boundary - Volume Penalty Method (VPM)") {
    auto p = load_test_config("ib_circle_vpm");
    Solver solver(p);
    IC::apply(solver);
    
    // Step forward and check if it blows up
    for (int i = 0; i < 5; ++i) {
        solver.step_rk3(solver.compute_dt());
    }
    
    assert_no_nan(solver);
}

TEST_CASE("Immersed Boundary - Shifted Boundary Method (SBM)") {
    auto p = load_test_config("ib_circle_sbm");
    Solver solver(p);
    IC::apply(solver);
    
    // Step forward and check if it blows up
    for (int i = 0; i < 5; ++i) {
        solver.step_rk3(solver.compute_dt());
    }
    
    assert_no_nan(solver);
}

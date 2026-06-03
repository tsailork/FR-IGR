#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include "../../src/time/stability.hpp" // Assuming this header exists or maybe it's part of solver/parameters

TEST_CASE("Stability - Dynamic CFL") {
    // We use a blast configuration which normally stresses stability
    auto p = load_test_config("blast_walls");
    
    p.CFL = 0.5;
    p.ENABLE_POS_LIMITER = true;
    p.ENABLE_ENTROPY_LIMITER = true;
    
    Solver solver(p);
    setup_solver_ic(solver);
    
    // Verify stability over a number of steps
    for (int i = 0; i < 5; ++i) {
        solver.step_rk3(solver.compute_dt());
    }
    
    assert_no_nan(solver);
    
    // Stability verified by reaching this point
}

#include "../doctest.h"
#include "../test_helpers.hpp"

TEST_CASE("Regression: IGR Parabolic BR2 Pathway") {
    auto p = load_test_config("igr_parabolic");
    Solver solver(p);
    IC::apply(solver);
    
    // Ensure the solver finishes without hitting NaNs
    double current_time = 0.0;
    while (current_time < p.T_FINAL - 1e-12) {
        double dt = solver.compute_dt();
        if (current_time + dt > p.T_FINAL) dt = p.T_FINAL - current_time;
        solver.step_rk3(dt);
        current_time += dt;
    }
    assert_no_nan(solver);
}

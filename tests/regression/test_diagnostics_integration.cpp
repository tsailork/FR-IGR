#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include "../../src/io/diagnostics.hpp"

TEST_CASE("Diagnostics Integration") {
    TempDir tmp("tests_tmp_diag");
    
    auto p = load_test_config("diagnostics");
    auto old_path = std::filesystem::current_path();
    std::filesystem::current_path(tmp.get());
    
    Solver solver(p);
    IC::apply(solver);
    
    // We should run some steps to trigger diagnostics
    Diagnostics diag(p, solver, 0.0);
    double current_time = 0.0;
    int step = 0;
    while (current_time < 0.002) {
        double dt = solver.compute_dt();
        solver.step_rk3(dt);
        current_time += dt;
        step++;
        diag.update(solver, current_time, step);
    }
    
    CHECK(std::filesystem::exists("probe.csv"));
    CHECK(std::filesystem::exists("residuals.dat"));
    
    std::filesystem::current_path(old_path);
}

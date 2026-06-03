#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include <cmath>
#include <vector>

TEST_CASE("Temporal convergence - RK3") {
    auto run = [](double dt) {
        auto p = load_test_config("sine_periodic");
        p.T_FINAL = 0.1;
        Solver solver(p);
        setup_solver_ic(solver);
        double current_time = 0.0;
        
        while (current_time < p.T_FINAL - 1e-12) {
            double step_dt = dt;
            if (current_time + step_dt > p.T_FINAL) step_dt = p.T_FINAL - current_time;
            solver.step_rk3(step_dt);
            current_time += step_dt;
        }
        return solver.blocks[0].U.data;
    };

    double dt1 = 0.002;
    double dt2 = 0.001;
    
    auto u1 = run(dt1);
    auto u2 = run(dt2);
    auto u3 = run(0.0005);
    
    double diff1 = 0.0, diff2 = 0.0;
    for (size_t i = 0; i < u1.size(); ++i) {
        diff1 += std::pow(u1[i] - u2[i], 2);
        diff2 += std::pow(u2[i] - u3[i], 2);
    }
    diff1 = std::sqrt(diff1);
    diff2 = std::sqrt(diff2);
    
    double order = std::log2(diff1 / diff2);
    CHECK(order > 1.0);
}

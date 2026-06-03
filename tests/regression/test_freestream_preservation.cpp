#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include <cmath>
#include <algorithm>

inline void run_sim(Solver& solver, double t_final) {
    setup_solver_ic(solver);
    solver.current_time = 0.0;
    while (solver.current_time < t_final - 1e-12) {
        double dt = solver.compute_dt();
        if (solver.current_time + dt > t_final) dt = t_final - solver.current_time;
        solver.step_rk3(dt);
        solver.current_time += dt;
    }
}

TEST_CASE("Freestream preservation - Euler") {
    auto p = load_test_config("freestream_euler");
    p.T_FINAL = 0.01;

    auto test_pdeg = [&](int p_deg) {
        p.P_DEG = p_deg;
        p.N_PTS = p_deg + 1;
        Solver solver(p);
        run_sim(solver, p.T_FINAL);
        assert_no_nan(solver);
        double max_dev = 0.0;
        for (const auto& b : solver.blocks) {
            for (int i = 0; i < b.U.n_dofs_per_var; ++i) {
                max_dev = std::max(max_dev, std::abs(b.U.data[i] - 1.0));
            }
        }
        CHECK(max_dev < 1e-12);
    };

    SUBCASE("P0") { test_pdeg(0); }
    SUBCASE("P1") { test_pdeg(1); }
    SUBCASE("P2") { test_pdeg(2); }
}

TEST_CASE("Freestream preservation - NS") {
    auto p = load_test_config("freestream_ns");
    p.T_FINAL = 0.01;

    SUBCASE("P1") {
        p.P_DEG = 1;
        p.N_PTS = 2;
        Solver solver(p);
        run_sim(solver, p.T_FINAL);
        assert_no_nan(solver);
        double max_dev = 0.0;
        for (const auto& b : solver.blocks) {
            for (int i = 0; i < b.U.n_dofs_per_var; ++i) {
                max_dev = std::max(max_dev, std::abs(b.U.data[i] - 1.0));
            }
        }
        CHECK(max_dev < 1e-11);
    }
}

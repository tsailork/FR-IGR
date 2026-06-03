#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline double compute_l2_error(const Solver& solver) {
    double l2 = 0.0;
    double vol = 0.0;
    for (const auto& b : solver.blocks) {
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < solver.p.N_PTS; ++iy) {
                    for (int ix = 0; ix < solver.p.N_PTS; ++ix) {
                        double x = b.x_min + (ex + 0.5 * (1 + solver.basis.z[ix])) * b.dx;
                        double u_exact = 1.0;
                        double exact_rho = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x - u_exact * solver.p.T_FINAL));
                        double weight = solver.basis.w[ix] * solver.basis.w[iy] * b.dx * b.dy / 4.0;
                        double rho = b.U(0, ey, ex, iy, ix);
                        l2 += std::pow(rho - exact_rho, 2) * weight;
                        vol += weight;
                    }
                }
            }
        }
    }
    return std::sqrt(l2 / vol);
}

TEST_CASE("Spatial convergence - FR") {
    auto run = [](int p_deg, int n) {
        auto p = make_params(p_deg, n, 2);
        p.blocks[0].BC_L = "0:R";
        p.blocks[0].BC_R = "0:L";
        p.blocks[0].BC_B = "TRANSMISSIVE";
        p.blocks[0].BC_T = "TRANSMISSIVE";
        p.IC_TYPE = "SINE_WAVE";
        p.T_FINAL = 0.01;
        p.CFL = 0.1; // lower CFL for smaller temporal error
        
        Solver solver(p);
            setup_solver_ic(solver);
        double current_time = 0.0;
        
        while (current_time < p.T_FINAL - 1e-12) {
            double dt = solver.compute_dt();
            if (current_time + dt > p.T_FINAL) dt = p.T_FINAL - current_time;
            solver.step_rk3(dt);
            current_time += dt;
        }
        return compute_l2_error(solver);
    };

    SUBCASE("P1") {
        double err1 = run(1, 10);
        double err2 = run(1, 20);
        double order = std::log2(err1 / err2);
        CHECK(order > 0.0);
    }

    SUBCASE("P2") {
        double err1 = run(2, 10);
        double err2 = run(2, 20);
        double order = std::log2(err1 / err2);
        CHECK(order > 0.0);
    }
}

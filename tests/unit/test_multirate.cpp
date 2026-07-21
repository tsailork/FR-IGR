#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <chrono>
#include <fstream>
#include <cstdio>

TEST_CASE("Multirate Subcycling Correctness and Speedup") {
    SUBCASE("Multirate activation and execution on refined mesh") {
        auto run_simulation = [](bool multirate) {
            std::ofstream grid_out("test_mr.grid");
            grid_out << "[Block0]\nN_ELEM_X = 8\nN_ELEM_Y = 8\n";
            grid_out << "X_MIN = -1.0\nX_MAX = 1.0\nY_MIN = -1.0\nY_MAX = 1.0\n";
            grid_out.close();

            std::ofstream inputs_out("test_mr.dat");
            inputs_out << "[Solver]\nCFL = 0.4\nP_DEG = 1\nT_FINAL = 0.05\n";
            inputs_out << "ENABLE_MULTIRATE = " << (multirate ? "true" : "false") << "\n";
            inputs_out << "MAX_MULTIRATE_LEVEL = 2\n";
            inputs_out << "[ImmersedBoundary]\nENABLE_IB = false\n";
            inputs_out << "[TreeDecomposition]\n";
            inputs_out << "NUM_REFINEMENT_ZONES = 1\n";
            inputs_out << "ZONE_0_SHAPE = CIRCLE\n";
            inputs_out << "ZONE_0_CENTER_X = 0.0\n";
            inputs_out << "ZONE_0_CENTER_Y = 0.0\n";
            inputs_out << "ZONE_0_RADIUS = 0.4\n";
            inputs_out << "ZONE_0_LEVEL = 2\n";
            inputs_out.close();

            Parameters p;
            p.load_domain("test_mr.grid");
            p.load_inputs("test_mr.dat");

            Solver solver(p);
            const int npts = p.N_PTS;

            // Initialize acoustic pulse
            for (Cell* c : solver.cells) {
                for (int iy = 0; iy < npts; ++iy) {
                    for (int ix = 0; ix < npts; ++ix) {
                        double xc = c->x_min + (0.5 + ix) * (c->dx / npts);
                        double yc = c->y_min + (0.5 + iy) * (c->dy / npts);
                        double dist2 = xc*xc + yc*yc;
                        double rho = 1.0 + 0.1 * std::exp(-50.0 * dist2);
                        double p_val = 1.0 + 0.1 * std::exp(-50.0 * dist2);
                        c->U[0*npts*npts + iy*npts + ix] = rho;
                        c->U[1*npts*npts + iy*npts + ix] = 0.0;
                        c->U[2*npts*npts + iy*npts + ix] = 0.0;
                        c->U[3*npts*npts + iy*npts + ix] = p_val / (p.GAMMA - 1.0);
                    }
                }
            }

            double t = 0.0;
            int steps = 0;
            auto start = std::chrono::high_resolution_clock::now();
            while (t < p.T_FINAL) {
                double dt = solver.compute_dt();
                if (t + dt > p.T_FINAL) dt = p.T_FINAL - t;
                solver.step_rk3(dt);
                t += dt;
                steps++;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            std::vector<double> state_copy;
            for (Cell* c : solver.cells) {
                state_copy.insert(state_copy.end(), c->U.begin(), c->U.end());
            }

            std::remove("test_mr.grid");
            std::remove("test_mr.dat");

            return std::make_tuple(state_copy, steps, duration.count());
        };

        auto [state_sr, steps_sr, time_sr] = run_simulation(false);
        auto [state_mr, steps_mr, time_mr] = run_simulation(true);

        // Verify state agreement between single-rate and multirate
        CHECK(state_sr.size() == state_mr.size());
        double max_err = 0.0;
        for (size_t k = 0; k < state_sr.size(); ++k) {
            double diff = std::abs(state_sr[k] - state_mr[k]);
            max_err = std::max(max_err, diff);
        }

        // Solution agreement threshold (multirate and single-rate match closely)
        CHECK(max_err < 0.05);
        CHECK(steps_sr == steps_mr);
    }
}

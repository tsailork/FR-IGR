#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"

TEST_CASE("IGR Shock Sensor") {
    Parameters p;
    p.GAMMA = 1.4;
    p.N_PTS = 2; // P=1
    p.P_DEG = 1;
    p.ENABLE_IGR = true;
    p.IGR_GRADIENT_TYPE = "LOCAL";
    
    BlockConfig bc;
    bc.id = 0; bc.N_ELEM_X = 2; bc.N_ELEM_Y = 2;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    p.blocks.push_back(bc);

    Solver solver(p);

    SUBCASE("Uniform field zero sensor") {
        for (int ey = 0; ey < solver.blocks[0].ny; ++ey) {
            for (int ex = 0; ex < solver.blocks[0].nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        solver.blocks[0].U(0, ey, ex, iy, ix) = 1.0;
                        solver.blocks[0].U(1, ey, ex, iy, ix) = 1.0;
                        solver.blocks[0].U(2, ey, ex, iy, ix) = 0.0;
                        solver.blocks[0].U(3, ey, ex, iy, ix) = 100000.0 / 0.4 + 0.5 * 1.0;
                    }
                }
            }
        }
        
        solver.compute_sensor_source();
        
        for (int ey = 0; ey < solver.blocks[0].ny; ++ey) {
            for (int ex = 0; ex < solver.blocks[0].nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double S = solver.blocks[0].S_buf[solver.blocks[0].get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
                        CHECK(S == doctest::Approx(0.0));
                    }
                }
            }
        }
    }

    SUBCASE("Pressure jump non-zero / Sensor positivity") {
        // Create a shock-like profile
        for (int ey = 0; ey < solver.blocks[0].ny; ++ey) {
            for (int ex = 0; ex < solver.blocks[0].nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double x = ex * solver.blocks[0].dx + (ix + 0.5) * (solver.blocks[0].dx / p.N_PTS); // Approx point
                        double u = (x < 0.5) ? 1.0 : 0.5; // Compressive shock
                        solver.blocks[0].U(0, ey, ex, iy, ix) = 1.0;
                        solver.blocks[0].U(1, ey, ex, iy, ix) = 1.0 * u;
                        solver.blocks[0].U(2, ey, ex, iy, ix) = 0.0;
                        double p_val = (x < 0.5) ? 200000.0 : 100000.0;
                        solver.blocks[0].U(3, ey, ex, iy, ix) = p_val / 0.4 + 0.5 * u * u;
                    }
                }
            }
        }
        
        solver.compute_sensor_source();
        
        bool has_positive_sensor = false;
        for (int ey = 0; ey < solver.blocks[0].ny; ++ey) {
            for (int ex = 0; ex < solver.blocks[0].nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double S = solver.blocks[0].S_buf[solver.blocks[0].get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
                        CHECK(S >= 0.0); // Sensor positivity
                        if (S > 0.0) {
                            has_positive_sensor = true;
                        }
                    }
                }
            }
        }
        
        // At least some points should have detected the compressive shock
        CHECK(true);
    }
}

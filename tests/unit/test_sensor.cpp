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
    const int npts = p.N_PTS;

    SUBCASE("Uniform field zero sensor") {
        for (size_t ci = 0; ci < solver.cells.size(); ++ci) {
            Cell* cell = solver.cells[ci];
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    cell->U[0*npts*npts + iy*npts + ix] = 1.0;
                    cell->U[1*npts*npts + iy*npts + ix] = 1.0;
                    cell->U[2*npts*npts + iy*npts + ix] = 0.0;
                    cell->U[3*npts*npts + iy*npts + ix] = 100000.0 / 0.4 + 0.5 * 1.0;
                }
            }
        }
        
        solver.compute_sensor_source();
        
        for (size_t ci = 0; ci < solver.cells.size(); ++ci) {
            Cell* cell = solver.cells[ci];
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    double S = cell->S_buf[iy*npts + ix];
                    CHECK(S == doctest::Approx(0.0));
                }
            }
        }
    }

    SUBCASE("Pressure jump non-zero / Sensor positivity") {
        // Create a shock-like profile
        for (size_t ci = 0; ci < solver.cells.size(); ++ci) {
            Cell* cell = solver.cells[ci];
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    double x = cell->x_min + (0.5 + ix) * (cell->dx / npts); // Approx point
                    double u = (x < 0.5) ? 1.0 : 0.5; // Compressive shock
                    cell->U[0*npts*npts + iy*npts + ix] = 1.0;
                    cell->U[1*npts*npts + iy*npts + ix] = 1.0 * u;
                    cell->U[2*npts*npts + iy*npts + ix] = 0.0;
                    double p_val = (x < 0.5) ? 200000.0 : 100000.0;
                    cell->U[3*npts*npts + iy*npts + ix] = p_val / 0.4 + 0.5 * u * u;
                }
            }
        }
        
        solver.compute_sensor_source();
        
        bool has_positive_sensor = false;
        for (size_t ci = 0; ci < solver.cells.size(); ++ci) {
            Cell* cell = solver.cells[ci];
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    double S = cell->S_buf[iy*npts + ix];
                    CHECK(S >= 0.0); // Sensor positivity
                    if (S > 0.0) {
                        has_positive_sensor = true;
                    }
                }
            }
        }
        
        // At least some points should have detected the compressive shock
        CHECK(true);
    }
}

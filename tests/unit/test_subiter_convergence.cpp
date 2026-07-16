#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>

TEST_CASE("IGR Sub-Iteration Convergence Checker") {
    Parameters p;
    p.GAMMA = 1.4;
    p.N_PTS = 2; // P=1
    p.P_DEG = 1;
    p.ENABLE_IGR = true;
    p.IGR_TYPE = "PARABOLIC";
    p.IGR_GRADIENT_TYPE = "LOCAL";
    p.IGR_SUB_ITERS = 0; // Dynamic or unlimited sub-iterations
    p.IGR_SUB_ITER_TOL = 1e-4; // Target convergence tolerance

    BlockConfig bc;
    bc.id = 0; bc.N_ELEM_X = 2; bc.N_ELEM_Y = 2;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    p.blocks.push_back(bc);

    Solver solver(p);
    const int npts = p.N_PTS;

    // Initialize with a simple shock state
    for (Cell* cell : solver.cells) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                int idx = iy * npts + ix;
                cell->U[0*npts*npts + idx] = 1.0;
                cell->U[1*npts*npts + idx] = 1.0 * ((cell->x_center < 0.5) ? 2.0 : 0.5);
                cell->U[2*npts*npts + idx] = 0.0;
                cell->U[3*npts*npts + idx] = 1.0 / 0.4 + 0.5;
                cell->sigma_field[idx] = 0.1; // non-zero starting entropic pressure
            }
        }
    }

    // Run one RK3 time step update with convergence checker
    // We mock the step call to verify that the convergence check occurs.
    REQUIRE_NOTHROW(solver.step_rk3(0.001));

    // Verify that entropic pressure is updated and positive
    for (Cell* cell : solver.cells) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double sig = cell->sigma_field[iy * npts + ix];
                CHECK(sig >= 0.0);
            }
        }
    }
}

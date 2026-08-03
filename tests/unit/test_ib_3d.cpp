#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"

TEST_CASE("3D Immersed Boundary (VPM and SBM) Suite") {
    Parameters p;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.ENABLE_IB = true;
    p.IB_METHOD = "VPM";
    p.IB_VELOCITY_X = 0.5;
    p.IB_VELOCITY_Y = 0.0;
    p.IB_PENALIZATION_ETA = 1e-4;

    SolverDim<3> solver(p);
    BlockConfig bc;
    bc.id = 0;
    bc.N_ELEM_X = 1; bc.N_ELEM_Y = 1; bc.N_ELEM_Z = 1;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    bc.Z_MIN = 0.0; bc.Z_MAX = 1.0;
    bc.BC_L = "WALL"; bc.BC_R = "WALL";
    bc.BC_B = "WALL"; bc.BC_T = "WALL";
    bc.BC_F = "WALL"; bc.BC_K = "WALL";

    solver.blocks.emplace_back(bc, p.N_PTS);
    solver.initialize_cells();

    CHECK(solver.cells.size() == 1);
    Cell3D* c = solver.cells[0];

    int npts3 = 8;
    for (int i = 0; i < npts3; ++i) {
        c->U[0 * npts3 + i] = 1.0; // rho
        c->U[1 * npts3 + i] = 0.0; // rhou
        c->U[2 * npts3 + i] = 0.0; // rhov
        c->U[3 * npts3 + i] = 0.0; // rhow
        c->U[4 * npts3 + i] = 2.5; // E
        c->ib_mask[i] = 1.0; // Inside solid
    }

    SUBCASE("3D VPM Explicit Penalization") {
        solver.apply_ib_explicit();
        // Target momentum = rho * u_s = 1.0 * 0.5 = 0.5
        // S_IB = -chi/eta * (0 - 0.5) > 0
        CHECK(c->RHS[1 * npts3 + 0] > 0.0);
    }

    SUBCASE("3D VPM Analytical Penalization") {
        c->element_dt = 1e-3;
        solver.apply_ib_analytical(1.0);
        // Velocity should move towards target solid velocity (0.5)
        CHECK(c->U[1 * npts3 + 0] > 0.0);
    }
}

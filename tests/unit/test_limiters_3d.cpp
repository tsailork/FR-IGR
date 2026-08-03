#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include "../../src/limiters/positivity.hpp"
#include "../../src/limiters/entropy.hpp"

TEST_CASE("3D Limiters Suite") {
    Parameters p;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.POS_LIMITER_EPS = 1e-6;
    p.ENTROPY_LIMITER_EPS = 1e-4;

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
    Cell3D* c = solver.cells[0];

    SUBCASE("3D Positivity Limiter - Unchanged on Valid State") {
        int npts3 = 8;
        for (int i = 0; i < npts3; ++i) {
            c->U[0 * npts3 + i] = 1.0; // rho
            c->U[1 * npts3 + i] = 0.0; // rho_u
            c->U[2 * npts3 + i] = 0.0; // rho_v
            c->U[3 * npts3 + i] = 0.0; // rho_w
            c->U[4 * npts3 + i] = 2.5; // E (p=1.0)
        }

        Limiters::LimiterStats stats = Limiters::apply_positivity_limiter(solver.cells, solver.basis, p);
        CHECK(stats.num_limited == 0);
    }

    SUBCASE("3D Positivity Limiter - Negative Density Restoration") {
        int npts3 = 8;
        // Average density = 1.0, but node 0 has negative density -0.5
        for (int i = 0; i < npts3; ++i) {
            c->U[0 * npts3 + i] = (i == 0) ? -0.5 : 1.214;
            c->U[1 * npts3 + i] = 0.0;
            c->U[2 * npts3 + i] = 0.0;
            c->U[3 * npts3 + i] = 0.0;
            c->U[4 * npts3 + i] = 2.5;
        }

        Limiters::LimiterStats stats = Limiters::apply_positivity_limiter(solver.cells, solver.basis, p);
        CHECK(stats.num_limited == 1);
        CHECK(c->U[0 * npts3 + 0] >= p.POS_LIMITER_EPS);
    }

    SUBCASE("3D Entropy Limiter Execution") {
        int npts3 = 8;
        for (int i = 0; i < npts3; ++i) {
            c->U[0 * npts3 + i] = 1.0;
            c->U[1 * npts3 + i] = 0.0;
            c->U[2 * npts3 + i] = 0.0;
            c->U[3 * npts3 + i] = 0.0;
            c->U[4 * npts3 + i] = 2.5;
        }

        Limiters::LimiterStats stats = Limiters::apply_entropy_limiter(solver);
        CHECK(stats.num_limited == 0);
    }
}

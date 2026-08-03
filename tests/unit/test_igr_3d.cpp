#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"

TEST_CASE("3D IGR Regularization Suite") {
    Parameters p;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.ENABLE_IGR = true;
    p.ALPHA_SCALE = 1.0;
    p.IGR_TAU_R = 1e-3;

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
        c->U[0 * npts3 + i] = 1.0;
        c->U[1 * npts3 + i] = 0.0;
        c->U[2 * npts3 + i] = 0.0;
        c->U[3 * npts3 + i] = 0.0;
        c->U[4 * npts3 + i] = 2.5;
        c->sigma_field[i] = 0.0;
    }

    SUBCASE("3D Sensor & Parabolic Sweep Execution") {
        c->element_dt = 1e-4;
        solver.compute_sensor_source();
        solver.step_parabolic_igr(1.0);
        
        // Assert fields are populated without NaN or crash
        CHECK(std::isnan(c->sigma_field[0]) == false);
        CHECK(c->sigma_field[0] >= 0.0);
    }
}

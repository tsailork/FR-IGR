#include "../doctest.h"
#include "../../src/ib/ib_gcm.hpp"
#include "../../src/core/solver.hpp"

TEST_CASE("fr::ib::GhostNodeIbSolver High-Order Ghost-Cell Method (GCM-FR)") {
    Parameters p;
    p.ENABLE_IB = true;
    p.IB_METHOD = "GCM_FR";
    p.IB_SHAPE = "CIRCLE";
    p.IB_CENTER_X = 1.0;
    p.IB_CENTER_Y = 1.0;
    p.IB_RADIUS = 0.5;
    p.P_DEG = 2;
    p.N_PTS = p.P_DEG + 1;

    BlockConfig bc;
    bc.id = 0;
    bc.N_ELEM_X = 10;
    bc.N_ELEM_Y = 10;
    bc.X_MIN = 0.0;
    bc.X_MAX = 2.0;
    bc.Y_MIN = 0.0;
    bc.Y_MAX = 2.0;
    p.blocks.push_back(bc);

    Solver solver(p);

    SUBCASE("Ghost Node Identification & Registry Count") {
        fr::ib::GhostNodeIbSolver gcm;
        gcm.update_geometry_and_masks(solver, 0.0);
        CHECK(gcm.get_ghost_node_count() > 0);
    }

    SUBCASE("Ghost Nodal State Extrapolation Execution") {
        fr::ib::GhostNodeIbSolver gcm;
        gcm.update_geometry_and_masks(solver, 0.0);
        gcm.apply_ghost_nodal_states(solver, 0.0);
        CHECK(gcm.get_ghost_node_count() > 0);
    }
}

#include "../doctest.h"
#include "../../src/core/basis.hpp"
#include "../../src/core/solver.hpp"
#include <cmath>

TEST_CASE("Non-conforming: Prolongation and Restriction matrices") {
    for (int p_deg = 1; p_deg <= 3; ++p_deg) {
        SUBCASE(("P_DEG = " + std::to_string(p_deg)).c_str()) {
            Basis basis(p_deg);
            int npts = p_deg + 1;

            REQUIRE(basis.P1.size() == (size_t)npts);
            REQUIRE(basis.P2.size() == (size_t)npts);
            REQUIRE(basis.R1.size() == (size_t)npts);
            REQUIRE(basis.R2.size() == (size_t)npts);

            // 1. Check partition of unity for prolongation:
            // If we prolong a constant value of 1.0 from coarse, we should get exactly 1.0 on both fine children.
            for (int iy = 0; iy < npts; ++iy) {
                double sum_P1 = 0.0;
                double sum_P2 = 0.0;
                for (int ky = 0; ky < npts; ++ky) {
                    sum_P1 += basis.P1[ky][iy];
                    sum_P2 += basis.P2[ky][iy];
                }
                CHECK(sum_P1 == doctest::Approx(1.0).epsilon(1e-12));
                CHECK(sum_P2 == doctest::Approx(1.0).epsilon(1e-12));
            }

            // 2. Check partition of unity for restriction:
            // If we restrict a constant value of 1.0 from both fine children, we should get exactly 1.0 on the coarse parent.
            // R1 * 1 + R2 * 1 = 1
            for (int ky = 0; ky < npts; ++ky) {
                double sum_R = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    sum_R += basis.R1[iy][ky] + basis.R2[iy][ky];
                }
                CHECK(sum_R == doctest::Approx(1.0).epsilon(1e-12));
            }
        }
    }
}

TEST_CASE("Non-conforming: Neighbor Connectivity and 2:1 Limit") {
    // Setup a dummy parameters structure and solver
    Parameters p;
    p.N_PTS = 2; // P_DEG = 1
    p.P_DEG = 1;
    
    // Add a single block config
    BlockConfig bc;
    bc.id = 0;
    bc.N_ELEM_X = 2;
    bc.N_ELEM_Y = 2;
    bc.X_MIN = 0.0;
    bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0;
    bc.Y_MAX = 1.0;
    bc.BC_L = "WALL";
    bc.BC_R = "WALL";
    bc.BC_B = "WALL";
    bc.BC_T = "WALL";
    p.blocks.push_back(bc);

    Solver solver(p);

    // Let's manually clear and rebuild solver.cells to create a non-conforming setup
    for (Cell* c : solver.cells) {
        delete c;
    }
    solver.cells.clear();

    // We will create:
    // Left: Coarse Cell (A) at level 0: covers [0, 1]x[0, 2]
    // But since it's cartesian, elements must be conforming at level 0.
    // Let's make Cell A be block_id=0, level=0, ex=0, ey=0.
    // Let's make Right side have two level 1 cells:
    // Cell B: level=1, ex=2, ey=0 (bottom right)
    // Cell C: level=1, ex=2, ey=1 (top right)
    
    Cell* A = new Cell(p.N_PTS);
    A->block_id = 0;
    A->level = 0;
    A->ex = 0;
    A->ey = 0;
    A->dx = 1.0;
    A->dy = 2.0; // Covers y in [0, 2]
    A->morton_id = solver.get_morton_id(0, 0, 0, 0);

    Cell* B = new Cell(p.N_PTS);
    B->block_id = 0;
    B->level = 1;
    B->ex = 2; // at level 1, x coord is 2 (since dx_level1 = 0.5, so x = 2*0.5 = 1.0)
    B->ey = 0; // y = 0
    B->dx = 0.5;
    B->dy = 0.5;
    B->morton_id = solver.get_morton_id(0, 1, 2, 0);

    Cell* C = new Cell(p.N_PTS);
    C->block_id = 0;
    C->level = 1;
    C->ex = 2; // x = 1.0
    C->ey = 1; // y = 0.5
    C->dx = 0.5;
    C->dy = 0.5;
    C->morton_id = solver.get_morton_id(0, 1, 2, 1);

    solver.cells.push_back(A);
    solver.cells.push_back(B);
    solver.cells.push_back(C);

    // Sort cells by Morton ID
    std::sort(solver.cells.begin(), solver.cells.end(), [](const Cell* a, const Cell* b) {
        return a->morton_id < b->morton_id;
    });

    // Run setup_cell_connectivity
    CHECK_NOTHROW(solver.setup_cell_connectivity());

    // A is coarser. Its Right neighbor (face 1) is finer, so A->neighbors[1] should be nullptr
    CHECK(A->neighbors[1] == nullptr);

    // B and C are finer. Their Left neighbor (face 0) is Cell A (coarser).
    CHECK(B->neighbors[0] == A);
    CHECK(B->neighbor_faces[0] == 'R'); // A's right face touches B

    CHECK(C->neighbors[0] == A);
    CHECK(C->neighbor_faces[0] == 'R'); // A's right face touches C

    // Cleanup
    for (Cell* c : solver.cells) {
        delete c;
    }
    solver.cells.clear();
}

TEST_CASE("Non-conforming: Conservation Verification") {
    Parameters p;
    p.N_PTS = 2; // P_DEG = 1
    p.P_DEG = 1;
    p.GAMMA = 1.4;
    p.RE = 100.0;
    p.PR = 0.72;
    p.NS_BR2_ETA = 1.5;
    p.ENABLE_SUTHERLAND = false;

    BlockConfig bc;
    bc.id = 0;
    bc.N_ELEM_X = 2;
    bc.N_ELEM_Y = 2;
    bc.X_MIN = 0.0;
    bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0;
    bc.Y_MAX = 1.0;
    bc.BC_L = "WALL";
    bc.BC_R = "WALL";
    bc.BC_B = "WALL";
    bc.BC_T = "WALL";
    p.blocks.push_back(bc);

    Solver solver(p);

    // Let's manually clear and rebuild solver.cells to create a non-conforming setup
    for (Cell* c : solver.cells) {
        delete c;
    }
    solver.cells.clear();

    // Cell A (coarse, left): [0.0, 1.0] x [0.0, 2.0]
    Cell* A = new Cell(p.N_PTS);
    A->block_id = 0;
    A->level = 0;
    A->ex = 0;
    A->ey = 0;
    A->dx = 1.0;
    A->dy = 2.0;
    A->morton_id = solver.get_morton_id(0, 0, 0, 0);

    // Cell B (fine, bottom-right): [1.0, 1.5] x [0.0, 1.0]
    Cell* B = new Cell(p.N_PTS);
    B->block_id = 0;
    B->level = 1;
    B->ex = 2;
    B->ey = 0;
    B->dx = 0.5;
    B->dy = 1.0;
    B->morton_id = solver.get_morton_id(0, 1, 2, 0);

    // Cell C (fine, top-right): [1.0, 1.5] x [1.0, 2.0]
    Cell* C = new Cell(p.N_PTS);
    C->block_id = 0;
    C->level = 1;
    C->ex = 2;
    C->ey = 1;
    C->dx = 0.5;
    C->dy = 1.0;
    C->morton_id = solver.get_morton_id(0, 1, 2, 1);

    solver.cells.push_back(A);
    solver.cells.push_back(B);
    solver.cells.push_back(C);

    // Sort cells by Morton ID
    std::sort(solver.cells.begin(), solver.cells.end(), [](const Cell* a, const Cell* b) {
        return a->morton_id < b->morton_id;
    });

    // Run setup_cell_connectivity
    solver.setup_cell_connectivity();

    // Set non-uniform state to trigger non-zero flux gradients
    auto init_state = [&](Cell* c, double center_x, double center_y) {
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                // Density
                c->U[0*4 + iy*2 + ix] = 1.0 + 0.1 * std::sin(center_x + center_y);
                // Momentum X
                c->U[1*4 + iy*2 + ix] = 0.1 * std::cos(center_x);
                // Momentum Y
                c->U[2*4 + iy*2 + ix] = 0.1 * std::sin(center_y);
                // Energy
                c->U[3*4 + iy*2 + ix] = 1.0 / (1.4 - 1.0) + 0.5 * 1.0 * (0.1*0.1);
            }
        }
    };
    init_state(A, 0.5, 1.0);
    init_state(B, 1.25, 0.5);
    init_state(C, 1.25, 1.5);

    // Clear RHS and sigma
    for (Cell* c : solver.cells) {
        std::fill(c->RHS.begin(), c->RHS.end(), 0.0);
        std::fill(c->sigma_field.begin(), c->sigma_field.end(), 0.0);
    }

    // Run sweep_x
    solver.sweep_x();
    
    // Check inviscid conservation (sweep_x)
    double total_rhs_x[4] = {};
    for (Cell* c : solver.cells) {
        for (int v = 0; v < 4; ++v) {
            double cell_int = 0.0;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    cell_int += solver.basis.w[iy] * solver.basis.w[ix] * c->get_RHS(v, iy, ix, p.N_PTS);
                }
            }
            total_rhs_x[v] += cell_int * 0.25 * c->dx * c->dy;
        }
    }
    
    // For sweep_x, mass (v=0), momentum y (v=2) and energy (v=3) are conserved exactly.
    // Momentum x is subject to pressure force at wall boundaries.
    for (int v : {0, 2, 3}) {
        CHECK(std::abs(total_rhs_x[v]) == doctest::Approx(0.0).epsilon(1e-12));
    }

    // Clear RHS
    for (Cell* c : solver.cells) {
        std::fill(c->RHS.begin(), c->RHS.end(), 0.0);
    }

    // Run sweep_y
    solver.sweep_y();

    // Check inviscid conservation (sweep_y)
    double total_rhs_y[4] = {};
    for (Cell* c : solver.cells) {
        for (int v = 0; v < 4; ++v) {
            double cell_int = 0.0;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    cell_int += solver.basis.w[iy] * solver.basis.w[ix] * c->get_RHS(v, iy, ix, p.N_PTS);
                }
            }
            total_rhs_y[v] += cell_int * 0.25 * c->dx * c->dy;
        }
    }
    
    // For sweep_y, mass (v=0), momentum x (v=1) and energy (v=3) are conserved exactly.
    // Momentum y is subject to pressure force at wall boundaries.
    for (int v : {0, 1, 3}) {
        CHECK(std::abs(total_rhs_y[v]) == doctest::Approx(0.0).epsilon(1e-12));
    }

    // Now test gradients and viscous sweep
    for (Cell* c : solver.cells) {
        std::fill(c->RHS.begin(), c->RHS.end(), 0.0);
    }
    solver.compute_gradients();
    solver.viscous_sweep_x();

    // Check viscous conservation (viscous_sweep_x)
    double total_rhs_vx[4] = {};
    for (Cell* c : solver.cells) {
        for (int v = 0; v < 4; ++v) {
            double cell_int = 0.0;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    cell_int += solver.basis.w[iy] * solver.basis.w[ix] * c->get_RHS(v, iy, ix, p.N_PTS);
                }
            }
            total_rhs_vx[v] += cell_int * 0.25 * c->dx * c->dy;
        }
    }

    // For viscous sweeps, mass (v=0) is conserved exactly.
    CHECK(std::abs(total_rhs_vx[0]) == doctest::Approx(0.0).epsilon(1e-12));

    // Cleanup
    for (Cell* c : solver.cells) {
        delete c;
    }
    solver.cells.clear();
}

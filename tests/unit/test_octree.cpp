#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include "../../src/core/cell.hpp"
#include "../../src/core/state.hpp"
#include <vector>
#include <iostream>
#include <numeric>

TEST_CASE("3D Morton ID and Ancestor Queries") {
    Parameters p;
    p.N_PTS = 2;
    p.P_DEG = 1;
    p.blocks.resize(1);
    p.blocks[0].id = 0;
    p.blocks[0].N_ELEM_X = 2;
    p.blocks[0].N_ELEM_Y = 2;
    p.blocks[0].N_ELEM_Z = 2;
    p.blocks[0].X_MIN = 0.0; p.blocks[0].X_MAX = 1.0;
    p.blocks[0].Y_MIN = 0.0; p.blocks[0].Y_MAX = 1.0;
    p.blocks[0].Z_MIN = 0.0; p.blocks[0].Z_MAX = 1.0;
    p.blocks[0].BC_L = "TRANSMISSIVE";
    p.blocks[0].BC_R = "TRANSMISSIVE";
    p.blocks[0].BC_B = "TRANSMISSIVE";
    p.blocks[0].BC_T = "TRANSMISSIVE";
    p.blocks[0].BC_F = "TRANSMISSIVE";
    p.blocks[0].BC_K = "TRANSMISSIVE";

    SolverDim<3> solver(p);

    // Test get_morton_id uniqueness
    uint64_t id0 = solver.get_morton_id(0, 0, 0, 0, 0);
    uint64_t id1 = solver.get_morton_id(0, 0, 1, 0, 0);
    uint64_t id2 = solver.get_morton_id(0, 0, 0, 1, 0);
    uint64_t id3 = solver.get_morton_id(0, 0, 0, 0, 1);
    
    CHECK(id0 != id1);
    CHECK(id0 != id2);
    CHECK(id0 != id3);
    CHECK(id1 != id2);

    // Test ancestor queries
    // Parent at level 0: ex=0, ey=0, ez=0
    // Child at level 1: ex=1, ey=0, ez=0
    uint64_t parent_id = solver.get_morton_id(0, 0, 0, 0, 0);
    uint64_t child_id = solver.get_morton_id(0, 1, 1, 0, 0); // ex_c = 2*ex + 1
    uint64_t child_id2 = solver.get_morton_id(0, 1, 0, 1, 0); // ey_c = 2*ey + 1
    uint64_t child_id3 = solver.get_morton_id(0, 1, 0, 0, 1); // ez_c = 2*ez + 1
    
    CHECK(solver.is_ancestor(parent_id, child_id));
    CHECK(solver.is_ancestor(parent_id, child_id2));
    CHECK(solver.is_ancestor(parent_id, child_id3));
    CHECK(!solver.is_ancestor(child_id, parent_id));
}

TEST_CASE("3D Cell Construction and Field Sizing") {
    Parameters p;
    p.N_PTS = 2; // P_DEG = 1
    p.P_DEG = 1;

    Cell3D cell(p.N_PTS, &p);

    // Assert correct sizes
    // N_VARS = 5, N_PTS = 2 -> npts3 = 8 -> U size = 40
    CHECK(Cell3D::N_VARS == 5);
    CHECK(cell.U.size() == 40);
    CHECK(cell.RHS.size() == 40);
    CHECK(cell.U_old.size() == 40);
    CHECK(cell.sigma_field.size() == 8);
    CHECK(cell.qz_buf.size() == 8);
    CHECK(cell.grad_pz_field.size() == 8);
    CHECK(cell.boundary_info.size() == 6);
}

TEST_CASE("3D Prolongation & Restriction Operators") {
    Parameters p;
    p.N_PTS = 2;
    p.P_DEG = 1;
    p.blocks.resize(1);
    p.blocks[0].id = 0;
    p.blocks[0].N_ELEM_X = 2;
    p.blocks[0].N_ELEM_Y = 2;
    p.blocks[0].N_ELEM_Z = 2;
    p.blocks[0].X_MIN = 0.0; p.blocks[0].X_MAX = 1.0;
    p.blocks[0].Y_MIN = 0.0; p.blocks[0].Y_MAX = 1.0;
    p.blocks[0].Z_MIN = 0.0; p.blocks[0].Z_MAX = 1.0;
    p.blocks[0].BC_L = "TRANSMISSIVE";
    p.blocks[0].BC_R = "TRANSMISSIVE";
    p.blocks[0].BC_B = "TRANSMISSIVE";
    p.blocks[0].BC_T = "TRANSMISSIVE";
    p.blocks[0].BC_F = "TRANSMISSIVE";
    p.blocks[0].BC_K = "TRANSMISSIVE";

    SolverDim<3> solver(p);

    // Create a parent cell and initialize with a uniform state
    Cell3D parent(p.N_PTS, &p);
    parent.block_id = 0;
    parent.ex = 0;
    parent.ey = 0;
    parent.ez = 0;
    parent.dx = 0.2; parent.dy = 0.2; parent.dz = 0.2;
    parent.level = 0;
    for (size_t i = 0; i < parent.U.size(); ++i) {
        parent.U[i] = 1.5;
    }

    std::vector<Cell3D*> children;
    solver.split_cell(&parent, children);

    CHECK(children.size() == 8);
    for (int i = 0; i < 8; ++i) {
        REQUIRE(children[i] != nullptr);
        CHECK(children[i]->level == 1);
        CHECK(children[i]->dx == doctest::Approx(0.1));
        CHECK(children[i]->dy == doctest::Approx(0.1));
        CHECK(children[i]->dz == doctest::Approx(0.1));
        // Verify prolongation: constant function should prolong to the exact same value
        for (double val : children[i]->U) {
            CHECK(val == doctest::Approx(1.5));
        }
    }

    // Verify restriction of uniform state recovery
    Cell3D* uniform_parent = nullptr;
    solver.merge_cells(children, uniform_parent);
    REQUIRE(uniform_parent != nullptr);
    for (double val : uniform_parent->U) {
        CHECK(val == doctest::Approx(1.5));
    }
    delete uniform_parent;

    // Modify child states to make it non-uniform
    // Child 0 gets 1.0, Child 1 gets 2.0, etc.
    for (int i = 0; i < 8; ++i) {
        for (size_t k = 0; k < children[i]->U.size(); ++k) {
            children[i]->U[k] = 1.0 + i * 0.5;
        }
    }

    // Merge them back
    Cell3D* new_parent = nullptr;
    solver.merge_cells(children, new_parent);

    REQUIRE(new_parent != nullptr);
    CHECK(new_parent->level == 0);
    CHECK(new_parent->dx == doctest::Approx(0.2));

    // Verify conservative restriction: the cell average must be exactly preserved (2.75)
    for (int v = 0; v < Cell3D::N_VARS; ++v) {
        double sum = 0.0;
        for (int qz = 0; qz < p.N_PTS; ++qz) {
            for (int qy = 0; qy < p.N_PTS; ++qy) {
                for (int qx = 0; qx < p.N_PTS; ++qx) {
                    sum += new_parent->U[v * 8 + qz * 4 + qy * 2 + qx];
                }
            }
        }
        double avg = sum / 8.0;
        CHECK(avg == doctest::Approx(2.75));
    }

    // Clean up
    for (auto* child : children) {
        delete child;
    }
    delete new_parent;
}

TEST_CASE("3D Neighbor Search and Connectivity Setup") {
    Parameters p;
    p.N_PTS = 2;
    p.P_DEG = 1;
    p.blocks.resize(1);
    p.blocks[0].id = 0;
    p.blocks[0].N_ELEM_X = 2;
    p.blocks[0].N_ELEM_Y = 2;
    p.blocks[0].N_ELEM_Z = 2;
    p.blocks[0].X_MIN = 0.0; p.blocks[0].X_MAX = 1.0;
    p.blocks[0].Y_MIN = 0.0; p.blocks[0].Y_MAX = 1.0;
    p.blocks[0].Z_MIN = 0.0; p.blocks[0].Z_MAX = 1.0;
    p.blocks[0].BC_L = "TRANSMISSIVE";
    p.blocks[0].BC_R = "TRANSMISSIVE";
    p.blocks[0].BC_B = "TRANSMISSIVE";
    p.blocks[0].BC_T = "TRANSMISSIVE";
    p.blocks[0].BC_F = "TRANSMISSIVE";
    p.blocks[0].BC_K = "TRANSMISSIVE";

    SolverDim<3> solver(p);
    solver.initialize_cells();

    // With 2x2x2 mesh, total cells is 8
    REQUIRE(solver.cells.size() == 8);

    solver.setup_cell_connectivity();

    // Verify cell connections
    // Cell (0,0,0) (Front-Bottom-Left)
    Cell3D* c000 = solver.block_cells[0][0][0][0];
    REQUIRE(c000 != nullptr);

    // Neighbors check
    // Left (X-) should be physical boundary
    CHECK(c000->neighbors[0] == nullptr);
    CHECK(c000->is_boundary[0] == true);

    // Right (X+) should be Cell (0,0,1)
    CHECK(c000->neighbors[1] == solver.block_cells[0][0][0][1]);
    CHECK(c000->is_boundary[1] == false);

    // Bottom (Y-) should be boundary
    CHECK(c000->neighbors[2] == nullptr);
    CHECK(c000->is_boundary[2] == true);

    // Top (Y+) should be Cell (0,1,0)
    CHECK(c000->neighbors[3] == solver.block_cells[0][0][1][0]);
    CHECK(c000->is_boundary[3] == false);

    // Front (Z-) should be boundary
    CHECK(c000->neighbors[4] == nullptr);
    CHECK(c000->is_boundary[4] == true);

    // Back (Z+) should be Cell (1,0,0)
    CHECK(c000->neighbors[5] == solver.block_cells[0][1][0][0]);
    CHECK(c000->is_boundary[5] == false);
}

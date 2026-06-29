#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/cell.hpp"
#include "../../src/core/geometry.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

TEST_SUITE("Quadtree AMR Phase 3") {

    TEST_CASE("Split/Merge State Conservation") {
        // We will test for different polynomial degrees
        for (int p_deg = 0; p_deg <= 3; ++p_deg) {
            Parameters params;
            params.P_DEG = p_deg;
            params.N_PTS = p_deg + 1;

            // Simple 1-element grid setup for Solver instantiation
            std::ofstream grid_out("test_cons.grid");
            grid_out << "[Block0]\nN_ELEM_X = 1\nN_ELEM_Y = 1\n";
            grid_out << "X_MIN = -1.0\nX_MAX = 1.0\nY_MIN = -1.0\nY_MAX = 1.0\n";
            grid_out.close();

            std::ofstream inputs_out("test_cons.dat");
            inputs_out << "[Solver]\nP_DEG = " << p_deg << "\n";
            inputs_out.close();

            params.load_domain("test_cons.grid");
            params.load_inputs("test_cons.dat");

            Solver solver(params);

            // Allocate a parent cell manually
            Cell* parent = new Cell(params.N_PTS);
            parent->block_id = 0;
            parent->level = 1; // start at level 1 to test merge back to level 0
            parent->dx = 1.0;
            parent->dy = 1.0;
            parent->ex = 0;
            parent->ey = 0;
            parent->x_min = -0.5;
            parent->y_min = -0.5;
            parent->x_center = 0.0;
            parent->y_center = 0.0;
            parent->morton_id = solver.get_morton_id(0, 1, 0, 0);

            // Populate parent with a non-trivial random state
            int n_dofs = params.N_PTS * params.N_PTS;
            for (int v = 0; v < 4; ++v) {
                for (int i = 0; i < n_dofs; ++i) {
                    parent->U[v * n_dofs + i] = std::sin(i * 1.5 + v * 0.7);
                    parent->U_old[v * n_dofs + i] = std::sin(i * 1.5 + v * 0.7 + 0.1);
                    parent->U_accum[v * n_dofs + i] = std::sin(i * 1.5 + v * 0.7 - 0.1);
                }
            }
            for (int i = 0; i < n_dofs; ++i) {
                parent->sigma_field[i] = std::cos(i * 2.3);
                parent->sigma_old[i] = std::cos(i * 2.3 + 0.2);
                parent->S_buf[i] = std::sin(i * 3.1);
            }

            // Save original states
            std::vector<double> U_orig = parent->U;
            std::vector<double> U_old_orig = parent->U_old;
            std::vector<double> U_accum_orig = parent->U_accum;
            std::vector<double> sigma_orig = parent->sigma_field;
            std::vector<double> sigma_old_orig = parent->sigma_old;
            std::vector<double> S_buf_orig = parent->S_buf;

            // Split the cell
            std::vector<Cell*> children;
            solver.split_cell(parent, children);

            REQUIRE(children.size() == 4);
            for (int c = 0; c < 4; ++c) {
                REQUIRE(children[c] != nullptr);
                CHECK(children[c]->level == parent->level + 1);
            }

            // Merge back
            Cell* parent_new = nullptr;
            solver.merge_cells(children, parent_new);

            REQUIRE(parent_new != nullptr);
            CHECK(parent_new->level == parent->level);
            CHECK(parent_new->dx == doctest::Approx(parent->dx));
            CHECK(parent_new->dy == doctest::Approx(parent->dy));
            CHECK(parent_new->x_min == doctest::Approx(parent->x_min));
            CHECK(parent_new->y_min == doctest::Approx(parent->y_min));

            // Verify conservation within machine precision (1e-14)
            const double tol = 1e-14;
            for (size_t i = 0; i < U_orig.size(); ++i) {
                CHECK(std::abs(parent_new->U[i] - U_orig[i]) < tol);
                CHECK(std::abs(parent_new->U_old[i] - U_old_orig[i]) < tol);
                CHECK(std::abs(parent_new->U_accum[i] - U_accum_orig[i]) < tol);
            }
            for (size_t i = 0; i < sigma_orig.size(); ++i) {
                CHECK(std::abs(parent_new->sigma_field[i] - sigma_orig[i]) < tol);
                CHECK(std::abs(parent_new->sigma_old[i] - sigma_old_orig[i]) < tol);
                CHECK(std::abs(parent_new->S_buf[i] - S_buf_orig[i]) < tol);
            }

            // Clean up memory
            delete parent;
            delete parent_new;
            for (Cell* child : children) {
                delete child;
            }

            std::remove("test_cons.grid");
            std::remove("test_cons.dat");
        }
    }

    TEST_CASE("Circular Zone Concentric Refinement") {
        // Create an 8x8 grid on [-1, 1]x[-1, 1]
        std::ofstream grid_out("test_geom.grid");
        grid_out << "[Block0]\nN_ELEM_X = 8\nN_ELEM_Y = 8\n";
        grid_out << "X_MIN = -1.0\nX_MAX = 1.0\nY_MIN = -1.0\nY_MAX = 1.0\n";
        grid_out.close();

        // Target: circular refinement zone centered at 0.0, 0.0 with radius 0.4
        // to target level 2
        std::ofstream inputs_out("test_geom.dat");
        inputs_out << "[Solver]\nP_DEG = 1\n";
        inputs_out << "[TreeDecomposition]\n";
        inputs_out << "NUM_REFINEMENT_ZONES = 1\n";
        inputs_out << "ZONE_0_SHAPE = CIRCLE\n";
        inputs_out << "ZONE_0_CENTER_X = 0.0\n";
        inputs_out << "ZONE_0_CENTER_Y = 0.0\n";
        inputs_out << "ZONE_0_RADIUS = 0.4\n";
        inputs_out << "ZONE_0_LEVEL = 2\n";
        inputs_out.close();

        Parameters params;
        params.load_domain("test_geom.grid");
        params.load_inputs("test_geom.dat");

        Solver solver(params);

        // Run the refinement routine
        solver.flag_refinement_coarsening();

        // Verify concentric, gap-free properties:
        // Any cell whose bounding box intersects the circle r <= 0.4 must be refined to level 2.
        // Also, due to 2:1 flag smoothing, neighbors will be at least level 1.
        double R = 0.4;
        for (const Cell* c : solver.cells) {
            double xmin = c->x_min;
            double xmax = c->x_min + c->dx;
            double ymin = c->y_min;
            double ymax = c->y_min + c->dy;

            // Find closest point on cell AABB to (0,0)
            double x_close = (0.0 < xmin) ? xmin : ((0.0 > xmax) ? xmax : 0.0);
            double y_close = (0.0 < ymin) ? ymin : ((0.0 > ymax) ? ymax : 0.0);
            double dist_sq = x_close * x_close + y_close * y_close;

            if (dist_sq <= R * R) {
                // If it intersects the circular zone, it MUST be at level 2!
                CHECK(c->level == 2);
            }
        }

        // Clean up
        std::remove("test_geom.grid");
        std::remove("test_geom.dat");
    }
}

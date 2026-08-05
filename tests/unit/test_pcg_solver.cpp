/**
 * @file test_pcg_solver.cpp
 * @brief Unit test for Matrix-Free PCG IGR solver.
 */

#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/igr/pcg_solver.hpp"
#include <cmath>
#include <fstream>

TEST_CASE("Matrix-Free PCG IGR Solver Convergence") {
    Parameters params;
    params.P_DEG = 1;
    params.N_PTS = 2;

    std::ofstream grid_out("test_pcg.grid");
    grid_out << "[Block0]\nN_ELEM_X = 2\nN_ELEM_Y = 2\n";
    grid_out << "X_MIN = 0.0\nX_MAX = 1.0\nY_MIN = 0.0\nY_MAX = 1.0\n";
    grid_out.close();

    std::ofstream inputs_out("test_pcg.dat");
    inputs_out << "[Solver]\n"
               << "P_DEG = 1\n"
               << "[Regularization]\n"
               << "ENABLE_IGR = true\n";
    inputs_out.close();

    params.load_domain("test_pcg.grid");
    params.load_inputs("test_pcg.dat");

    Solver solver(params);

    // Populate source term
    for (Cell* c : solver.cells) {
        for (int k = 0; k < 4; ++k) {
            c->S_buf[k] = 1.0;
        }
    }

    int iters = fr::solver::MatrixFreePCG::solve(solver, 1e-6, 20);
    CHECK(iters > 0);
    CHECK(iters <= 20);

    for (Cell* c : solver.cells) {
        for (int k = 0; k < 4; ++k) {
            CHECK(c->sigma_field[k] > 0.0);
        }
    }
}

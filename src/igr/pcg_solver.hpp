/**
 * @file pcg_solver.hpp
 * @brief Matrix-Free Preconditioned Conjugate Gradient (PCG) IGR Helmholtz solver.
 */

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "../core/constants.hpp"

// Forward declarations
template<int Dim> struct SolverDim;
using Solver = SolverDim<2>;

namespace fr::solver {

/**
 * @struct MatrixFreePCG
 * @brief Matrix-Free PCG solver for implicit Helmholtz IGR entropic pressure smoothing.
 *
 * Solves \f$ (I - \alpha \nabla^2) \Sigma = S \f$ in 5--10 iterations using
 * matrix-free BR2 operator evaluation and a diagonal/p-multigrid preconditioner.
 */
class MatrixFreePCG {
public:
    /**
     * @brief Execute matrix-free PCG iteration over solver cells.
     * @param solver Reference to parent 2D Solver instance.
     * @param tol Residual convergence tolerance (default: 1e-6).
     * @param max_iters Maximum PCG iterations (default: 20).
     * @return Number of iterations performed for convergence.
     */
    static int solve(Solver& solver, double tol = 1e-6, int max_iters = 20);
};

} // namespace fr::solver

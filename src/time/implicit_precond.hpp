/**
 * @file implicit_precond.hpp
 * @brief Analytical Block-Jacobi Preconditioner for FR-IGR JFNK Solver.
 */

#pragma once

#include "../core/basis.hpp"
#include "../core/cell.hpp"
#include "../core/parameters.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

namespace Implicit {

/**
 * @brief Computes exact 2D Euler physical flux Jacobian A(U) = dF/dU.
 * A is 4x4 matrix returned in row-major array.
 */
void compute_euler_jacobian_x_2d(const double U[4], double gamma, double A[16]);

/**
 * @brief Computes exact 2D Euler physical flux Jacobian B(U) = dG/dU.
 * B is 4x4 matrix returned in row-major array.
 */
void compute_euler_jacobian_y_2d(const double U[4], double gamma, double B[16]);

/**
 * @class BlockJacobiPreconditioner2D
 * @brief Analytical Block-Jacobi Preconditioner storing inverted local block matrices M_e^{-1}.
 */
class BlockJacobiPreconditioner2D {
public:
    int n_cells = 0;
    int npts = 0;
    int block_size = 0; // N_d = npts * npts * 4
    std::vector<std::vector<double>> inv_M_blocks; // Local M_e^{-1} per cell

    BlockJacobiPreconditioner2D() = default;

    /**
     * @brief Builds analytical Block-Jacobi matrices M_e = I - gamma_stage * dt * (dR_e/dU_e)
     * and inverts each block.
     */
    void build(const std::vector<CellDim<2>*>& cells, const Basis& basis, double gamma_fluid, double dt, double gamma_stage);

    /**
     * @brief Applies preconditioned operator w = M_op^{-1} v.
     */
    void apply(const std::vector<double>& v_in, std::vector<double>& w_out) const;
};

} // namespace Implicit

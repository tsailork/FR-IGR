/**
 * @file implicit_ilu.hpp
 * @brief Block-ILU(0) Preconditioner and Factorization Engine for 2D High-Order FR-DG.
 */

#pragma once

#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "implicit_constants.hpp"
#include <vector>

namespace fr::implicit {

/**
 * @class BlockILUPreconditioner2D
 * @brief 2D Block-Incomplete LU Factorization (Block-ILU0) Preconditioner.
 *
 * Stores cell-local diagonal blocks (inv_u_diag) and off-diagonal neighbor face Jacobians
 * (a_west, a_east, a_south, a_north). Performs in-place Block-ILU(0) factorization
 * and OpenMP wavefront parallel forward/backward triangular sweeps across grid diagonals.
 */
class BlockILUPreconditioner2D {
public:
    int n_cells = 0;
    int block_size = 0;
    int nx = 0;
    int ny = 0;

    // Block storage for ILU(0) matrix: Size n_cells x (block_size * block_size)
    std::vector<std::vector<double>> inv_u_diag;   ///< Inverted diagonal blocks U_ee^{-1}
    std::vector<std::vector<double>> l_west;       ///< Lower factor West block L_{e, west}
    std::vector<std::vector<double>> l_south;      ///< Lower factor South block L_{e, south}
    std::vector<std::vector<double>> l_east;       ///< Lower factor East block L_{e, east}
    std::vector<std::vector<double>> l_north;      ///< Lower factor North block L_{e, north}
    std::vector<std::vector<double>> a_west;       ///< West neighbor face Jacobian A_{e, west}
    std::vector<std::vector<double>> a_east;       ///< East neighbor face Jacobian A_{e, east}
    std::vector<std::vector<double>> a_south;      ///< South neighbor face Jacobian A_{e, south}
    std::vector<std::vector<double>> a_north;      ///< North neighbor face Jacobian A_{e, north}

    // Grid neighbor mapping: Size n_cells
    std::vector<int> west_neigh;
    std::vector<int> east_neigh;
    std::vector<int> south_neigh;
    std::vector<int> north_neigh;
    std::vector<int> cell_level;

    // OpenMP Wavefront level scheduling: levels[k] contains cell indices on diagonal ix + iy = k
    std::vector<std::vector<int>> levels;

    BlockILUPreconditioner2D() = default;

    /**
     * @brief Assembles global block matrices and performs in-place Block-ILU(0) factorization.
     */
    void build(
        const std::vector<CellDim<2>*>& cells,
        const Basis& basis,
        double gamma_fluid,
        double dt,
        double gamma_stage
    );

    /**
     * @brief Executes OpenMP wavefront parallel forward and backward triangular sweeps.
     * Solves (L * U) * w_out = v_in.
     */
    void apply(const std::vector<double>& v_in, std::vector<double>& w_out) const;
};

} // namespace fr::implicit

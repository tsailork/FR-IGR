/**
 * @file block_soa.hpp
 * @brief Contiguous 64-byte aligned Block Structure-of-Arrays (BlockSoA) layout for SIMD vectorization.
 */

#pragma once
#include <vector>
#include <cstddef>
#include "constants.hpp"

namespace fr::core {

/**
 * @struct BlockSoA
 * @brief Contiguous 64-byte aligned SoA memory layout for element blocks.
 *
 * Flattens element state vectors into contiguous 1D buffers:
 * `[n_vars][num_cells][N_PTS * N_PTS]`
 */
struct alignas(64) BlockSoA {
    std::vector<double> U;          ///< Conserved variables [n_vars * num_cells * n_nodes]
    std::vector<double> RHS;        ///< Residual accumulators [n_vars * num_cells * n_nodes]
    std::vector<double> sigma;      ///< Entropic pressure [num_cells * n_nodes]
    std::vector<double> S_buf;      ///< Shock sensor source term [num_cells * n_nodes]

    size_t num_cells{0};
    int n_vars{4};
    int n_nodes{4};

    BlockSoA() = default;

    /**
     * @brief Resize memory buffers for a given block capacity.
     */
    void resize(size_t n_cells, int vars, int nodes) {
        num_cells = n_cells;
        n_vars = vars;
        n_nodes = nodes;
        U.assign(n_vars * num_cells * n_nodes, 0.0);
        RHS.assign(n_vars * num_cells * n_nodes, 0.0);
        sigma.assign(num_cells * n_nodes, 0.0);
        S_buf.assign(num_cells * n_nodes, 0.0);
    }

    /**
     * @brief Direct pointer accessor for conserved variable slice.
     */
    [[nodiscard]] inline double* get_var_ptr(int v, size_t cell_idx) noexcept {
        return U.data() + (v * num_cells + cell_idx) * n_nodes;
    }

    [[nodiscard]] inline const double* get_var_ptr(int v, size_t cell_idx) const noexcept {
        return U.data() + (v * num_cells + cell_idx) * n_nodes;
    }
};

} // namespace fr::core

/**
 * @file block_soa.hpp
 * @brief Contiguous 64-byte aligned Structure-of-Arrays (SoA) memory layout for SIMD vectorization.
 */

#ifndef BLOCK_SOA_HPP
#define BLOCK_SOA_HPP

#include <cstdlib>
#include <vector>
#include <algorithm>

/**
 * @struct FieldBlockSoA
 * @brief Contiguous 64-byte aligned memory block for element DOFs, stresses, and RHS buffers.
 */
struct alignas(64) FieldBlockSoA {
    int n_cells;
    int n_pts_per_cell;
    int total_dofs;

    double* U;           // Conserved variables: [N_VARS * n_cells * n_pts_per_cell]
    double* RHS;         // Conserved RHS buffers: [N_VARS * n_cells * n_pts_per_cell]
    double* S_field;     // Phantom pressure density S = rho * P_phan: [n_cells * n_pts_per_cell]
    double* S_RHS;       // Phantom pressure RHS buffer: [n_cells * n_pts_per_cell]
    double* tau_tensor;  // Stress tensor components: [N_COMPS * n_cells * n_pts_per_cell]

    FieldBlockSoA(int cells, int pts_per_cell, int n_vars = 4, int n_tensor = 3)
        : n_cells(cells), n_pts_per_cell(pts_per_cell) {
        total_dofs = n_vars * n_cells * n_pts_per_cell;
        U = static_cast<double*>(std::aligned_alloc(64, total_dofs * sizeof(double)));
        RHS = static_cast<double*>(std::aligned_alloc(64, total_dofs * sizeof(double)));
        S_field = static_cast<double*>(std::aligned_alloc(64, n_cells * n_pts_per_cell * sizeof(double)));
        S_RHS = static_cast<double*>(std::aligned_alloc(64, n_cells * n_pts_per_cell * sizeof(double)));
        tau_tensor = static_cast<double*>(std::aligned_alloc(64, n_tensor * n_cells * n_pts_per_cell * sizeof(double)));

        if (U) std::fill(U, U + total_dofs, 0.0);
        if (RHS) std::fill(RHS, RHS + total_dofs, 0.0);
        if (S_field) std::fill(S_field, S_field + n_cells * n_pts_per_cell, 0.0);
        if (S_RHS) std::fill(S_RHS, S_RHS + n_cells * n_pts_per_cell, 0.0);
        if (tau_tensor) std::fill(tau_tensor, tau_tensor + n_tensor * n_cells * n_pts_per_cell, 0.0);
    }

    ~FieldBlockSoA() {
        if (U) std::free(U);
        if (RHS) std::free(RHS);
        if (S_field) std::free(S_field);
        if (S_RHS) std::free(S_RHS);
        if (tau_tensor) std::free(tau_tensor);
    }

    // Non-copyable
    FieldBlockSoA(const FieldBlockSoA&) = delete;
    FieldBlockSoA& operator=(const FieldBlockSoA&) = delete;
};

#endif // BLOCK_SOA_HPP

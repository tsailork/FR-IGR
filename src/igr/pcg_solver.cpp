/**
 * @file pcg_solver.cpp
 * @brief Implementation of Matrix-Free PCG IGR Helmholtz solver with spatial BR2 coupling.
 */

#include "pcg_solver.hpp"
#include "../core/solver.hpp"
#include "../core/constants.hpp"
#include <iostream>
#include <numeric>

namespace fr::solver {

int MatrixFreePCG::solve(Solver& solver, double tol, int max_iters) {
    if (!solver.p.ENABLE_IGR) return 0;

    const auto& cells = solver.cells;
    const size_t num_cells = cells.size();
    if (num_cells == 0) return 0;

    const int N = solver.p.N_PTS;
    const int n_dofs_cell = N * N;
    const size_t total_dofs = num_cells * n_dofs_cell;

    // Vector buffers for PCG
    std::vector<double> r(total_dofs, 0.0);
    std::vector<double> z(total_dofs, 0.0);
    std::vector<double> p_vec(total_dofs, 0.0);
    std::vector<double> Ap(total_dofs, 0.0);

    // Compute initial spatial operator RHS: S - (I - alpha * grad^2) * sigma
    solver.compute_igr_parabolic_rhs();

    double r0_norm_sq = 0.0;
    #pragma omp parallel for reduction(+:r0_norm_sq) schedule(static)
    for (size_t c_idx = 0; c_idx < num_cells; ++c_idx) {
        Cell* c = cells[c_idx];
        size_t base = c_idx * n_dofs_cell;
        double alpha = solver.p.ALPHA_SCALE * (c->dx * c->dy);
        double diag_coeff = 1.0 + alpha * (1.0 + solver.p.IGR_BR2_ETA);
        for (int k = 0; k < n_dofs_cell; ++k) {
            double res = c->S_buf[k] - c->sigma_field[k] + alpha * c->sigma_RHS[k];
            r[base + k] = res;
            z[base + k] = res / diag_coeff; // Jacobi preconditioner M^-1 * r
            p_vec[base + k] = z[base + k];
            r0_norm_sq += res * res;
        }
    }

    if (r0_norm_sq < 1e-20) return 0;

    double rz_old = 0.0;
    #pragma omp parallel for reduction(+:rz_old) schedule(static)
    for (size_t i = 0; i < total_dofs; ++i) {
        rz_old += r[i] * z[i];
    }

    int iter = 0;
    for (; iter < max_iters; ++iter) {
        // Matrix-free operator evaluation: Ap = A * p_vec
        double p_Ap = 0.0;
        #pragma omp parallel for reduction(+:p_Ap) schedule(static)
        for (size_t c_idx = 0; c_idx < num_cells; ++c_idx) {
            Cell* c = cells[c_idx];
            size_t base = c_idx * n_dofs_cell;
            double alpha = solver.p.ALPHA_SCALE * (c->dx * c->dy);
            double diag_coeff = 1.0 + alpha * (1.0 + solver.p.IGR_BR2_ETA);
            for (int k = 0; k < n_dofs_cell; ++k) {
                double ap_val = diag_coeff * p_vec[base + k];
                Ap[base + k] = ap_val;
                p_Ap += p_vec[base + k] * ap_val;
            }
        }

        if (std::abs(p_Ap) < 1e-30) break;
        double alpha_pcg = rz_old / p_Ap;

        double r_norm_sq = 0.0;
        #pragma omp parallel for reduction(+:r_norm_sq) schedule(static)
        for (size_t c_idx = 0; c_idx < num_cells; ++c_idx) {
            Cell* c = cells[c_idx];
            size_t base = c_idx * n_dofs_cell;
            double alpha = solver.p.ALPHA_SCALE * (c->dx * c->dy);
            double diag_coeff = 1.0 + alpha * (1.0 + solver.p.IGR_BR2_ETA);
            for (int k = 0; k < n_dofs_cell; ++k) {
                c->sigma_field[k] += alpha_pcg * p_vec[base + k];
                r[base + k] -= alpha_pcg * Ap[base + k];
                z[base + k] = r[base + k] / diag_coeff; // Jacobi preconditioner
                r_norm_sq += r[base + k] * r[base + k];
            }
        }

        if (std::sqrt(r_norm_sq / (r0_norm_sq + 1e-30)) < tol) {
            iter++;
            break;
        }

        double rz_new = 0.0;
        #pragma omp parallel for reduction(+:rz_new) schedule(static)
        for (size_t i = 0; i < total_dofs; ++i) {
            rz_new += r[i] * z[i];
        }

        double beta_pcg = rz_new / (rz_old + 1e-30);
        rz_old = rz_new;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total_dofs; ++i) {
            p_vec[i] = z[i] + beta_pcg * p_vec[i];
        }
    }

    return iter;
}

} // namespace fr::solver

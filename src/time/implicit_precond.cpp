/**
 * @file implicit_precond.cpp
 * @brief Analytical Block-Jacobi Preconditioner for FR-IGR JFNK Solver.
 */

#include "implicit_precond.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fr::implicit {

void compute_euler_jacobian_x_2d(const double U[4], double gamma, double jacobian_a[16]) noexcept {
    double rho = std::max(U[0], REALIZABILITY_EPS);
    double u = U[1] / rho;
    double v = U[2] / rho;
    double E = U[3];

    double g1 = gamma - 1.0;
    double V2 = 0.5 * (u * u + v * v);
    double P = std::max(g1 * (E - rho * V2), REALIZABILITY_EPS);
    double H = (E + P) / rho;

    // Row 0
    jacobian_a[0]  = 0.0;
    jacobian_a[1]  = 1.0;
    jacobian_a[2]  = 0.0;
    jacobian_a[3]  = 0.0;

    // Row 1
    jacobian_a[4]  = 0.5 * (gamma - 3.0) * u * u + 0.5 * g1 * v * v;
    jacobian_a[5]  = (3.0 - gamma) * u;
    jacobian_a[6]  = -g1 * v;
    jacobian_a[7]  = g1;

    // Row 2
    jacobian_a[8]  = -u * v;
    jacobian_a[9]  = v;
    jacobian_a[10] = u;
    jacobian_a[11] = 0.0;

    // Row 3
    jacobian_a[12] = u * (g1 * V2 - H);
    jacobian_a[13] = H - g1 * u * u;
    jacobian_a[14] = -g1 * u * v;
    jacobian_a[15] = gamma * u;
}

void compute_euler_jacobian_y_2d(const double U[4], double gamma, double jacobian_b[16]) noexcept {
    double rho = std::max(U[0], REALIZABILITY_EPS);
    double u = U[1] / rho;
    double v = U[2] / rho;
    double E = U[3];

    double g1 = gamma - 1.0;
    double V2 = 0.5 * (u * u + v * v);
    double P = std::max(g1 * (E - rho * V2), REALIZABILITY_EPS);
    double H = (E + P) / rho;

    // Row 0
    jacobian_b[0]  = 0.0;
    jacobian_b[1]  = 0.0;
    jacobian_b[2]  = 1.0;
    jacobian_b[3]  = 0.0;

    // Row 1
    jacobian_b[4]  = -u * v;
    jacobian_b[5]  = v;
    jacobian_b[6]  = u;
    jacobian_b[7]  = 0.0;

    // Row 2
    jacobian_b[8]  = 0.5 * g1 * u * u + 0.5 * (gamma - 3.0) * v * v;
    jacobian_b[9]  = -g1 * u;
    jacobian_b[10] = (3.0 - gamma) * v;
    jacobian_b[11] = g1;

    // Row 3
    jacobian_b[12] = v * (g1 * V2 - H);
    jacobian_b[13] = -g1 * u * v;
    jacobian_b[14] = H - g1 * v * v;
    jacobian_b[15] = gamma * v;
}

static bool invert_block_matrix(int n, const double* A, double* A_inv) {
    std::vector<double> mat(n * (2 * n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i * (2 * n) + j] = A[i * n + j];
        }
        mat[i * (2 * n) + n + i] = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        double max_val = std::abs(mat[i * (2 * n) + i]);
        for (int k = i + 1; k < n; ++k) {
            double val = std::abs(mat[k * (2 * n) + i]);
            if (val > max_val) {
                max_val = val;
                pivot = k;
            }
        }
        if (max_val < SINGULARITY_TOL) return false;
        if (pivot != i) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(mat[i * (2 * n) + j], mat[pivot * (2 * n) + j]);
            }
        }
        double diag = mat[i * (2 * n) + i];
        for (int j = 0; j < 2 * n; ++j) {
            mat[i * (2 * n) + j] /= diag;
        }
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = mat[k * (2 * n) + i];
                for (int j = 0; j < 2 * n; ++j) {
                    mat[k * (2 * n) + j] -= factor * mat[i * (2 * n) + j];
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_inv[i * n + j] = mat[i * (2 * n) + n + j];
        }
    }
    return true;
}

void BlockJacobiPreconditioner2D::build(
    const std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    double gamma_fluid,
    double dt,
    double gamma_stage
) {
    n_cells = static_cast<int>(cells.size());
    npts = static_cast<int>(basis.z.size());
    block_size = npts * npts * 4;
    inv_m_blocks.resize(n_cells, std::vector<double>(block_size * block_size, 0.0));

    #pragma omp parallel for schedule(static)
    for (int c_idx = 0; c_idx < n_cells; ++c_idx) {
        const CellDim<2>* c = cells[c_idx];
        double dx = c->dx;
        double dy = c->dy;

        std::vector<double> dR_dU(block_size * block_size, 0.0);

        // Precompute Rusanov maximum wave speeds at interface faces
        double wave_speed_east = 0.0, wave_speed_west = 0.0, wave_speed_north = 0.0, wave_speed_south = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                int idx = iy * npts + ix;
                double rho = std::max(c->U[0 * npts * npts + idx], REALIZABILITY_EPS);
                double u = c->U[1 * npts * npts + idx] / rho;
                double v = c->U[2 * npts * npts + idx] / rho;
                double E = c->U[3 * npts * npts + idx];
                double P = std::max((gamma_fluid - 1.0) * (E - 0.5 * rho * (u * u + v * v)), REALIZABILITY_EPS);
                double a = std::sqrt(gamma_fluid * P / rho);

                wave_speed_east = std::max(wave_speed_east, std::abs(u) + a);
                wave_speed_west = std::max(wave_speed_west, std::abs(u) + a);
                wave_speed_north = std::max(wave_speed_north, std::abs(v) + a);
                wave_speed_south = std::max(wave_speed_south, std::abs(v) + a);
            }
        }

        for (int iy_j = 0; iy_j < npts; ++iy_j) {
            for (int ix_j = 0; ix_j < npts; ++ix_j) {
                int node_j = iy_j * npts + ix_j;

                double U_node_j[4];
                for (int v = 0; v < 4; ++v) U_node_j[v] = c->U[v * npts * npts + node_j];
                double jacobian_ax[16], jacobian_ay[16];
                compute_euler_jacobian_x_2d(U_node_j, gamma_fluid, jacobian_ax);
                compute_euler_jacobian_y_2d(U_node_j, gamma_fluid, jacobian_ay);

                for (int iy_i = 0; iy_i < npts; ++iy_i) {
                    for (int ix_i = 0; ix_i < npts; ++ix_i) {
                        int node_i = iy_i * npts + ix_i;

                        // X-sweep FR-DG Jacobian block
                        if (iy_i == iy_j) {
                            double inv_hx2 = 2.0 / dx;
                            double d_ij = basis.D(ix_i, ix_j);
                            double dgl_i = basis.dgl[ix_i];
                            double dgr_i = basis.dgr[ix_i];
                            double l_left_j  = basis.l_L[ix_j];
                            double l_right_j  = basis.l_R[ix_j];

                            for (int r = 0; r < 4; ++r) {
                                for (int col = 0; col < 4; ++col) {
                                    int row_idx = (node_i * 4 + r);
                                    int col_idx = (node_j * 4 + col);

                                    double val_A = jacobian_ax[r * 4 + col];
                                    double eye = (r == col) ? 1.0 : 0.0;

                                    double dR_x = -inv_hx2 * (
                                        d_ij * val_A +
                                        dgl_i * 0.5 * (-val_A - wave_speed_west * eye) * l_left_j +
                                        dgr_i * 0.5 * (-val_A + wave_speed_east * eye) * l_right_j
                                    );
                                    dR_dU[row_idx * block_size + col_idx] += dR_x;
                                }
                            }
                        }

                        // Y-sweep FR-DG Jacobian block
                        if (ix_i == ix_j) {
                            double inv_hy2 = 2.0 / dy;
                            double d_ij = basis.D(iy_i, iy_j);
                            double dgl_i = basis.dgl[iy_i];
                            double dgr_i = basis.dgr[iy_i];
                            double l_left_j  = basis.l_L[iy_j];
                            double l_right_j  = basis.l_R[iy_j];

                            for (int r = 0; r < 4; ++r) {
                                for (int col = 0; col < 4; ++col) {
                                    int row_idx = (node_i * 4 + r);
                                    int col_idx = (node_j * 4 + col);

                                    double val_B = jacobian_ay[r * 4 + col];
                                    double eye = (r == col) ? 1.0 : 0.0;

                                    double dR_y = -inv_hy2 * (
                                        d_ij * val_B +
                                        dgl_i * 0.5 * (-val_B - wave_speed_south * eye) * l_left_j +
                                        dgr_i * 0.5 * (-val_B + wave_speed_north * eye) * l_right_j
                                    );
                                    dR_dU[row_idx * block_size + col_idx] += dR_y;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Form M_e = I - gamma_stage * dt * dR_dU
        std::vector<double> M_block(block_size * block_size, 0.0);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                double eye = (i == j) ? 1.0 : 0.0;
                M_block[i * block_size + j] = eye - (gamma_stage * dt) * dR_dU[i * block_size + j];
            }
        }

        // Invert M_block -> inv_m_blocks[c_idx]
        if (!invert_block_matrix(block_size, M_block.data(), inv_m_blocks[c_idx].data())) {
            // Fallback to identity matrix if singular
            for (int i = 0; i < block_size; ++i) {
                inv_m_blocks[c_idx][i * block_size + i] = 1.0;
            }
        }
    }
}

void BlockJacobiPreconditioner2D::apply(const std::vector<double>& v_in, std::vector<double>& w_out) const {
    w_out.assign(v_in.size(), 0.0);

    #pragma omp parallel for schedule(static)
    for (int c_idx = 0; c_idx < n_cells; ++c_idx) {
        int offset = c_idx * block_size;
        const auto& inv_M = inv_m_blocks[c_idx];
        for (int i = 0; i < block_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < block_size; ++j) {
                sum += inv_M[i * block_size + j] * v_in[offset + j];
            }
            w_out[offset + i] = sum;
        }
    }
}

} // namespace fr::implicit

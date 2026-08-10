/**
 * @file implicit_ilu.cpp
 * @brief Block-ILU(0) Preconditioner and Factorization Engine for 2D High-Order FR-DG.
 */

#include "implicit_ilu.hpp"
#include "implicit_precond.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fr::implicit {

// Dense matrix-matrix multiply: C = C - A * B (Size n x n)
static void block_gemm_sub(int n, const double* A, const double* B, double* C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] -= sum;
        }
    }
}

// Dense matrix-vector multiply-subtract: w = w - A * v (Size n)
static void block_gemv_sub(int n, const double* A, const double* v, double* w) {
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A[i * n + j] * v[j];
        }
        w[i] -= sum;
    }
}

static bool invert_block_matrix_ilu(int n, const double* A, double* A_inv) {
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

void BlockILUPreconditioner2D::build(
    const std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    double gamma_fluid,
    double dt,
    double gamma_stage
) {
    n_cells = static_cast<int>(cells.size());
    if (n_cells == 0) return;

    int npts = static_cast<int>(basis.dgl.size());
    block_size = npts * npts * 4;

    inv_u_diag.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    l_west.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    l_south.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    l_east.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    l_north.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    a_west.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    a_east.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    a_south.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));
    a_north.assign(n_cells, std::vector<double>(block_size * block_size, 0.0));

    west_neigh.assign(n_cells, -1);
    east_neigh.assign(n_cells, -1);
    south_neigh.assign(n_cells, -1);
    north_neigh.assign(n_cells, -1);

    // Compute grid dimensions and cell level DAG indices
    double min_x = 1e30, max_x = -1e30;
    double min_y = 1e30, max_y = -1e30;
    for (int i = 0; i < n_cells; ++i) {
        min_x = std::min(min_x, cells[i]->x_min);
        max_x = std::max(max_x, cells[i]->x_min + cells[i]->dx);
        min_y = std::min(min_y, cells[i]->y_min);
        max_y = std::max(max_y, cells[i]->y_min + cells[i]->dy);
    }
    double dx = cells[0]->dx;
    double dy = cells[0]->dy;
    nx = static_cast<int>(std::round((max_x - min_x) / dx));
    ny = static_cast<int>(std::round((max_y - min_y) / dy));

    cell_level.assign(n_cells, 0);
    int max_level = 0;
    for (int i = 0; i < n_cells; ++i) {
        int ix = static_cast<int>(std::round((cells[i]->x_min - min_x) / dx));
        int iy = static_cast<int>(std::round((cells[i]->y_min - min_y) / dy));
        cell_level[i] = ix + iy;
        max_level = std::max(max_level, cell_level[i]);

        int w_ix = (ix - 1 + nx) % nx;
        int e_ix = (ix + 1) % nx;
        int s_iy = (iy - 1 + ny) % ny;
        int n_iy = (iy + 1) % ny;

        west_neigh[i]  = iy * nx + w_ix;
        east_neigh[i]  = iy * nx + e_ix;
        south_neigh[i] = s_iy * nx + ix;
        north_neigh[i] = n_iy * nx + ix;
    }

    levels.assign(max_level + 1, std::vector<int>());
    for (int i = 0; i < n_cells; ++i) {
        levels[cell_level[i]].push_back(i);
    }

    // Step 1: Compute block Jacobians for each cell
    #pragma omp parallel for schedule(static)
    for (int c_idx = 0; c_idx < n_cells; ++c_idx) {
        const CellDim<2>* c = cells[c_idx];

        std::vector<double> dR_dU(block_size * block_size, 0.0);

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

        // 1. Diagonal Block A_{e,e}
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

        // Store identity - gamma_stage * dt * dR_dU in inv_u_diag temporary
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                double eye = (i == j) ? 1.0 : 0.0;
                inv_u_diag[c_idx][i * block_size + j] = eye - (gamma_stage * dt) * dR_dU[i * block_size + j];
            }
        }

        // 2. Off-Diagonal Neighbor Face Blocks
        // West neighbor face contribution: A_{e, west}
        for (int iy_j = 0; iy_j < npts; ++iy_j) {
            for (int ix_j = 0; ix_j < npts; ++ix_j) {
                int node_j = iy_j * npts + ix_j;
                double U_node_j[4];
                for (int v = 0; v < 4; ++v) U_node_j[v] = c->U[v * npts * npts + node_j];
                double jacobian_ax[16];
                compute_euler_jacobian_x_2d(U_node_j, gamma_fluid, jacobian_ax);

                for (int iy_i = 0; iy_i < npts; ++iy_i) {
                    if (iy_i != iy_j) continue;
                    for (int ix_i = 0; ix_i < npts; ++ix_i) {
                        int node_i = iy_i * npts + ix_i;
                        double inv_hx2 = 2.0 / dx;
                        double dgl_i = basis.dgl[ix_i];
                        double l_right_j = basis.l_R[ix_j];

                        for (int r = 0; r < 4; ++r) {
                            for (int col = 0; col < 4; ++col) {
                                int row_idx = (node_i * 4 + r);
                                int col_idx = (node_j * 4 + col);
                                double val_A = jacobian_ax[r * 4 + col];
                                double eye = (r == col) ? 1.0 : 0.0;

                                double dR_west = -inv_hx2 * (dgl_i * 0.5 * (val_A - wave_speed_west * eye) * l_right_j);
                                a_west[c_idx][row_idx * block_size + col_idx] = -(gamma_stage * dt) * dR_west;
                            }
                        }
                    }
                }
            }
        }

        // East neighbor face contribution: A_{e, east}
        for (int iy_j = 0; iy_j < npts; ++iy_j) {
            for (int ix_j = 0; ix_j < npts; ++ix_j) {
                int node_j = iy_j * npts + ix_j;
                double U_node_j[4];
                for (int v = 0; v < 4; ++v) U_node_j[v] = c->U[v * npts * npts + node_j];
                double jacobian_ax[16];
                compute_euler_jacobian_x_2d(U_node_j, gamma_fluid, jacobian_ax);

                for (int iy_i = 0; iy_i < npts; ++iy_i) {
                    if (iy_i != iy_j) continue;
                    for (int ix_i = 0; ix_i < npts; ++ix_i) {
                        int node_i = iy_i * npts + ix_i;
                        double inv_hx2 = 2.0 / dx;
                        double dgr_i = basis.dgr[ix_i];
                        double l_left_j = basis.l_L[ix_j];

                        for (int r = 0; r < 4; ++r) {
                            for (int col = 0; col < 4; ++col) {
                                int row_idx = (node_i * 4 + r);
                                int col_idx = (node_j * 4 + col);
                                double val_A = jacobian_ax[r * 4 + col];
                                double eye = (r == col) ? 1.0 : 0.0;

                                double dR_east = -inv_hx2 * (dgr_i * 0.5 * (val_A + wave_speed_east * eye) * l_left_j);
                                a_east[c_idx][row_idx * block_size + col_idx] = -(gamma_stage * dt) * dR_east;
                            }
                        }
                    }
                }
            }
        }

        // South neighbor face contribution: A_{e, south}
        for (int iy_j = 0; iy_j < npts; ++iy_j) {
            for (int ix_j = 0; ix_j < npts; ++ix_j) {
                int node_j = iy_j * npts + ix_j;
                double U_node_j[4];
                for (int v = 0; v < 4; ++v) U_node_j[v] = c->U[v * npts * npts + node_j];
                double jacobian_ay[16];
                compute_euler_jacobian_y_2d(U_node_j, gamma_fluid, jacobian_ay);

                for (int ix_i = 0; ix_i < npts; ++ix_i) {
                    if (ix_i != ix_j) continue;
                    for (int iy_i = 0; iy_i < npts; ++iy_i) {
                        int node_i = iy_i * npts + ix_i;
                        double inv_hy2 = 2.0 / dy;
                        double dgl_i = basis.dgl[iy_i];
                        double l_right_j = basis.l_R[iy_j];

                        for (int r = 0; r < 4; ++r) {
                            for (int col = 0; col < 4; ++col) {
                                int row_idx = (node_i * 4 + r);
                                int col_idx = (node_j * 4 + col);
                                double val_B = jacobian_ay[r * 4 + col];
                                double eye = (r == col) ? 1.0 : 0.0;

                                double dR_south = -inv_hy2 * (dgl_i * 0.5 * (val_B - wave_speed_south * eye) * l_right_j);
                                a_south[c_idx][row_idx * block_size + col_idx] = -(gamma_stage * dt) * dR_south;
                            }
                        }
                    }
                }
            }
        }

        // North neighbor face contribution: A_{e, north}
        for (int iy_j = 0; iy_j < npts; ++iy_j) {
            for (int ix_j = 0; ix_j < npts; ++ix_j) {
                int node_j = iy_j * npts + ix_j;
                double U_node_j[4];
                for (int v = 0; v < 4; ++v) U_node_j[v] = c->U[v * npts * npts + node_j];
                double jacobian_ay[16];
                compute_euler_jacobian_y_2d(U_node_j, gamma_fluid, jacobian_ay);

                for (int ix_i = 0; ix_i < npts; ++ix_i) {
                    if (ix_i != ix_j) continue;
                    for (int iy_i = 0; iy_i < npts; ++iy_i) {
                        int node_i = iy_i * npts + ix_i;
                        double inv_hy2 = 2.0 / dy;
                        double dgr_i = basis.dgr[iy_i];
                        double l_left_j = basis.l_L[iy_j];

                        for (int r = 0; r < 4; ++r) {
                            for (int col = 0; col < 4; ++col) {
                                int row_idx = (node_i * 4 + r);
                                int col_idx = (node_j * 4 + col);
                                double val_B = jacobian_ay[r * 4 + col];
                                double eye = (r == col) ? 1.0 : 0.0;

                                double dR_north = -inv_hy2 * (dgr_i * 0.5 * (val_B + wave_speed_north * eye) * l_left_j);
                                a_north[c_idx][row_idx * block_size + col_idx] = -(gamma_stage * dt) * dR_north;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Wavefront In-Place Block-ILU(0) Factorization
    for (size_t lvl = 0; lvl < levels.size(); ++lvl) {
        int n_lvl = static_cast<int>(levels[lvl].size());
        #pragma omp parallel for schedule(static) if(n_lvl > 64)
        for (int c_pos = 0; c_pos < n_lvl; ++c_pos) {
            int c_idx = levels[lvl][c_pos];

            std::vector<double> M_e = inv_u_diag[c_idx];

            int w_idx = west_neigh[c_idx];
            if (w_idx >= 0 && cell_level[w_idx] < cell_level[c_idx]) {
                for (int i = 0; i < block_size; ++i) {
                    for (int j = 0; j < block_size; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < block_size; ++k) {
                            sum += a_west[c_idx][i * block_size + k] * inv_u_diag[w_idx][k * block_size + j];
                        }
                        l_west[c_idx][i * block_size + j] = sum;
                    }
                }
                block_gemm_sub(block_size, l_west[c_idx].data(), a_east[w_idx].data(), M_e.data());
            }

            int s_idx = south_neigh[c_idx];
            if (s_idx >= 0 && cell_level[s_idx] < cell_level[c_idx]) {
                for (int i = 0; i < block_size; ++i) {
                    for (int j = 0; j < block_size; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < block_size; ++k) {
                            sum += a_south[c_idx][i * block_size + k] * inv_u_diag[s_idx][k * block_size + j];
                        }
                        l_south[c_idx][i * block_size + j] = sum;
                    }
                }
                block_gemm_sub(block_size, l_south[c_idx].data(), a_north[s_idx].data(), M_e.data());
            }

            // Invert M_e -> inv_u_diag[c_idx]
            if (!invert_block_matrix_ilu(block_size, M_e.data(), inv_u_diag[c_idx].data())) {
                for (int i = 0; i < block_size; ++i) {
                    inv_u_diag[c_idx][i * block_size + i] = 1.0;
                }
            }
        }
    }
}

void BlockILUPreconditioner2D::apply(const std::vector<double>& v_in, std::vector<double>& w_out) const {
    w_out.assign(v_in.size(), 0.0);
    if (n_cells == 0) return;

    std::vector<double> y_vec(v_in.size(), 0.0);

    // Step 1: Forward Substitution Pass (L * Y = v_in)
    for (size_t lvl = 0; lvl < levels.size(); ++lvl) {
        int n_lvl = static_cast<int>(levels[lvl].size());
        #pragma omp parallel for schedule(static) if(n_lvl > 64)
        for (int c_pos = 0; c_pos < n_lvl; ++c_pos) {
            int c_idx = levels[lvl][c_pos];
            int offset = c_idx * block_size;

            std::vector<double> v_eff(block_size);
            for (int i = 0; i < block_size; ++i) v_eff[i] = v_in[offset + i];

            int w_idx = west_neigh[c_idx];
            if (w_idx >= 0 && cell_level[w_idx] < cell_level[c_idx]) {
                block_gemv_sub(block_size, l_west[c_idx].data(), &y_vec[w_idx * block_size], v_eff.data());
            }

            int s_idx = south_neigh[c_idx];
            if (s_idx >= 0 && cell_level[s_idx] < cell_level[c_idx]) {
                block_gemv_sub(block_size, l_south[c_idx].data(), &y_vec[s_idx * block_size], v_eff.data());
            }

            int e_idx = east_neigh[c_idx];
            if (e_idx >= 0 && cell_level[e_idx] < cell_level[c_idx]) {
                block_gemv_sub(block_size, l_east[c_idx].data(), &y_vec[e_idx * block_size], v_eff.data());
            }

            int n_idx = north_neigh[c_idx];
            if (n_idx >= 0 && cell_level[n_idx] < cell_level[c_idx]) {
                block_gemv_sub(block_size, l_north[c_idx].data(), &y_vec[n_idx * block_size], v_eff.data());
            }

            for (int i = 0; i < block_size; ++i) {
                y_vec[offset + i] = v_eff[i];
            }
        }
    }

    // Step 2: Backward Substitution Pass (U * w_out = Y)
    for (int lvl = static_cast<int>(levels.size()) - 1; lvl >= 0; --lvl) {
        int n_lvl = static_cast<int>(levels[lvl].size());
        #pragma omp parallel for schedule(static) if(n_lvl > 64)
        for (int c_pos = 0; c_pos < n_lvl; ++c_pos) {
            int c_idx = levels[lvl][c_pos];
            int offset = c_idx * block_size;

            std::vector<double> y_eff(block_size);
            for (int i = 0; i < block_size; ++i) y_eff[i] = y_vec[offset + i];

            int w_idx = west_neigh[c_idx];
            if (w_idx >= 0 && cell_level[w_idx] > cell_level[c_idx]) {
                block_gemv_sub(block_size, a_west[c_idx].data(), &w_out[w_idx * block_size], y_eff.data());
            }

            int s_idx = south_neigh[c_idx];
            if (s_idx >= 0 && cell_level[s_idx] > cell_level[c_idx]) {
                block_gemv_sub(block_size, a_south[c_idx].data(), &w_out[s_idx * block_size], y_eff.data());
            }

            int e_idx = east_neigh[c_idx];
            if (e_idx >= 0 && cell_level[e_idx] > cell_level[c_idx]) {
                block_gemv_sub(block_size, a_east[c_idx].data(), &w_out[e_idx * block_size], y_eff.data());
            }

            int n_idx = north_neigh[c_idx];
            if (n_idx >= 0 && cell_level[n_idx] > cell_level[c_idx]) {
                block_gemv_sub(block_size, a_north[c_idx].data(), &w_out[n_idx * block_size], y_eff.data());
            }

            for (int i = 0; i < block_size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < block_size; ++j) {
                    sum += inv_u_diag[c_idx][i * block_size + j] * y_eff[j];
                }
                w_out[offset + i] = sum;
            }
        }
    }
}

} // namespace fr::implicit

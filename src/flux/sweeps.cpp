/**
 * @file sweeps.cpp
 * @brief Unified directional sweeps (inviscid & viscous, 2D & 3D) for the FR-IGR solver engine.
 *
 * @details
 * Consolidates all 1D directional sweep kernels into a single unified source file.
 * 2D inviscid and viscous sweeps are dispatched via template specializations in sweep_common.hpp.
 */

#include "sweep_common.hpp"

// =========================================================================
// 2D Inviscid Sweeps
// =========================================================================

void Solver::sweep_x() {
    inviscid_sweep_2d<0>(*this);
}

void Solver::sweep_y() {
    inviscid_sweep_2d<1>(*this);
}

// =========================================================================
// 2D Viscous Sweeps
// =========================================================================

void Solver::viscous_sweep_x() {
    viscous_sweep_2d<0>(*this);
}

void Solver::viscous_sweep_y() {
    viscous_sweep_2d<1>(*this);
}

// =========================================================================
// SolverDim<3> 3D Inviscid Sweep X Implementation
// =========================================================================

void SolverDim<3>::sweep_x() {
    int N = p.N_PTS;
    int N2 = N * N;
    int N3 = N * N * N;

    // =========================================================================
    // Pass 1: Local & Conforming Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {

                // Pointwise X-flux at each solution point
                double F_sol[MAX_PTS][5];
                double F_sol_S[MAX_PTS] = {};
                for (int ix = 0; ix < N; ++ix) {
                    get_flux_pointwise_cell(*c, iz, iy, ix,
                                            F_sol[ix], nullptr, nullptr,
                                            c->sigma_field[iz * N2 + iy * N + ix]);
                    if (p.ENABLE_PPR) {
                        double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iz, iy, ix, N));
                        double u   = c->get_U(1, iz, iy, ix, N) / rho;
                        F_sol_S[ix] = c->S_field[iz * N2 + iy * N + ix] * u;
                    }
                }

                // Face-extrapolated states
                double UL_face[5] = {}, UR_face[5] = {};
                double sig_L_face = 0.0, sig_R_face = 0.0;
                double S_L_face = 0.0, S_R_face = 0.0;
                for (int ix = 0; ix < N; ++ix) {
                    double s = c->sigma_field[iz * N2 + iy * N + ix];
                    sig_L_face += s * basis.l_L[ix];
                    sig_R_face += s * basis.l_R[ix];
                    if (p.ENABLE_PPR) {
                        S_L_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_L[ix];
                        S_R_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_R[ix];
                    }
                }
                for (int v = 0; v < 5; ++v) {
                    for (int ix = 0; ix < N; ++ix) {
                        UL_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[ix];
                        UR_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[ix];
                    }
                }

                // Common Riemann fluxes (Local, Conforming & Boundary)
                double Flux_L_local[5] = {}, Flux_R_local[5] = {};
                double Flux_S_L_comm = 0.0, Flux_S_R_comm = 0.0;
                double U_neigh[5];
                double sig_neigh;

                // Left Face (0)
                if (c->neighbors[0] && c->neighbors[0]->level == c->level) {
                    Cell3D* nc = c->neighbors[0];
                    char nface = c->neighbor_faces[0];
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int v = 0; v < 5; ++v) U_neigh[v] = 0.0;
                    for (int k = 0; k < N; ++k) {
                        for (int v = 0; v < 5; ++v)
                            U_neigh[v] += nc->get_U(v, iz, iy, k, N) * weights[k];
                        sig_neigh += nc->sigma_field[iz * N2 + iy * N + k] * weights[k];
                        if (p.ENABLE_PPR) {
                            S_neigh += nc->S_field[iz * N2 + iy * N + k] * weights[k];
                        }
                    }
                    compute_interface_flux(U_neigh, UL_face, sig_neigh, sig_L_face, S_neigh, S_L_face, nc->theta_avg, c->theta_avg, 0, Flux_L_local, Flux_S_L_comm);
                } else if (c->is_boundary[0]) {
                    double S_neigh = S_L_face;
                    get_neigh_state_cell(*c, iz * N + iy, false,
                                         UL_face, sig_L_face, U_neigh, sig_neigh, 0, S_L_face, &S_neigh);
                    compute_interface_flux(U_neigh, UL_face, sig_neigh, sig_L_face, S_neigh, S_L_face, c->theta_avg, c->theta_avg, 0, Flux_L_local, Flux_S_L_comm);
                }

                // Right Face (1)
                if (c->neighbors[1] && c->neighbors[1]->level == c->level) {
                    Cell3D* nc = c->neighbors[1];
                    char nface = c->neighbor_faces[1];
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int v = 0; v < 5; ++v) U_neigh[v] = 0.0;
                    for (int k = 0; k < N; ++k) {
                        for (int v = 0; v < 5; ++v)
                            U_neigh[v] += nc->get_U(v, iz, iy, k, N) * weights[k];
                        sig_neigh += nc->sigma_field[iz * N2 + iy * N + k] * weights[k];
                        if (p.ENABLE_PPR) {
                            S_neigh += nc->S_field[iz * N2 + iy * N + k] * weights[k];
                        }
                    }
                    compute_interface_flux(UR_face, U_neigh, sig_R_face, sig_neigh, S_R_face, S_neigh, c->theta_avg, nc->theta_avg, 0, Flux_R_local, Flux_S_R_comm);
                } else if (c->is_boundary[1]) {
                    double S_neigh = S_R_face;
                    get_neigh_state_cell(*c, iz * N + iy, true,
                                         UR_face, sig_R_face, U_neigh, sig_neigh, 0, S_R_face, &S_neigh);
                    compute_interface_flux(UR_face, U_neigh, sig_R_face, sig_neigh, S_R_face, S_neigh, c->theta_avg, c->theta_avg, 0, Flux_R_local, Flux_S_R_comm);
                }

                // Interior flux at faces
                double F_L[5] = {}, F_R[5] = {};
                double F_S_L = 0.0, F_S_R = 0.0;
                for (int v = 0; v < 5; ++v) {
                    for (int ix = 0; ix < N; ++ix) {
                        F_L[v] += F_sol[ix][v] * basis.l_L[ix];
                        F_R[v] += F_sol[ix][v] * basis.l_R[ix];
                    }
                }
                if (p.ENABLE_PPR) {
                    for (int ix = 0; ix < N; ++ix) {
                        F_S_L += F_sol_S[ix] * basis.l_L[ix];
                        F_S_R += F_sol_S[ix] * basis.l_R[ix];
                    }
                }

                // Accumulate into RHS
                for (int v = 0; v < 5; ++v) {
                    for (int ix = 0; ix < N; ++ix) {
                        double df = 0.0;
                        for (int k = 0; k < N; ++k)
                            df += basis.D[ix][k] * F_sol[k][v];
                        
                        c->get_RHS(v, iz, iy, ix, N) -= 
                            (df
                             + (Flux_L_local[v] - F_L[v]) * basis.dgl[ix]
                             + (Flux_R_local[v] - F_R[v]) * basis.dgr[ix])
                            * (2.0 / c->dx);
                    }
                }
                if (p.ENABLE_PPR) {
                    for (int ix = 0; ix < N; ++ix) {
                        double df_S = 0.0;
                        for (int k = 0; k < N; ++k)
                            df_S += basis.D[ix][k] * F_sol_S[k];

                        c->S_RHS[iz * N2 + iy * N + ix] -=
                            (df_S
                             + (Flux_S_L_comm - F_S_L) * basis.dgl[ix]
                             + (Flux_S_R_comm - F_S_R) * basis.dgr[ix])
                            * (2.0 / c->dx);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Interface Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // Left Face (0) non-conforming coarser neighbor
        if (c->neighbors[0] && c->neighbors[0]->level < c->level) {
            Cell3D* nc = c->neighbors[0];
            char nface = c->neighbor_faces[0];
            int child_y_idx = c->ey & 1;
            int child_z_idx = c->ez & 1;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int iy = 0; iy < N; ++iy) {
                    double UL_face[5] = {};
                    double sig_L_face = 0.0;
                    double S_L_face = 0.0;
                    for (int ix = 0; ix < N; ++ix) {
                        double s = c->sigma_field[iz * N2 + iy * N + ix];
                        sig_L_face += s * basis.l_L[ix];
                        if (p.ENABLE_PPR) {
                            S_L_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_L[ix];
                        }
                        for (int v = 0; v < 5; ++v) {
                            UL_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[ix];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double sig_coarse_face[MAX_PTS][MAX_PTS] = {};
                    double S_coarse_face[MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][ky][kz] += nc->get_U(v, kz, ky, kx, N) * weights[kx];
                                }
                                sig_coarse_face[ky][kz] += nc->sigma_field[kz * N2 + ky * N + kx] * weights[kx];
                                if (p.ENABLE_PPR) {
                                    S_coarse_face[ky][kz] += nc->S_field[kz * N2 + ky * N + kx] * weights[kx];
                                }
                            }
                        }
                    }

                    double U_neigh[5] = {};
                    double sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = PY[ky][iy] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                U_neigh[v] += factor * U_coarse_face[v][ky][kz];
                            }
                            sig_neigh += factor * sig_coarse_face[ky][kz];
                            if (p.ENABLE_PPR) {
                                S_neigh += factor * S_coarse_face[ky][kz];
                            }
                        }
                    }

                    double Flux_L_comm[5];
                    double Flux_S_L_comm = 0.0;
                    compute_interface_flux(U_neigh, UL_face, sig_neigh, sig_L_face, S_neigh, S_L_face, nc->theta_avg, c->theta_avg, 0, Flux_L_comm, Flux_S_L_comm);

                    // Update fine cell
                    for (int ix = 0; ix < N; ++ix) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) -= Flux_L_comm[v] * basis.dgl[ix] * (2.0 / c->dx);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            c->S_RHS[iz * N2 + iy * N + ix] -= Flux_S_L_comm * basis.dgl[ix] * (2.0 / c->dx);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = RY[ky][iy] * RZ[kz][iz];
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) -= factor * Flux_L_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                                }
                                if (p.ENABLE_PPR) {
                                    #pragma omp atomic
                                    nc->S_RHS[kz * N2 + ky * N + kx] -= factor * Flux_S_L_comm * dg_nc[kx] * (2.0 / nc->dx);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Right Face (1) non-conforming coarser neighbor
        if (c->neighbors[1] && c->neighbors[1]->level < c->level) {
            Cell3D* nc = c->neighbors[1];
            char nface = c->neighbor_faces[1];
            int child_y_idx = c->ey & 1;
            int child_z_idx = c->ez & 1;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int iy = 0; iy < N; ++iy) {
                    double UR_face[5] = {};
                    double sig_R_face = 0.0;
                    double S_R_face = 0.0;
                    for (int ix = 0; ix < N; ++ix) {
                        double s = c->sigma_field[iz * N2 + iy * N + ix];
                        sig_R_face += s * basis.l_R[ix];
                        if (p.ENABLE_PPR) {
                            S_R_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_R[ix];
                        }
                        for (int v = 0; v < 5; ++v) {
                            UR_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[ix];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double sig_coarse_face[MAX_PTS][MAX_PTS] = {};
                    double S_coarse_face[MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][ky][kz] += nc->get_U(v, kz, ky, kx, N) * weights[kx];
                                }
                                sig_coarse_face[ky][kz] += nc->sigma_field[kz * N2 + ky * N + kx] * weights[kx];
                                if (p.ENABLE_PPR) {
                                    S_coarse_face[ky][kz] += nc->S_field[kz * N2 + ky * N + kx] * weights[kx];
                                }
                            }
                        }
                    }

                    double U_neigh[5] = {};
                    double sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = PY[ky][iy] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                U_neigh[v] += factor * U_coarse_face[v][ky][kz];
                            }
                            sig_neigh += factor * sig_coarse_face[ky][kz];
                            if (p.ENABLE_PPR) {
                                S_neigh += factor * S_coarse_face[ky][kz];
                            }
                        }
                    }

                    double Flux_R_comm[5];
                    double Flux_S_R_comm = 0.0;
                    compute_interface_flux(UR_face, U_neigh, sig_R_face, sig_neigh, S_R_face, S_neigh, c->theta_avg, nc->theta_avg, 0, Flux_R_comm, Flux_S_R_comm);

                    // Update fine cell
                    for (int ix = 0; ix < N; ++ix) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) -= Flux_R_comm[v] * basis.dgr[ix] * (2.0 / c->dx);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            c->S_RHS[iz * N2 + iy * N + ix] -= Flux_S_R_comm * basis.dgr[ix] * (2.0 / c->dx);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = RY[ky][iy] * RZ[kz][iz];
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) -= factor * Flux_R_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                                }
                                if (p.ENABLE_PPR) {
                                    #pragma omp atomic
                                    nc->S_RHS[kz * N2 + ky * N + kx] -= factor * Flux_S_R_comm * dg_nc[kx] * (2.0 / nc->dx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
// SolverDim<3> 3D Inviscid Sweep Y Implementation
// =========================================================================

void SolverDim<3>::sweep_y() {
    int N = p.N_PTS;
    int N2 = N * N;
    int N3 = N * N * N;

    // =========================================================================
    // Pass 1: Local & Conforming Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        for (int iz = 0; iz < N; ++iz) {
            for (int ix = 0; ix < N; ++ix) {

                // Pointwise Y-flux at each solution point
                double G_sol[MAX_PTS][5];
                double G_sol_S[MAX_PTS] = {};
                for (int iy = 0; iy < N; ++iy) {
                    get_flux_pointwise_cell(*c, iz, iy, ix,
                                            nullptr, G_sol[iy], nullptr,
                                            c->sigma_field[iz * N2 + iy * N + ix]);
                    if (p.ENABLE_PPR) {
                        double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iz, iy, ix, N));
                        double v   = c->get_U(2, iz, iy, ix, N) / rho;
                        G_sol_S[iy] = c->S_field[iz * N2 + iy * N + ix] * v;
                    }
                }

                // Face-extrapolated states
                double UB_face[5] = {}, UT_face[5] = {};
                double sig_B_face = 0.0, sig_T_face = 0.0;
                double S_B_face = 0.0, S_T_face = 0.0;
                for (int iy = 0; iy < N; ++iy) {
                    double s = c->sigma_field[iz * N2 + iy * N + ix];
                    sig_B_face += s * basis.l_L[iy];
                    sig_T_face += s * basis.l_R[iy];
                    if (p.ENABLE_PPR) {
                        S_B_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_L[iy];
                        S_T_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_R[iy];
                    }
                }
                for (int v = 0; v < 5; ++v) {
                    for (int iy = 0; iy < N; ++iy) {
                        UB_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iy];
                        UT_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iy];
                    }
                }

                // Common Riemann fluxes (Local, Conforming & Boundary)
                double Flux_B_local[5] = {}, Flux_T_local[5] = {};
                double Flux_S_B_comm = 0.0, Flux_S_T_comm = 0.0;
                double U_neigh[5];
                double sig_neigh;

                // Bottom Face (2)
                if (c->neighbors[2] && c->neighbors[2]->level == c->level) {
                    Cell3D* nc = c->neighbors[2];
                    char nface = c->neighbor_faces[2];
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int v = 0; v < 5; ++v) U_neigh[v] = 0.0;
                    for (int k = 0; k < N; ++k) {
                        for (int v = 0; v < 5; ++v)
                            U_neigh[v] += nc->get_U(v, iz, k, ix, N) * weights[k];
                        sig_neigh += nc->sigma_field[iz * N2 + k * N + ix] * weights[k];
                        if (p.ENABLE_PPR) {
                            S_neigh += nc->S_field[iz * N2 + k * N + ix] * weights[k];
                        }
                    }
                    compute_interface_flux(U_neigh, UB_face, sig_neigh, sig_B_face, S_neigh, S_B_face, nc->theta_avg, c->theta_avg, 1, Flux_B_local, Flux_S_B_comm);
                } else if (c->is_boundary[2]) {
                    double S_neigh = S_B_face;
                    get_neigh_state_cell(*c, iz * N + ix, false,
                                         UB_face, sig_B_face, U_neigh, sig_neigh, 1, S_B_face, &S_neigh);
                    compute_interface_flux(U_neigh, UB_face, sig_neigh, sig_B_face, S_neigh, S_B_face, c->theta_avg, c->theta_avg, 1, Flux_B_local, Flux_S_B_comm);
                }

                // Top Face (3)
                if (c->neighbors[3] && c->neighbors[3]->level == c->level) {
                    Cell3D* nc = c->neighbors[3];
                    char nface = c->neighbor_faces[3];
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int v = 0; v < 5; ++v) U_neigh[v] = 0.0;
                    for (int k = 0; k < N; ++k) {
                        for (int v = 0; v < 5; ++v)
                            U_neigh[v] += nc->get_U(v, iz, k, ix, N) * weights[k];
                        sig_neigh += nc->sigma_field[iz * N2 + k * N + ix] * weights[k];
                        if (p.ENABLE_PPR) {
                            S_neigh += nc->S_field[iz * N2 + k * N + ix] * weights[k];
                        }
                    }
                    compute_interface_flux(UT_face, U_neigh, sig_T_face, sig_neigh, S_T_face, S_neigh, c->theta_avg, nc->theta_avg, 1, Flux_T_local, Flux_S_T_comm);
                } else if (c->is_boundary[3]) {
                    double S_neigh = S_T_face;
                    get_neigh_state_cell(*c, iz * N + ix, true,
                                         UT_face, sig_T_face, U_neigh, sig_neigh, 1, S_T_face, &S_neigh);
                    compute_interface_flux(UT_face, U_neigh, sig_T_face, sig_neigh, S_T_face, S_neigh, c->theta_avg, c->theta_avg, 1, Flux_T_local, Flux_S_T_comm);
                }

                // Interior flux at faces
                double G_L[5] = {}, G_R[5] = {};
                double G_S_L = 0.0, G_S_R = 0.0;
                for (int v = 0; v < 5; ++v) {
                    for (int iy = 0; iy < N; ++iy) {
                        G_L[v] += G_sol[iy][v] * basis.l_L[iy];
                        G_R[v] += G_sol[iy][v] * basis.l_R[iy];
                    }
                }
                if (p.ENABLE_PPR) {
                    for (int iy = 0; iy < N; ++iy) {
                        G_S_L += G_sol_S[iy] * basis.l_L[iy];
                        G_S_R += G_sol_S[iy] * basis.l_R[iy];
                    }
                }

                // Accumulate into RHS
                for (int v = 0; v < 5; ++v) {
                    for (int iy = 0; iy < N; ++iy) {
                        double dg = 0.0;
                        for (int k = 0; k < N; ++k)
                            dg += basis.D[iy][k] * G_sol[k][v];
                        
                        c->get_RHS(v, iz, iy, ix, N) -= 
                            (dg
                             + (Flux_B_local[v] - G_L[v]) * basis.dgl[iy]
                             + (Flux_T_local[v] - G_R[v]) * basis.dgr[iy])
                            * (2.0 / c->dy);
                    }
                }
                if (p.ENABLE_PPR) {
                    for (int iy = 0; iy < N; ++iy) {
                        double dg_S = 0.0;
                        for (int k = 0; k < N; ++k)
                            dg_S += basis.D[iy][k] * G_sol_S[k];

                        c->S_RHS[iz * N2 + iy * N + ix] -=
                            (dg_S
                             + (Flux_S_B_comm - G_S_L) * basis.dgl[iy]
                             + (Flux_S_T_comm - G_S_R) * basis.dgr[iy])
                            * (2.0 / c->dy);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Interface Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // Bottom Face (2) non-conforming coarser neighbor
        if (c->neighbors[2] && c->neighbors[2]->level < c->level) {
            Cell3D* nc = c->neighbors[2];
            char nface = c->neighbor_faces[2];
            int child_x_idx = c->ex & 1;
            int child_z_idx = c->ez & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int ix = 0; ix < N; ++ix) {
                    double UB_face[5] = {};
                    double sig_B_face = 0.0;
                    double S_B_face = 0.0;
                    for (int iy = 0; iy < N; ++iy) {
                        double s = c->sigma_field[iz * N2 + iy * N + ix];
                        sig_B_face += s * basis.l_L[iy];
                        if (p.ENABLE_PPR) {
                            S_B_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_L[iy];
                        }
                        for (int v = 0; v < 5; ++v) {
                            UB_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iy];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double sig_coarse_face[MAX_PTS][MAX_PTS] = {};
                    double S_coarse_face[MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][kz] += nc->get_U(v, kz, ky, kx, N) * weights[ky];
                                }
                                sig_coarse_face[kx][kz] += nc->sigma_field[kz * N2 + ky * N + kx] * weights[ky];
                                if (p.ENABLE_PPR) {
                                    S_coarse_face[kx][kz] += nc->S_field[kz * N2 + ky * N + kx] * weights[ky];
                                }
                            }
                        }
                    }

                    double U_neigh[5] = {};
                    double sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                U_neigh[v] += factor * U_coarse_face[v][kx][kz];
                            }
                            sig_neigh += factor * sig_coarse_face[kx][kz];
                            if (p.ENABLE_PPR) {
                                S_neigh += factor * S_coarse_face[kx][kz];
                            }
                        }
                    }

                    double Flux_B_comm[5];
                    double Flux_S_B_comm = 0.0;
                    compute_interface_flux(U_neigh, UB_face, sig_neigh, sig_B_face, S_neigh, S_B_face, nc->theta_avg, c->theta_avg, 1, Flux_B_comm, Flux_S_B_comm);

                    // Update fine cell
                    for (int iy = 0; iy < N; ++iy) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) -= Flux_B_comm[v] * basis.dgl[iy] * (2.0 / c->dy);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            c->S_RHS[iz * N2 + iy * N + ix] -= Flux_S_B_comm * basis.dgl[iy] * (2.0 / c->dy);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RZ[kz][iz];
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) -= factor * Flux_B_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                                }
                                if (p.ENABLE_PPR) {
                                    #pragma omp atomic
                                    nc->S_RHS[kz * N2 + ky * N + kx] -= factor * Flux_S_B_comm * dg_nc[ky] * (2.0 / nc->dy);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Top Face (3) non-conforming coarser neighbor
        if (c->neighbors[3] && c->neighbors[3]->level < c->level) {
            Cell3D* nc = c->neighbors[3];
            char nface = c->neighbor_faces[3];
            int child_x_idx = c->ex & 1;
            int child_z_idx = c->ez & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int ix = 0; ix < N; ++ix) {
                    double UT_face[5] = {};
                    double sig_T_face = 0.0;
                    double S_T_face = 0.0;
                    for (int iy = 0; iy < N; ++iy) {
                        double s = c->sigma_field[iz * N2 + iy * N + ix];
                        sig_T_face += s * basis.l_R[iy];
                        if (p.ENABLE_PPR) {
                            S_T_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_R[iy];
                        }
                        for (int v = 0; v < 5; ++v) {
                            UT_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iy];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double sig_coarse_face[MAX_PTS][MAX_PTS] = {};
                    double S_coarse_face[MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][kz] += nc->get_U(v, kz, ky, kx, N) * weights[ky];
                                }
                                sig_coarse_face[kx][kz] += nc->sigma_field[kz * N2 + ky * N + kx] * weights[ky];
                                if (p.ENABLE_PPR) {
                                    S_coarse_face[kx][kz] += nc->S_field[kz * N2 + ky * N + kx] * weights[ky];
                                }
                            }
                        }
                    }

                    double U_neigh[5] = {};
                    double sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                U_neigh[v] += factor * U_coarse_face[v][kx][kz];
                            }
                            sig_neigh += factor * sig_coarse_face[kx][kz];
                            if (p.ENABLE_PPR) {
                                S_neigh += factor * S_coarse_face[kx][kz];
                            }
                        }
                    }

                    double Flux_T_comm[5];
                    double Flux_S_T_comm = 0.0;
                    compute_interface_flux(UT_face, U_neigh, sig_T_face, sig_neigh, S_T_face, S_neigh, c->theta_avg, nc->theta_avg, 1, Flux_T_comm, Flux_S_T_comm);

                    // Update fine cell
                    for (int iy = 0; iy < N; ++iy) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) -= Flux_T_comm[v] * basis.dgr[iy] * (2.0 / c->dy);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            c->S_RHS[iz * N2 + iy * N + ix] -= Flux_S_T_comm * basis.dgr[iy] * (2.0 / c->dy);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RZ[kz][iz];
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) -= factor * Flux_T_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                                }
                                if (p.ENABLE_PPR) {
                                    #pragma omp atomic
                                    nc->S_RHS[kz * N2 + ky * N + kx] -= factor * Flux_S_T_comm * dg_nc[ky] * (2.0 / nc->dy);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
void SolverDim<3>::sweep_z() {
    int N = p.N_PTS;
    int N2 = N * N;
    int N3 = N * N * N;

    // =========================================================================
    // Pass 1: Local & Conforming Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {

                // Pointwise Z-flux at each solution point
                double H_sol[MAX_PTS][5];
                double H_sol_S[MAX_PTS] = {};
                for (int iz = 0; iz < N; ++iz) {
                    get_flux_pointwise_cell(*c, iz, iy, ix,
                                            nullptr, nullptr, H_sol[iz],
                                            c->sigma_field[iz * N2 + iy * N + ix]);
                    if (p.ENABLE_PPR) {
                        double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iz, iy, ix, N));
                        double w   = c->get_U(3, iz, iy, ix, N) / rho;
                        H_sol_S[iz] = c->S_field[iz * N2 + iy * N + ix] * w;
                    }
                }

                // Face-extrapolated states
                double UF_face[5] = {}, UK_face[5] = {};
                double sig_F_face = 0.0, sig_K_face = 0.0;
                double S_F_face = 0.0, S_K_face = 0.0;
                for (int iz = 0; iz < N; ++iz) {
                    double s = c->sigma_field[iz * N2 + iy * N + ix];
                    sig_F_face += s * basis.l_L[iz];
                    sig_K_face += s * basis.l_R[iz];
                    if (p.ENABLE_PPR) {
                        S_F_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_L[iz];
                        S_K_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_R[iz];
                    }
                }
                for (int v = 0; v < 5; ++v) {
                    for (int iz = 0; iz < N; ++iz) {
                        UF_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iz];
                        UK_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iz];
                    }
                }

                // Common Riemann fluxes (Local, Conforming & Boundary)
                double Flux_F_local[5] = {}, Flux_K_local[5] = {};
                double Flux_S_F_comm = 0.0, Flux_S_K_comm = 0.0;
                double U_neigh[5];
                double sig_neigh;

                // Front Face (4)
                if (c->neighbors[4] && c->neighbors[4]->level == c->level) {
                    Cell3D* nc = c->neighbors[4];
                    char nface = c->neighbor_faces[4];
                    const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
                    sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int v = 0; v < 5; ++v) U_neigh[v] = 0.0;
                    for (int k = 0; k < N; ++k) {
                        for (int v = 0; v < 5; ++v)
                            U_neigh[v] += nc->get_U(v, k, iy, ix, N) * weights[k];
                        sig_neigh += nc->sigma_field[k * N2 + iy * N + ix] * weights[k];
                        if (p.ENABLE_PPR) {
                            S_neigh += nc->S_field[k * N2 + iy * N + ix] * weights[k];
                        }
                    }
                    compute_interface_flux(U_neigh, UF_face, sig_neigh, sig_F_face, S_neigh, S_F_face, nc->theta_avg, c->theta_avg, 2, Flux_F_local, Flux_S_F_comm);
                } else if (c->is_boundary[4]) {
                    double S_neigh = S_F_face;
                    get_neigh_state_cell(*c, iy * N + ix, false,
                                         UF_face, sig_F_face, U_neigh, sig_neigh, 2, S_F_face, &S_neigh);
                    compute_interface_flux(U_neigh, UF_face, sig_neigh, sig_F_face, S_neigh, S_F_face, c->theta_avg, c->theta_avg, 2, Flux_F_local, Flux_S_F_comm);
                }

                // Back Face (5)
                if (c->neighbors[5] && c->neighbors[5]->level == c->level) {
                    Cell3D* nc = c->neighbors[5];
                    char nface = c->neighbor_faces[5];
                    const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
                    sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int v = 0; v < 5; ++v) U_neigh[v] = 0.0;
                    for (int k = 0; k < N; ++k) {
                        for (int v = 0; v < 5; ++v)
                            U_neigh[v] += nc->get_U(v, k, iy, ix, N) * weights[k];
                        sig_neigh += nc->sigma_field[k * N2 + iy * N + ix] * weights[k];
                        if (p.ENABLE_PPR) {
                            S_neigh += nc->S_field[k * N2 + iy * N + ix] * weights[k];
                        }
                    }
                    compute_interface_flux(UK_face, U_neigh, sig_K_face, sig_neigh, S_K_face, S_neigh, c->theta_avg, nc->theta_avg, 2, Flux_K_local, Flux_S_K_comm);
                } else if (c->is_boundary[5]) {
                    double S_neigh = S_K_face;
                    get_neigh_state_cell(*c, iy * N + ix, true,
                                         UK_face, sig_K_face, U_neigh, sig_neigh, 2, S_K_face, &S_neigh);
                    compute_interface_flux(UK_face, U_neigh, sig_K_face, sig_neigh, S_K_face, S_neigh, c->theta_avg, c->theta_avg, 2, Flux_K_local, Flux_S_K_comm);
                }

                // Interior flux at faces
                double H_L[5] = {}, H_R[5] = {};
                double H_S_L = 0.0, H_S_R = 0.0;
                for (int v = 0; v < 5; ++v) {
                    for (int iz = 0; iz < N; ++iz) {
                        H_L[v] += H_sol[iz][v] * basis.l_L[iz];
                        H_R[v] += H_sol[iz][v] * basis.l_R[iz];
                    }
                }
                if (p.ENABLE_PPR) {
                    for (int iz = 0; iz < N; ++iz) {
                        H_S_L += H_sol_S[iz] * basis.l_L[iz];
                        H_S_R += H_sol_S[iz] * basis.l_R[iz];
                    }
                }

                // Accumulate into RHS
                for (int v = 0; v < 5; ++v) {
                    for (int iz = 0; iz < N; ++iz) {
                        double dh = 0.0;
                        for (int k = 0; k < N; ++k)
                            dh += basis.D[iz][k] * H_sol[k][v];
                        
                        c->get_RHS(v, iz, iy, ix, N) -= 
                            (dh
                             + (Flux_F_local[v] - H_L[v]) * basis.dgl[iz]
                             + (Flux_K_local[v] - H_R[v]) * basis.dgr[iz])
                            * (2.0 / c->dz);
                    }
                }
                if (p.ENABLE_PPR) {
                    for (int iz = 0; iz < N; ++iz) {
                        double dh_S = 0.0;
                        for (int k = 0; k < N; ++k)
                            dh_S += basis.D[iz][k] * H_sol_S[k];

                        c->S_RHS[iz * N2 + iy * N + ix] -=
                            (dh_S
                             + (Flux_S_F_comm - H_S_L) * basis.dgl[iz]
                             + (Flux_S_K_comm - H_S_R) * basis.dgr[iz])
                            * (2.0 / c->dz);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Interface Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // Front Face (4) non-conforming coarser neighbor
        if (c->neighbors[4] && c->neighbors[4]->level < c->level) {
            Cell3D* nc = c->neighbors[4];
            char nface = c->neighbor_faces[4];
            int child_x_idx = c->ex & 1;
            int child_y_idx = c->ey & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'F') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    double UF_face[5] = {};
                    double sig_F_face = 0.0;
                    double S_F_face = 0.0;
                    for (int iz = 0; iz < N; ++iz) {
                        double s = c->sigma_field[iz * N2 + iy * N + ix];
                        sig_F_face += s * basis.l_L[iz];
                        if (p.ENABLE_PPR) {
                            S_F_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_L[iz];
                        }
                        for (int v = 0; v < 5; ++v) {
                            UF_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iz];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double sig_coarse_face[MAX_PTS][MAX_PTS] = {};
                    double S_coarse_face[MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
                    for (int kx = 0; kx < N; ++kx) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][ky] += nc->get_U(v, kz, ky, kx, N) * weights[kz];
                                }
                                sig_coarse_face[kx][ky] += nc->sigma_field[kz * N2 + ky * N + kx] * weights[kz];
                                if (p.ENABLE_PPR) {
                                    S_coarse_face[kx][ky] += nc->S_field[kz * N2 + ky * N + kx] * weights[kz];
                                }
                            }
                        }
                    }

                    double U_neigh[5] = {};
                    double sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PY[ky][iy];
                            for (int v = 0; v < 5; ++v) {
                                U_neigh[v] += factor * U_coarse_face[v][kx][ky];
                            }
                            sig_neigh += factor * sig_coarse_face[kx][ky];
                            if (p.ENABLE_PPR) {
                                S_neigh += factor * S_coarse_face[kx][ky];
                            }
                        }
                    }

                    double Flux_F_comm[5];
                    double Flux_S_F_comm = 0.0;
                    compute_interface_flux(U_neigh, UF_face, sig_neigh, sig_F_face, S_neigh, S_F_face, nc->theta_avg, c->theta_avg, 2, Flux_F_comm, Flux_S_F_comm);

                    // Update fine cell
                    for (int iz = 0; iz < N; ++iz) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) -= Flux_F_comm[v] * basis.dgl[iz] * (2.0 / c->dz);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            c->S_RHS[iz * N2 + iy * N + ix] -= Flux_S_F_comm * basis.dgl[iz] * (2.0 / c->dz);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RY[ky][iy];
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) -= factor * Flux_F_comm[v] * dg_nc[kz] * (2.0 / nc->dz);
                                }
                                if (p.ENABLE_PPR) {
                                    #pragma omp atomic
                                    nc->S_RHS[kz * N2 + ky * N + kx] -= factor * Flux_S_F_comm * dg_nc[kz] * (2.0 / nc->dz);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Back Face (5) non-conforming coarser neighbor
        if (c->neighbors[5] && c->neighbors[5]->level < c->level) {
            Cell3D* nc = c->neighbors[5];
            char nface = c->neighbor_faces[5];
            int child_x_idx = c->ex & 1;
            int child_y_idx = c->ey & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'F') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    double UK_face[5] = {};
                    double sig_K_face = 0.0;
                    double S_K_face = 0.0;
                    for (int iz = 0; iz < N; ++iz) {
                        double s = c->sigma_field[iz * N2 + iy * N + ix];
                        sig_K_face += s * basis.l_R[iz];
                        if (p.ENABLE_PPR) {
                            S_K_face += c->S_field[iz * N2 + iy * N + ix] * basis.l_R[iz];
                        }
                        for (int v = 0; v < 5; ++v) {
                            UK_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iz];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double sig_coarse_face[MAX_PTS][MAX_PTS] = {};
                    double S_coarse_face[MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
                    for (int kx = 0; kx < N; ++kx) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][ky] += nc->get_U(v, kz, ky, kx, N) * weights[kz];
                                }
                                sig_coarse_face[kx][ky] += nc->sigma_field[kz * N2 + ky * N + kx] * weights[kz];
                                if (p.ENABLE_PPR) {
                                    S_coarse_face[kx][ky] += nc->S_field[kz * N2 + ky * N + kx] * weights[kz];
                                }
                            }
                        }
                    }

                    double U_neigh[5] = {};
                    double sig_neigh = 0.0;
                    double S_neigh = 0.0;
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PY[ky][iy];
                            for (int v = 0; v < 5; ++v) {
                                U_neigh[v] += factor * U_coarse_face[v][kx][ky];
                            }
                            sig_neigh += factor * sig_coarse_face[kx][ky];
                            if (p.ENABLE_PPR) {
                                S_neigh += factor * S_coarse_face[kx][ky];
                            }
                        }
                    }

                    double Flux_K_comm[5];
                    double Flux_S_K_comm = 0.0;
                    compute_interface_flux(UK_face, U_neigh, sig_K_face, sig_neigh, S_K_face, S_neigh, c->theta_avg, nc->theta_avg, 2, Flux_K_comm, Flux_S_K_comm);

                    // Update fine cell
                    for (int iz = 0; iz < N; ++iz) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) -= Flux_K_comm[v] * basis.dgr[iz] * (2.0 / c->dz);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            c->S_RHS[iz * N2 + iy * N + ix] -= Flux_S_K_comm * basis.dgr[iz] * (2.0 / c->dz);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RY[ky][iy];
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) -= factor * Flux_K_comm[v] * dg_nc[kz] * (2.0 / nc->dz);
                                }
                                if (p.ENABLE_PPR) {
                                    #pragma omp atomic
                                    nc->S_RHS[kz * N2 + ky * N + kx] -= factor * Flux_S_K_comm * dg_nc[kz] * (2.0 / nc->dz);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
// SolverDim<3> 3D Viscous Sweep X Implementation
// =========================================================================

static void compute_viscous_flux_x_3d(const double U[5],
                                       const double dUdx[5], const double dUdy[5], const double dUdz[5],
                                       double mu, double kappa, double gamma,
                                       double Fv[5], bool enable_suth, double suth_c, double pr, double pos_eps)
{
    double rho = std::max(pos_eps, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;
    double w = U[3] / rho;

    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dudz = (dUdz[1] - u * dUdz[0]) / rho;

    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;
    double dvdz = (dUdz[2] - v * dUdz[0]) / rho;

    double dwdx = (dUdx[3] - w * dUdx[0]) / rho;
    double dwdy = (dUdy[3] - w * dUdy[0]) / rho;
    double dwdz = (dUdz[3] - w * dUdz[0]) / rho;

    double p = (gamma - 1.0) * (U[4] - 0.5 * rho * (u*u + v*v + w*w));
    if (p < pos_eps) p = pos_eps;
    double T = p / rho;

    double mu_local = mu;
    double kappa_local = kappa;
    if (enable_suth) {
        double T_norm = std::max(1e-8, T);
        mu_local = mu * (std::pow(T_norm, 1.5) * (1.0 + suth_c) / (T_norm + suth_c));
        kappa_local = mu_local * gamma / ((gamma - 1.0) * pr);
    }

    double dpdx = (gamma - 1.0) * (dUdx[4] - 0.5*(u*u+v*v+w*w)*dUdx[0]
                                    - rho*(u*dudx + v*dvdx + w*dwdx));
    double dTdx = (dpdx - T * dUdx[0]) / rho;

    double tau_xx = mu_local * (4.0/3.0 * dudx - 2.0/3.0 * (dvdy + dwdz));
    double tau_xy = mu_local * (dudy + dvdx);
    double tau_xz = mu_local * (dudz + dwdx);

    double qx = -kappa_local * dTdx;

    Fv[0] = 0.0;
    Fv[1] = tau_xx;
    Fv[2] = tau_xy;
    Fv[3] = tau_xz;
    Fv[4] = u * tau_xx + v * tau_xy + w * tau_xz - qx;
}

void SolverDim<3>::viscous_sweep_x() {
    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);
    int N = p.N_PTS;
    int N2 = N * N;
    int N3 = N * N * N;

    // =========================================================================
    // Pass 1: Local & Conforming Pass
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        auto get_suth_mu = [&](const double State[5]) {
            double r = std::max(p.POS_LIMITER_EPS, State[0]);
            double press = (p.GAMMA - 1.0) * (State[4] - 0.5 * (State[1]*State[1] + State[2]*State[2] + State[3]*State[3]) / r);
            if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        auto extrapolate_neighbor_x = [&](const Cell3D* nc, char nface, int iz, int iy, double U_nb[5], double dUdx_nb[5], double dUdy_nb[5], double dUdz_nb[5]) {
            const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
            for (int v = 0; v < 5; ++v) {
                U_nb[v] = 0.0;
                dUdx_nb[v] = 0.0;
                dUdy_nb[v] = 0.0;
                dUdz_nb[v] = 0.0;
            }
            int nc_offset = nc->cell_index * 5 * N3;
            for (int k = 0; k < N; ++k) {
                int flat = iz * N2 + iy * N + k;
                for (int v = 0; v < 5; ++v) {
                    U_nb[v] += nc->get_U(v, iz, iy, k, N) * weights[k];
                    dUdx_nb[v] += global_grad_Ux[nc_offset + v * N3 + flat] * weights[k];
                    dUdy_nb[v] += global_grad_Uy[nc_offset + v * N3 + flat] * weights[k];
                    dUdz_nb[v] += global_grad_Uz[nc_offset + v * N3 + flat] * weights[k];
                }
            }
        };

        int c_offset = c->cell_index * 5 * N3;

        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {

                // --- 1. Pointwise viscous X-flux at each solution point ---
                double Fv_sol[MAX_PTS][5];
                for (int ix = 0; ix < N; ++ix) {
                    int flat = iz * N2 + iy * N + ix;
                    double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                    for (int v = 0; v < 5; ++v) {
                        U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                        dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                    }
                    compute_viscous_flux_x_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                             mu, kappa, p.GAMMA, Fv_sol[ix],
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                }

                // --- 2. Face-extrapolated viscous fluxes & states ---
                double Fv_L[5] = {}, Fv_R[5] = {};
                double UL_face[5] = {}, UR_face[5] = {};
                for (int v = 0; v < 5; ++v) {
                    for (int ix = 0; ix < N; ++ix) {
                        Fv_L[v] += Fv_sol[ix][v] * basis.l_L[ix];
                        Fv_R[v] += Fv_sol[ix][v] * basis.l_R[ix];
                        UL_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[ix];
                        UR_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[ix];
                    }
                }

                // --- 3. Left common viscous flux (conforming/boundary) ---
                double Fv_L_local[5] = {};
                if (c->neighbors[0] && c->neighbors[0]->level == c->level) {
                    Cell3D* nc = c->neighbors[0];
                    char nface = c->neighbor_faces[0];
                    double UL_nb[5], dUdx_nb[5], dUdy_nb[5], dUdz_nb[5], Fv_L_nb[5];
                    extrapolate_neighbor_x(nc, nface, iz, iy, UL_nb, dUdx_nb, dUdy_nb, dUdz_nb);
                    compute_viscous_flux_x_3d(UL_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Fv_L_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    double mu_L_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_L_face = 0.5 * (get_suth_mu(UL_face) + get_suth_mu(UL_nb));
                    }
                    double penalty_L = br2_eta * mu_L_face / c->dx;
                    for (int v = 0; v < 5; ++v) {
                        Fv_L_local[v] = 0.5 * (Fv_L_nb[v] + Fv_L[v])
                                      + penalty_L * (UL_face[v] - UL_nb[v]);
                    }
                } else if (c->is_boundary[0]) {
                    double UL_nb[5], sig_dummy;
                    get_neigh_state_cell(*c, iz * N + iy, false, UL_face, 0.0, UL_nb, sig_dummy, 0);
                    for (int v = 0; v < 5; ++v) {
                        Fv_L_local[v] = Fv_L[v];
                    }
                }

                // --- 4. Right common viscous flux (conforming/boundary) ---
                double Fv_R_local[5] = {};
                if (c->neighbors[1] && c->neighbors[1]->level == c->level) {
                    Cell3D* nc = c->neighbors[1];
                    char nface = c->neighbor_faces[1];
                    double UR_nb[5], dUdx_nb[5], dUdy_nb[5], dUdz_nb[5], Fv_R_nb[5];
                    extrapolate_neighbor_x(nc, nface, iz, iy, UR_nb, dUdx_nb, dUdy_nb, dUdz_nb);
                    compute_viscous_flux_x_3d(UR_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Fv_R_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    double mu_R_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_R_face = 0.5 * (get_suth_mu(UR_face) + get_suth_mu(UR_nb));
                    }
                    double penalty_R = br2_eta * mu_R_face / c->dx;
                    for (int v = 0; v < 5; ++v) {
                        Fv_R_local[v] = 0.5 * (Fv_R[v] + Fv_R_nb[v])
                                      + penalty_R * (UR_nb[v] - UR_face[v]);
                    }
                } else if (c->is_boundary[1]) {
                    double UR_nb[5], sig_dummy;
                    get_neigh_state_cell(*c, iz * N + iy, true, UR_face, 0.0, UR_nb, sig_dummy, 0);
                    for (int v = 0; v < 5; ++v) {
                        Fv_R_local[v] = Fv_R[v];
                    }
                }

                // --- 5. Accumulate into RHS ---
                for (int v = 0; v < 5; ++v) {
                    for (int ix = 0; ix < N; ++ix) {
                        double df = 0.0;
                        for (int kk = 0; kk < N; ++kk)
                            df += basis.D[ix][kk] * Fv_sol[kk][v];
                        
                        c->get_RHS(v, iz, iy, ix, N) += 
                            (df
                             + (Fv_L_local[v] - Fv_L[v]) * basis.dgl[ix]
                             + (Fv_R_local[v] - Fv_R[v]) * basis.dgr[ix])
                            * (2.0 / c->dx);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Restriction Pass
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        int c_offset = c->cell_index * 5 * N3;

        auto get_suth_mu = [&](const double State[5]) {
            double r = std::max(p.POS_LIMITER_EPS, State[0]);
            double press = (p.GAMMA - 1.0) * (State[4] - 0.5 * (State[1]*State[1] + State[2]*State[2] + State[3]*State[3]) / r);
            if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        // Left Face (0) non-conforming coarser neighbor
        if (c->neighbors[0] && c->neighbors[0]->level < c->level) {
            Cell3D* nc = c->neighbors[0];
            int nc_offset = nc->cell_index * 5 * N3;
            char nface = c->neighbor_faces[0];
            int child_y_idx = c->ey & 1;
            int child_z_idx = c->ez & 1;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int iy = 0; iy < N; ++iy) {
                    double UL_face[5] = {};
                    double Fv_L[5] = {};
                    double Fv_sol[MAX_PTS][5] = {};
                    for (int ix = 0; ix < N; ++ix) {
                        int flat = iz * N2 + iy * N + ix;
                        double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                        for (int v = 0; v < 5; ++v) {
                            U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                            dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                            dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                            dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                        }
                        compute_viscous_flux_x_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                                 mu, kappa, p.GAMMA, Fv_sol[ix],
                                                 p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    }
                    for (int v = 0; v < 5; ++v) {
                        for (int ix = 0; ix < N; ++ix) {
                            UL_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[ix];
                            Fv_L[v] += Fv_sol[ix][v] * basis.l_L[ix];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdx_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdy_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdz_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][ky][kz] += nc->get_U(v, kz, ky, kx, N) * weights[kx];
                                    dUdx_coarse_face[v][ky][kz] += global_grad_Ux[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kx];
                                    dUdy_coarse_face[v][ky][kz] += global_grad_Uy[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kx];
                                    dUdz_coarse_face[v][ky][kz] += global_grad_Uz[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kx];
                                }
                            }
                        }
                    }

                    double UL_nb[5] = {}, dUdx_nb[5] = {}, dUdy_nb[5] = {}, dUdz_nb[5] = {}, Fv_L_nb[5] = {};
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = PY[ky][iy] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                UL_nb[v] += factor * U_coarse_face[v][ky][kz];
                                dUdx_nb[v] += factor * dUdx_coarse_face[v][ky][kz];
                                dUdy_nb[v] += factor * dUdy_coarse_face[v][ky][kz];
                                dUdz_nb[v] += factor * dUdz_coarse_face[v][ky][kz];
                            }
                        }
                    }
                    compute_viscous_flux_x_3d(UL_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Fv_L_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);

                    double mu_L_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_L_face = 0.5 * (get_suth_mu(UL_face) + get_suth_mu(UL_nb));
                    }
                    double penalty_L = br2_eta * mu_L_face / c->dx;
                    double Fv_L_comm[5];
                    for (int v = 0; v < 5; ++v) {
                        Fv_L_comm[v] = 0.5 * (Fv_L[v] + Fv_L_nb[v])
                                     + penalty_L * (UL_face[v] - UL_nb[v]);
                    }

                    // Update fine cell local RHS
                    for (int ix = 0; ix < N; ++ix) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) += Fv_L_comm[v] * basis.dgl[ix] * (2.0 / c->dx);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor nc Right face
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = RY[ky][iy] * RZ[kz][iz];
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) += factor * Fv_L_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Right Face (1) non-conforming coarser neighbor
        if (c->neighbors[1] && c->neighbors[1]->level < c->level) {
            Cell3D* nc = c->neighbors[1];
            int nc_offset = nc->cell_index * 5 * N3;
            char nface = c->neighbor_faces[1];
            int child_y_idx = c->ey & 1;
            int child_z_idx = c->ez & 1;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int iy = 0; iy < N; ++iy) {
                    double UR_face[5] = {};
                    double Fv_R[5] = {};
                    double Fv_sol[MAX_PTS][5] = {};
                    for (int ix = 0; ix < N; ++ix) {
                        int flat = iz * N2 + iy * N + ix;
                        double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                        for (int v = 0; v < 5; ++v) {
                            U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                            dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                            dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                            dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                        }
                        compute_viscous_flux_x_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                                 mu, kappa, p.GAMMA, Fv_sol[ix],
                                                 p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    }
                    for (int v = 0; v < 5; ++v) {
                        for (int ix = 0; ix < N; ++ix) {
                            UR_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[ix];
                            Fv_R[v] += Fv_sol[ix][v] * basis.l_R[ix];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdx_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdy_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdz_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][ky][kz] += nc->get_U(v, kz, ky, kx, N) * weights[kx];
                                    dUdx_coarse_face[v][ky][kz] += global_grad_Ux[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kx];
                                    dUdy_coarse_face[v][ky][kz] += global_grad_Uy[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kx];
                                    dUdz_coarse_face[v][ky][kz] += global_grad_Uz[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kx];
                                }
                            }
                        }
                    }

                    double UR_nb[5] = {}, dUdx_nb[5] = {}, dUdy_nb[5] = {}, dUdz_nb[5] = {}, Fv_R_nb[5] = {};
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = PY[ky][iy] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                UR_nb[v] += factor * U_coarse_face[v][ky][kz];
                                dUdx_nb[v] += factor * dUdx_coarse_face[v][ky][kz];
                                dUdy_nb[v] += factor * dUdy_coarse_face[v][ky][kz];
                                dUdz_nb[v] += factor * dUdz_coarse_face[v][ky][kz];
                            }
                        }
                    }
                    compute_viscous_flux_x_3d(UR_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Fv_R_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);

                    double mu_R_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_R_face = 0.5 * (get_suth_mu(UR_face) + get_suth_mu(UR_nb));
                    }
                    double penalty_R = br2_eta * mu_R_face / c->dx;
                    double Fv_R_comm[5];
                    for (int v = 0; v < 5; ++v) {
                        Fv_R_comm[v] = 0.5 * (Fv_R[v] + Fv_R_nb[v])
                                     + penalty_R * (UR_nb[v] - UR_face[v]);
                    }

                    // Update fine cell local RHS
                    for (int ix = 0; ix < N; ++ix) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) += Fv_R_comm[v] * basis.dgr[ix] * (2.0 / c->dx);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor nc Left face
                    for (int kz = 0; kz < N; ++kz) {
                        for (int ky = 0; ky < N; ++ky) {
                            double factor = RY[ky][iy] * RZ[kz][iz];
                            for (int kx = 0; kx < N; ++kx) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) += factor * Fv_R_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
// SolverDim<3> 3D Viscous Sweep Y Implementation
// =========================================================================

static void compute_viscous_flux_y_3d(const double U[5],
                                       const double dUdx[5], const double dUdy[5], const double dUdz[5],
                                       double mu, double kappa, double gamma,
                                       double Gv[5], bool enable_suth, double suth_c, double pr, double pos_eps)
{
    double rho = std::max(pos_eps, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;
    double w = U[3] / rho;

    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dudz = (dUdz[1] - u * dUdz[0]) / rho;

    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;
    double dvdz = (dUdz[2] - v * dUdz[0]) / rho;

    double dwdx = (dUdx[3] - w * dUdx[0]) / rho;
    double dwdy = (dUdy[3] - w * dUdy[0]) / rho;
    double dwdz = (dUdz[3] - w * dUdz[0]) / rho;

    double p = (gamma - 1.0) * (U[4] - 0.5 * rho * (u*u + v*v + w*w));
    if (p < pos_eps) p = pos_eps;
    double T = p / rho;

    double mu_local = mu;
    double kappa_local = kappa;
    if (enable_suth) {
        double T_norm = std::max(1e-8, T);
        mu_local = mu * (std::pow(T_norm, 1.5) * (1.0 + suth_c) / (T_norm + suth_c));
        kappa_local = mu_local * gamma / ((gamma - 1.0) * pr);
    }

    double dpdy = (gamma - 1.0) * (dUdy[4] - 0.5*(u*u+v*v+w*w)*dUdy[0]
                                    - rho*(u*dudy + v*dvdy + w*dwdy));
    double dTdy = (dpdy - T * dUdy[0]) / rho;

    double tau_yy = mu_local * (4.0/3.0 * dvdy - 2.0/3.0 * (dudx + dwdz));
    double tau_yx = mu_local * (dudy + dvdx);
    double tau_yz = mu_local * (dvdz + dwdy);

    double qy = -kappa_local * dTdy;

    Gv[0] = 0.0;
    Gv[1] = tau_yx;
    Gv[2] = tau_yy;
    Gv[3] = tau_yz;
    Gv[4] = u * tau_yx + v * tau_yy + w * tau_yz - qy;
}

void SolverDim<3>::viscous_sweep_y() {
    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);
    int N = p.N_PTS;
    int N2 = N * N;
    int N3 = N * N * N;

    // =========================================================================
    // Pass 1: Local & Conforming Pass
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        auto get_suth_mu = [&](const double State[5]) {
            double r = std::max(p.POS_LIMITER_EPS, State[0]);
            double press = (p.GAMMA - 1.0) * (State[4] - 0.5 * (State[1]*State[1] + State[2]*State[2] + State[3]*State[3]) / r);
            if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        auto extrapolate_neighbor_y = [&](const Cell3D* nc, char nface, int iz, int ix, double U_nb[5], double dUdx_nb[5], double dUdy_nb[5], double dUdz_nb[5]) {
            const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
            for (int v = 0; v < 5; ++v) {
                U_nb[v] = 0.0;
                dUdx_nb[v] = 0.0;
                dUdy_nb[v] = 0.0;
                dUdz_nb[v] = 0.0;
            }
            int nc_offset = nc->cell_index * 5 * N3;
            for (int k = 0; k < N; ++k) {
                int flat = iz * N2 + k * N + ix;
                for (int v = 0; v < 5; ++v) {
                    U_nb[v] += nc->get_U(v, iz, k, ix, N) * weights[k];
                    dUdx_nb[v] += global_grad_Ux[nc_offset + v * N3 + flat] * weights[k];
                    dUdy_nb[v] += global_grad_Uy[nc_offset + v * N3 + flat] * weights[k];
                    dUdz_nb[v] += global_grad_Uz[nc_offset + v * N3 + flat] * weights[k];
                }
            }
        };

        int c_offset = c->cell_index * 5 * N3;

        for (int iz = 0; iz < N; ++iz) {
            for (int ix = 0; ix < N; ++ix) {

                // --- 1. Pointwise viscous Y-flux at each solution point ---
                double Gv_sol[MAX_PTS][5];
                for (int iy = 0; iy < N; ++iy) {
                    int flat = iz * N2 + iy * N + ix;
                    double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                    for (int v = 0; v < 5; ++v) {
                        U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                        dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                    }
                    compute_viscous_flux_y_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                             mu, kappa, p.GAMMA, Gv_sol[iy],
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                }

                // --- 2. Face-extrapolated viscous fluxes & states ---
                double Gv_L[5] = {}, Gv_R[5] = {};
                double UB_face[5] = {}, UT_face[5] = {};
                for (int v = 0; v < 5; ++v) {
                    for (int iy = 0; iy < N; ++iy) {
                        Gv_L[v] += Gv_sol[iy][v] * basis.l_L[iy];
                        Gv_R[v] += Gv_sol[iy][v] * basis.l_R[iy];
                        UB_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iy];
                        UT_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iy];
                    }
                }

                // --- 3. Bottom common viscous flux (conforming/boundary) ---
                double Gv_L_local[5] = {};
                if (c->neighbors[2] && c->neighbors[2]->level == c->level) {
                    Cell3D* nc = c->neighbors[2];
                    char nface = c->neighbor_faces[2];
                    double UB_nb[5], dUdx_nb[5], dUdy_nb[5], dUdz_nb[5], Gv_L_nb[5];
                    extrapolate_neighbor_y(nc, nface, iz, ix, UB_nb, dUdx_nb, dUdy_nb, dUdz_nb);
                    compute_viscous_flux_y_3d(UB_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Gv_L_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    double mu_L_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_L_face = 0.5 * (get_suth_mu(UB_face) + get_suth_mu(UB_nb));
                    }
                    double penalty_L = br2_eta * mu_L_face / c->dy;
                    for (int v = 0; v < 5; ++v) {
                        Gv_L_local[v] = 0.5 * (Gv_L_nb[v] + Gv_L[v])
                                      + penalty_L * (UB_face[v] - UB_nb[v]);
                    }
                } else if (c->is_boundary[2]) {
                    double UB_nb[5], sig_dummy;
                    get_neigh_state_cell(*c, iz * N + ix, false, UB_face, 0.0, UB_nb, sig_dummy, 1);
                    for (int v = 0; v < 5; ++v) {
                        Gv_L_local[v] = Gv_L[v];
                    }
                }

                // --- 4. Top common viscous flux (conforming/boundary) ---
                double Gv_R_local[5] = {};
                if (c->neighbors[3] && c->neighbors[3]->level == c->level) {
                    Cell3D* nc = c->neighbors[3];
                    char nface = c->neighbor_faces[3];
                    double UT_nb[5], dUdx_nb[5], dUdy_nb[5], dUdz_nb[5], Gv_R_nb[5];
                    extrapolate_neighbor_y(nc, nface, iz, ix, UT_nb, dUdx_nb, dUdy_nb, dUdz_nb);
                    compute_viscous_flux_y_3d(UT_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Gv_R_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    double mu_R_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_R_face = 0.5 * (get_suth_mu(UT_face) + get_suth_mu(UT_nb));
                    }
                    double penalty_R = br2_eta * mu_R_face / c->dy;
                    for (int v = 0; v < 5; ++v) {
                        Gv_R_local[v] = 0.5 * (Gv_R[v] + Gv_R_nb[v])
                                      + penalty_R * (UT_nb[v] - UT_face[v]);
                    }
                } else if (c->is_boundary[3]) {
                    double UT_nb[5], sig_dummy;
                    get_neigh_state_cell(*c, iz * N + ix, true, UT_face, 0.0, UT_nb, sig_dummy, 1);
                    for (int v = 0; v < 5; ++v) {
                        Gv_R_local[v] = Gv_R[v];
                    }
                }

                // --- 5. Accumulate into RHS ---
                for (int v = 0; v < 5; ++v) {
                    for (int iy = 0; iy < N; ++iy) {
                        double dg = 0.0;
                        for (int kk = 0; kk < N; ++kk)
                            dg += basis.D[iy][kk] * Gv_sol[kk][v];
                        
                        c->get_RHS(v, iz, iy, ix, N) += 
                            (dg
                             + (Gv_L_local[v] - Gv_L[v]) * basis.dgl[iy]
                             + (Gv_R_local[v] - Gv_R[v]) * basis.dgr[iy])
                            * (2.0 / c->dy);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Restriction Pass
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        int c_offset = c->cell_index * 5 * N3;

        auto get_suth_mu = [&](const double State[5]) {
            double r = std::max(p.POS_LIMITER_EPS, State[0]);
            double press = (p.GAMMA - 1.0) * (State[4] - 0.5 * (State[1]*State[1] + State[2]*State[2] + State[3]*State[3]) / r);
            if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        // Bottom Face (2) non-conforming coarser neighbor
        if (c->neighbors[2] && c->neighbors[2]->level < c->level) {
            Cell3D* nc = c->neighbors[2];
            int nc_offset = nc->cell_index * 5 * N3;
            char nface = c->neighbor_faces[2];
            int child_x_idx = c->ex & 1;
            int child_z_idx = c->ez & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int ix = 0; ix < N; ++ix) {
                    double UB_face[5] = {};
                    double Gv_L[5] = {};
                    double Gv_sol[MAX_PTS][5] = {};
                    for (int iy = 0; iy < N; ++iy) {
                        int flat = iz * N2 + iy * N + ix;
                        double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                        for (int v = 0; v < 5; ++v) {
                            U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                            dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                            dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                            dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                        }
                        compute_viscous_flux_y_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                                 mu, kappa, p.GAMMA, Gv_sol[iy],
                                                 p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    }
                    for (int v = 0; v < 5; ++v) {
                        for (int iy = 0; iy < N; ++iy) {
                            UB_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iy];
                            Gv_L[v] += Gv_sol[iy][v] * basis.l_L[iy];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdx_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdy_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdz_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][kz] += nc->get_U(v, kz, ky, kx, N) * weights[ky];
                                    dUdx_coarse_face[v][kx][kz] += global_grad_Ux[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[ky];
                                    dUdy_coarse_face[v][kx][kz] += global_grad_Uy[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[ky];
                                    dUdz_coarse_face[v][kx][kz] += global_grad_Uz[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[ky];
                                }
                            }
                        }
                    }

                    double UB_nb[5] = {}, dUdx_nb[5] = {}, dUdy_nb[5] = {}, dUdz_nb[5] = {}, Gv_L_nb[5] = {};
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                UB_nb[v] += factor * U_coarse_face[v][kx][kz];
                                dUdx_nb[v] += factor * dUdx_coarse_face[v][kx][kz];
                                dUdy_nb[v] += factor * dUdy_coarse_face[v][kx][kz];
                                dUdz_nb[v] += factor * dUdz_coarse_face[v][kx][kz];
                            }
                        }
                    }
                    compute_viscous_flux_y_3d(UB_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Gv_L_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);

                    double mu_L_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_L_face = 0.5 * (get_suth_mu(UB_face) + get_suth_mu(UB_nb));
                    }
                    double penalty_L = br2_eta * mu_L_face / c->dy;
                    double Gv_L_comm[5];
                    for (int v = 0; v < 5; ++v) {
                        Gv_L_comm[v] = 0.5 * (Gv_L[v] + Gv_L_nb[v])
                                     + penalty_L * (UB_face[v] - UB_nb[v]);
                    }

                    // Update fine cell local RHS
                    for (int iy = 0; iy < N; ++iy) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) += Gv_L_comm[v] * basis.dgl[iy] * (2.0 / c->dy);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor nc Top face
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RZ[kz][iz];
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) += factor * Gv_L_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Top Face (3) non-conforming coarser neighbor
        if (c->neighbors[3] && c->neighbors[3]->level < c->level) {
            Cell3D* nc = c->neighbors[3];
            int nc_offset = nc->cell_index * 5 * N3;
            char nface = c->neighbor_faces[3];
            int child_x_idx = c->ex & 1;
            int child_z_idx = c->ez & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PZ = (child_z_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RZ = (child_z_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int iz = 0; iz < N; ++iz) {
                for (int ix = 0; ix < N; ++ix) {
                    double UT_face[5] = {};
                    double Gv_R[5] = {};
                    double Gv_sol[MAX_PTS][5] = {};
                    for (int iy = 0; iy < N; ++iy) {
                        int flat = iz * N2 + iy * N + ix;
                        double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                        for (int v = 0; v < 5; ++v) {
                            U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                            dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                            dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                            dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                        }
                        compute_viscous_flux_y_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                                 mu, kappa, p.GAMMA, Gv_sol[iy],
                                                 p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    }
                    for (int v = 0; v < 5; ++v) {
                        for (int iy = 0; iy < N; ++iy) {
                            UT_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iy];
                            Gv_R[v] += Gv_sol[iy][v] * basis.l_R[iy];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdx_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdy_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdz_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][kz] += nc->get_U(v, kz, ky, kx, N) * weights[ky];
                                    dUdx_coarse_face[v][kx][kz] += global_grad_Ux[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[ky];
                                    dUdy_coarse_face[v][kx][kz] += global_grad_Uy[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[ky];
                                    dUdz_coarse_face[v][kx][kz] += global_grad_Uz[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[ky];
                                }
                            }
                        }
                    }

                    double UT_nb[5] = {}, dUdx_nb[5] = {}, dUdy_nb[5] = {}, dUdz_nb[5] = {}, Gv_R_nb[5] = {};
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PZ[kz][iz];
                            for (int v = 0; v < 5; ++v) {
                                UT_nb[v] += factor * U_coarse_face[v][kx][kz];
                                dUdx_nb[v] += factor * dUdx_coarse_face[v][kx][kz];
                                dUdy_nb[v] += factor * dUdy_coarse_face[v][kx][kz];
                                dUdz_nb[v] += factor * dUdz_coarse_face[v][kx][kz];
                            }
                        }
                    }
                    compute_viscous_flux_y_3d(UT_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Gv_R_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);

                    double mu_R_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_R_face = 0.5 * (get_suth_mu(UT_face) + get_suth_mu(UT_nb));
                    }
                    double penalty_R = br2_eta * mu_R_face / c->dy;
                    double Gv_R_comm[5];
                    for (int v = 0; v < 5; ++v) {
                        Gv_R_comm[v] = 0.5 * (Gv_R[v] + Gv_R_nb[v])
                                     + penalty_R * (UT_nb[v] - UT_face[v]);
                    }

                    // Update fine cell local RHS
                    for (int iy = 0; iy < N; ++iy) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) += Gv_R_comm[v] * basis.dgr[iy] * (2.0 / c->dy);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor nc Bottom face
                    for (int kz = 0; kz < N; ++kz) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RZ[kz][iz];
                            for (int ky = 0; ky < N; ++ky) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) += factor * Gv_R_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
static void compute_viscous_flux_z_3d(const double U[5],
                                       const double dUdx[5], const double dUdy[5], const double dUdz[5],
                                       double mu, double kappa, double gamma,
                                       double Hv[5], bool enable_suth, double suth_c, double pr, double pos_eps)
{
    double rho = std::max(pos_eps, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;
    double w = U[3] / rho;

    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dudz = (dUdz[1] - u * dUdz[0]) / rho;

    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;
    double dvdz = (dUdz[2] - v * dUdz[0]) / rho;

    double dwdx = (dUdx[3] - w * dUdx[0]) / rho;
    double dwdy = (dUdy[3] - w * dUdy[0]) / rho;
    double dwdz = (dUdz[3] - w * dUdz[0]) / rho;

    double p = (gamma - 1.0) * (U[4] - 0.5 * rho * (u*u + v*v + w*w));
    if (p < pos_eps) p = pos_eps;
    double T = p / rho;

    double mu_local = mu;
    double kappa_local = kappa;
    if (enable_suth) {
        double T_norm = std::max(1e-8, T);
        mu_local = mu * (std::pow(T_norm, 1.5) * (1.0 + suth_c) / (T_norm + suth_c));
        kappa_local = mu_local * gamma / ((gamma - 1.0) * pr);
    }

    double dpdz = (gamma - 1.0) * (dUdz[4] - 0.5*(u*u+v*v+w*w)*dUdz[0]
                                    - rho*(u*dudz + v*dvdz + w*dwdz));
    double dTdz = (dpdz - T * dUdz[0]) / rho;

    double tau_zz = mu_local * (4.0/3.0 * dwdz - 2.0/3.0 * (dudx + dvdy));
    double tau_zx = mu_local * (dudz + dwdx);
    double tau_zy = mu_local * (dvdz + dwdy);

    double qz = -kappa_local * dTdz;

    Hv[0] = 0.0;
    Hv[1] = tau_zx;
    Hv[2] = tau_zy;
    Hv[3] = tau_zz;
    Hv[4] = u * tau_zx + v * tau_zy + w * tau_zz - qz;
}

void SolverDim<3>::viscous_sweep_z() {
    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);
    int N = p.N_PTS;
    int N2 = N * N;
    int N3 = N * N * N;

    // =========================================================================
    // Pass 1: Local & Conforming Pass
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        auto get_suth_mu = [&](const double State[5]) {
            double r = std::max(p.POS_LIMITER_EPS, State[0]);
            double press = (p.GAMMA - 1.0) * (State[4] - 0.5 * (State[1]*State[1] + State[2]*State[2] + State[3]*State[3]) / r);
            if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        auto extrapolate_neighbor_z = [&](const Cell3D* nc, char nface, int iy, int ix, double U_nb[5], double dUdx_nb[5], double dUdy_nb[5], double dUdz_nb[5]) {
            const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
            for (int v = 0; v < 5; ++v) {
                U_nb[v] = 0.0;
                dUdx_nb[v] = 0.0;
                dUdy_nb[v] = 0.0;
                dUdz_nb[v] = 0.0;
            }
            int nc_offset = nc->cell_index * 5 * N3;
            for (int k = 0; k < N; ++k) {
                int flat = k * N2 + iy * N + ix;
                for (int v = 0; v < 5; ++v) {
                    U_nb[v] += nc->get_U(v, k, iy, ix, N) * weights[k];
                    dUdx_nb[v] += global_grad_Ux[nc_offset + v * N3 + flat] * weights[k];
                    dUdy_nb[v] += global_grad_Uy[nc_offset + v * N3 + flat] * weights[k];
                    dUdz_nb[v] += global_grad_Uz[nc_offset + v * N3 + flat] * weights[k];
                }
            }
        };

        int c_offset = c->cell_index * 5 * N3;

        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {

                // --- 1. Pointwise viscous Z-flux at each solution point ---
                double Hv_sol[MAX_PTS][5];
                for (int iz = 0; iz < N; ++iz) {
                    int flat = iz * N2 + iy * N + ix;
                    double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                    for (int v = 0; v < 5; ++v) {
                        U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                        dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                    }
                    compute_viscous_flux_z_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                             mu, kappa, p.GAMMA, Hv_sol[iz],
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                }

                // --- 2. Face-extrapolated viscous fluxes & states ---
                double Hv_L[5] = {}, Hv_R[5] = {};
                double UF_face[5] = {}, UK_face[5] = {};
                for (int v = 0; v < 5; ++v) {
                    for (int iz = 0; iz < N; ++iz) {
                        Hv_L[v] += Hv_sol[iz][v] * basis.l_L[iz];
                        Hv_R[v] += Hv_sol[iz][v] * basis.l_R[iz];
                        UF_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iz];
                        UK_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iz];
                    }
                }

                // --- 3. Front common viscous flux (conforming/boundary) ---
                double Hv_L_local[5] = {};
                if (c->neighbors[4] && c->neighbors[4]->level == c->level) {
                    Cell3D* nc = c->neighbors[4];
                    char nface = c->neighbor_faces[4];
                    double UF_nb[5], dUdx_nb[5], dUdy_nb[5], dUdz_nb[5], Hv_L_nb[5];
                    extrapolate_neighbor_z(nc, nface, iy, ix, UF_nb, dUdx_nb, dUdy_nb, dUdz_nb);
                    compute_viscous_flux_z_3d(UF_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Hv_L_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    double mu_L_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_L_face = 0.5 * (get_suth_mu(UF_face) + get_suth_mu(UF_nb));
                    }
                    double penalty_L = br2_eta * mu_L_face / c->dz;
                    for (int v = 0; v < 5; ++v) {
                        Hv_L_local[v] = 0.5 * (Hv_L_nb[v] + Hv_L[v])
                                      + penalty_L * (UF_face[v] - UF_nb[v]);
                    }
                } else if (c->is_boundary[4]) {
                    double UF_nb[5], sig_dummy;
                    get_neigh_state_cell(*c, iy * N + ix, false, UF_face, 0.0, UF_nb, sig_dummy, 2);
                    for (int v = 0; v < 5; ++v) {
                        Hv_L_local[v] = Hv_L[v];
                    }
                }

                // --- 4. Back common viscous flux (conforming/boundary) ---
                double Hv_R_local[5] = {};
                if (c->neighbors[5] && c->neighbors[5]->level == c->level) {
                    Cell3D* nc = c->neighbors[5];
                    char nface = c->neighbor_faces[5];
                    double UK_nb[5], dUdx_nb[5], dUdy_nb[5], dUdz_nb[5], Hv_R_nb[5];
                    extrapolate_neighbor_z(nc, nface, iy, ix, UK_nb, dUdx_nb, dUdy_nb, dUdz_nb);
                    compute_viscous_flux_z_3d(UK_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Hv_R_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    double mu_R_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_R_face = 0.5 * (get_suth_mu(UK_face) + get_suth_mu(UK_nb));
                    }
                    double penalty_R = br2_eta * mu_R_face / c->dz;
                    for (int v = 0; v < 5; ++v) {
                        Hv_R_local[v] = 0.5 * (Hv_R[v] + Hv_R_nb[v])
                                      + penalty_R * (UK_nb[v] - UK_face[v]);
                    }
                } else if (c->is_boundary[5]) {
                    double UK_nb[5], sig_dummy;
                    get_neigh_state_cell(*c, iy * N + ix, true, UK_face, 0.0, UK_nb, sig_dummy, 2);
                    for (int v = 0; v < 5; ++v) {
                        Hv_R_local[v] = Hv_R[v];
                    }
                }

                // --- 5. Accumulate into RHS ---
                for (int v = 0; v < 5; ++v) {
                    for (int iz = 0; iz < N; ++iz) {
                        double dh = 0.0;
                        for (int kk = 0; kk < N; ++kk)
                            dh += basis.D[iz][kk] * Hv_sol[kk][v];
                        
                        c->get_RHS(v, iz, iy, ix, N) += 
                            (dh
                             + (Hv_L_local[v] - Hv_L[v]) * basis.dgl[iz]
                             + (Hv_R_local[v] - Hv_R[v]) * basis.dgr[iz])
                            * (2.0 / c->dz);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Restriction Pass
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        int c_offset = c->cell_index * 5 * N3;

        auto get_suth_mu = [&](const double State[5]) {
            double r = std::max(p.POS_LIMITER_EPS, State[0]);
            double press = (p.GAMMA - 1.0) * (State[4] - 0.5 * (State[1]*State[1] + State[2]*State[2] + State[3]*State[3]) / r);
            if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        // Front Face (4) non-conforming coarser neighbor
        if (c->neighbors[4] && c->neighbors[4]->level < c->level) {
            Cell3D* nc = c->neighbors[4];
            int nc_offset = nc->cell_index * 5 * N3;
            char nface = c->neighbor_faces[4];
            int child_x_idx = c->ex & 1;
            int child_y_idx = c->ey & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'F') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    double UF_face[5] = {};
                    double Hv_L[5] = {};
                    double Hv_sol[MAX_PTS][5] = {};
                    for (int iz = 0; iz < N; ++iz) {
                        int flat = iz * N2 + iy * N + ix;
                        double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                        for (int v = 0; v < 5; ++v) {
                            U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                            dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                            dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                            dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                        }
                        compute_viscous_flux_z_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                                 mu, kappa, p.GAMMA, Hv_sol[iz],
                                                 p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    }
                    for (int v = 0; v < 5; ++v) {
                        for (int iz = 0; iz < N; ++iz) {
                            UF_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_L[iz];
                            Hv_L[v] += Hv_sol[iz][v] * basis.l_L[iz];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdx_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdy_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdz_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
                    for (int kx = 0; kx < N; ++kx) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][ky] += nc->get_U(v, kz, ky, kx, N) * weights[kz];
                                    dUdx_coarse_face[v][kx][ky] += global_grad_Ux[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kz];
                                    dUdy_coarse_face[v][kx][ky] += global_grad_Uy[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kz];
                                    dUdz_coarse_face[v][kx][ky] += global_grad_Uz[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kz];
                                }
                            }
                        }
                    }

                    double UF_nb[5] = {}, dUdx_nb[5] = {}, dUdy_nb[5] = {}, dUdz_nb[5] = {}, Hv_L_nb[5] = {};
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PY[ky][iy];
                            for (int v = 0; v < 5; ++v) {
                                UF_nb[v] += factor * U_coarse_face[v][kx][ky];
                                dUdx_nb[v] += factor * dUdx_coarse_face[v][kx][ky];
                                dUdy_nb[v] += factor * dUdy_coarse_face[v][kx][ky];
                                dUdz_nb[v] += factor * dUdz_coarse_face[v][kx][ky];
                            }
                        }
                    }
                    compute_viscous_flux_z_3d(UF_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Hv_L_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);

                    double mu_L_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_L_face = 0.5 * (get_suth_mu(UF_face) + get_suth_mu(UF_nb));
                    }
                    double penalty_L = br2_eta * mu_L_face / c->dz;
                    double Hv_L_comm[5];
                    for (int v = 0; v < 5; ++v) {
                        Hv_L_comm[v] = 0.5 * (Hv_L[v] + Hv_L_nb[v])
                                     + penalty_L * (UF_face[v] - UF_nb[v]);
                    }

                    // Update fine cell local RHS
                    for (int iz = 0; iz < N; ++iz) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) += Hv_L_comm[v] * basis.dgl[iz] * (2.0 / c->dz);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor nc Back face
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RY[ky][iy];
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) += factor * Hv_L_comm[v] * dg_nc[kz] * (2.0 / nc->dz);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Back Face (5) non-conforming coarser neighbor
        if (c->neighbors[5] && c->neighbors[5]->level < c->level) {
            Cell3D* nc = c->neighbors[5];
            int nc_offset = nc->cell_index * 5 * N3;
            char nface = c->neighbor_faces[5];
            int child_x_idx = c->ex & 1;
            int child_y_idx = c->ey & 1;
            const auto& PX = (child_x_idx == 0) ? basis.P1 : basis.P2;
            const auto& PY = (child_y_idx == 0) ? basis.P1 : basis.P2;
            const auto& RX = (child_x_idx == 0) ? basis.R1 : basis.R2;
            const auto& RY = (child_y_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'F') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    double UK_face[5] = {};
                    double Hv_R[5] = {};
                    double Hv_sol[MAX_PTS][5] = {};
                    for (int iz = 0; iz < N; ++iz) {
                        int flat = iz * N2 + iy * N + ix;
                        double U_pt[5], dUdx_pt[5], dUdy_pt[5], dUdz_pt[5];
                        for (int v = 0; v < 5; ++v) {
                            U_pt[v]    = c->get_U(v, iz, iy, ix, N);
                            dUdx_pt[v] = global_grad_Ux[c_offset + v * N3 + flat];
                            dUdy_pt[v] = global_grad_Uy[c_offset + v * N3 + flat];
                            dUdz_pt[v] = global_grad_Uz[c_offset + v * N3 + flat];
                        }
                        compute_viscous_flux_z_3d(U_pt, dUdx_pt, dUdy_pt, dUdz_pt,
                                                 mu, kappa, p.GAMMA, Hv_sol[iz],
                                                 p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);
                    }
                    for (int v = 0; v < 5; ++v) {
                        for (int iz = 0; iz < N; ++iz) {
                            UK_face[v] += c->get_U(v, iz, iy, ix, N) * basis.l_R[iz];
                            Hv_R[v] += Hv_sol[iz][v] * basis.l_R[iz];
                        }
                    }

                    double U_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdx_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdy_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    double dUdz_coarse_face[5][MAX_PTS][MAX_PTS] = {};
                    const double* weights = (nface == 'F') ? basis.l_L.data() : basis.l_R.data();
                    for (int kx = 0; kx < N; ++kx) {
                        for (int ky = 0; ky < N; ++ky) {
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    U_coarse_face[v][kx][ky] += nc->get_U(v, kz, ky, kx, N) * weights[kz];
                                    dUdx_coarse_face[v][kx][ky] += global_grad_Ux[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kz];
                                    dUdy_coarse_face[v][kx][ky] += global_grad_Uy[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kz];
                                    dUdz_coarse_face[v][kx][ky] += global_grad_Uz[nc_offset + v * N3 + kz * N2 + ky * N + kx] * weights[kz];
                                }
                            }
                        }
                    }

                    double UK_nb[5] = {}, dUdx_nb[5] = {}, dUdy_nb[5] = {}, dUdz_nb[5] = {}, Hv_R_nb[5] = {};
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = PX[kx][ix] * PY[ky][iy];
                            for (int v = 0; v < 5; ++v) {
                                UK_nb[v] += factor * U_coarse_face[v][kx][ky];
                                dUdx_nb[v] += factor * dUdx_coarse_face[v][kx][ky];
                                dUdy_nb[v] += factor * dUdy_coarse_face[v][kx][ky];
                                dUdz_nb[v] += factor * dUdz_coarse_face[v][kx][ky];
                            }
                        }
                    }
                    compute_viscous_flux_z_3d(UK_nb, dUdx_nb, dUdy_nb, dUdz_nb,
                                             mu, kappa, p.GAMMA, Hv_R_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR, p.POS_LIMITER_EPS);

                    double mu_R_face = mu;
                    if (p.ENABLE_SUTHERLAND) {
                        mu_R_face = 0.5 * (get_suth_mu(UK_face) + get_suth_mu(UK_nb));
                    }
                    double penalty_R = br2_eta * mu_R_face / c->dz;
                    double Hv_R_comm[5];
                    for (int v = 0; v < 5; ++v) {
                        Hv_R_comm[v] = 0.5 * (Hv_R[v] + Hv_R_nb[v])
                                     + penalty_R * (UK_nb[v] - UK_face[v]);
                    }

                    // Update fine cell local RHS
                    for (int iz = 0; iz < N; ++iz) {
                        for (int v = 0; v < 5; ++v) {
                            #pragma omp atomic
                            c->get_RHS(v, iz, iy, ix, N) += Hv_R_comm[v] * basis.dgr[iz] * (2.0 / c->dz);
                        }
                    }

                    // Restrict and accumulate to coarse neighbor nc Front face
                    for (int ky = 0; ky < N; ++ky) {
                        for (int kx = 0; kx < N; ++kx) {
                            double factor = RX[kx][ix] * RY[ky][iy];
                            for (int kz = 0; kz < N; ++kz) {
                                for (int v = 0; v < 5; ++v) {
                                    #pragma omp atomic
                                    nc->get_RHS(v, kz, ky, kx, N) += factor * Hv_R_comm[v] * dg_nc[kz] * (2.0 / nc->dz);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

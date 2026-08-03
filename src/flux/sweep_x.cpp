/**
 * @file sweep_x.cpp
 * @brief X-direction Flux Reconstruction sweep (tensor-product) on decoupled Cells.
 */

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::sweep_x() {
    // =========================================================================
    // Pass 1: Local & Conforming Sweep (highly optimized, vectorizable)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        for (int iy = 0; iy < p.N_PTS; ++iy) {

            // --- 1. Pointwise X-flux at each solution point ---
            double F_sol[MAX_PTS][4];
            double F_sol_S[MAX_PTS] = {};
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                get_flux_pointwise_cell(*c, iy, ix,
                                        F_sol[ix], nullptr,
                                        c->sigma_field[iy * p.N_PTS + ix]);
                if (p.ENABLE_PPR) {
                    double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                    double u   = c->get_U(1, iy, ix, p.N_PTS) / rho;
                    F_sol_S[ix] = u * c->S_field[iy * p.N_PTS + ix];
                }
            }

            // --- 2. Face-extrapolated states ---
            double UL_face[4] = {}, UR_face[4] = {};
            double sig_L_face = 0.0, sig_R_face = 0.0;
            double S_L_face = 0.0, S_R_face = 0.0;
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double s = c->sigma_field[iy * p.N_PTS + ix];
                sig_L_face += s * basis.l_L[ix];
                sig_R_face += s * basis.l_R[ix];
                if (p.ENABLE_PPR) {
                    S_L_face += c->S_field[iy * p.N_PTS + ix] * basis.l_L[ix];
                    S_R_face += c->S_field[iy * p.N_PTS + ix] * basis.l_R[ix];
                }
            }
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    UR_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[ix];
                }
            }

            // --- 3. Common Riemann fluxes (Local, Conforming & Boundary) ---
            double Flux_L_local[4] = {}, Flux_R_local[4] = {};
            double Flux_S_L_comm = 0.0, Flux_S_R_comm = 0.0;
            double U_neigh[4];
            double sig_neigh;

            // Left Face (0)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_L = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 0, iy) : nullptr;
            if (sfp_L) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_L, u_sb);
                double S_sb = compute_wall_phantom_pressure(UL_face, u_sb, S_L_face, p.PPR_WALL_BC, 2, p.POS_LIMITER_EPS, p.GAMMA);
                compute_interface_flux(u_sb, UL_face, sig_L_face, sig_L_face, S_sb, S_L_face, c->theta_avg, c->theta_avg, 0, Flux_L_local, Flux_S_L_comm);
            } else if (c->neighbors[0] && c->neighbors[0]->level == c->level) {
                Cell* nc = c->neighbors[0];
                char nface = c->neighbor_faces[0];
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[iy * p.N_PTS + k] * weights[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += nc->S_field[iy * p.N_PTS + k] * weights[k];
                    }
                }
                compute_interface_flux(U_neigh, UL_face, sig_neigh, sig_L_face, S_neigh, S_L_face, nc->theta_avg, c->theta_avg, 0, Flux_L_local, Flux_S_L_comm);
            } else if (c->is_boundary[0]) {
                double S_neigh = S_L_face;
                get_neigh_state_cell(*c, iy, false,
                                     UL_face, sig_L_face, U_neigh, sig_neigh, 0, S_L_face, &S_neigh);
                compute_interface_flux(U_neigh, UL_face, sig_neigh, sig_L_face, S_neigh, S_L_face, c->theta_avg, c->theta_avg, 0, Flux_L_local, Flux_S_L_comm);
            }

            // Right Face (1)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_R = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 1, iy) : nullptr;
            if (sfp_R) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_R, u_sb);
                double S_sb = compute_wall_phantom_pressure(UR_face, u_sb, S_R_face, p.PPR_WALL_BC, 2, p.POS_LIMITER_EPS, p.GAMMA);
                compute_interface_flux(UR_face, u_sb, sig_R_face, sig_R_face, S_R_face, S_sb, c->theta_avg, c->theta_avg, 0, Flux_R_local, Flux_S_R_comm);
            } else if (c->neighbors[1] && c->neighbors[1]->level == c->level) {
                Cell* nc = c->neighbors[1];
                char nface = c->neighbor_faces[1];
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[iy * p.N_PTS + k] * weights[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += nc->S_field[iy * p.N_PTS + k] * weights[k];
                    }
                }
                compute_interface_flux(UR_face, U_neigh, sig_R_face, sig_neigh, S_R_face, S_neigh, c->theta_avg, nc->theta_avg, 0, Flux_R_local, Flux_S_R_comm);
            } else if (c->is_boundary[1]) {
                double S_neigh = S_R_face;
                get_neigh_state_cell(*c, iy, true,
                                     UR_face, sig_R_face, U_neigh, sig_neigh, 0, S_R_face, &S_neigh);
                compute_interface_flux(UR_face, U_neigh, sig_R_face, sig_neigh, S_R_face, S_neigh, c->theta_avg, c->theta_avg, 0, Flux_R_local, Flux_S_R_comm);
            }

            // --- 4. Interior flux at faces (for correction) ---
            double F_L[4] = {}, F_R[4] = {};
            double F_S_L = 0.0, F_S_R = 0.0;
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    F_L[v] += F_sol[ix][v] * basis.l_L[ix];
                    F_R[v] += F_sol[ix][v] * basis.l_R[ix];
                }
            }
            if (p.ENABLE_PPR) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    F_S_L += F_sol_S[ix] * basis.l_L[ix];
                    F_S_R += F_sol_S[ix] * basis.l_R[ix];
                }
            }

            // --- 5. Accumulate into RHS (single-write, extremely fast) ---
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double df = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        df += basis.D[ix][k] * F_sol[k][v];
                    
                    c->get_RHS(v, iy, ix, p.N_PTS) -= 
                        (df
                         + (Flux_L_local[v] - F_L[v]) * basis.dgl[ix]
                         + (Flux_R_local[v] - F_R[v]) * basis.dgr[ix])
                        * (2.0 / c->dx);
                }
            }
            if (p.ENABLE_PPR) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double df_S = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        df_S += basis.D[ix][k] * F_sol_S[k];

                    c->S_RHS[iy * p.N_PTS + ix] -=
                        (df_S
                         + (Flux_S_L_comm - F_S_L) * basis.dgl[ix]
                         + (Flux_S_R_comm - F_S_R) * basis.dgr[ix])
                        * (2.0 / c->dx);
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Interface Sweep (sparse, only active near hanging nodes)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // Left Face (0) non-conforming coarser neighbor
        if (c->neighbors[0] && c->neighbors[0]->level < c->level) {
            Cell* nc = c->neighbors[0];
            char nface = c->neighbor_faces[0];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double UL_face[4] = {};
                double sig_L_face = 0.0;
                double S_L_face = 0.0;
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_L_face += s * basis.l_L[ix];
                    if (p.ENABLE_PPR) {
                        S_L_face += c->S_field[iy * p.N_PTS + ix] * basis.l_L[ix];
                    }
                    for (int v = 0; v < 4; ++v) {
                        UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                double S_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                        }
                        sig_coarse_face[ky] += nc->sigma_field[ky * p.N_PTS + kx] * weights[kx];
                        if (p.ENABLE_PPR) {
                            S_coarse_face[ky] += nc->S_field[ky * p.N_PTS + kx] * weights[kx];
                        }
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[ky][iy] * U_coarse_face[v][ky];
                    }
                    sig_neigh += P[ky][iy] * sig_coarse_face[ky];
                    if (p.ENABLE_PPR) {
                        S_neigh += P[ky][iy] * S_coarse_face[ky];
                    }
                }

                double Flux_L_comm[4];
                double Flux_S_L_comm = 0.0;
                compute_interface_flux(U_neigh, UL_face, sig_neigh, sig_L_face, S_neigh, S_L_face, nc->theta_avg, c->theta_avg, 0, Flux_L_comm, Flux_S_L_comm);

                // Update fine cell
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_L_comm[v] * basis.dgl[ix] * (2.0 / c->dx);
                    }
                    if (p.ENABLE_PPR) {
                        #pragma omp atomic
                        c->S_RHS[iy * p.N_PTS + ix] -= Flux_S_L_comm * basis.dgl[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_L_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            nc->S_RHS[ky * p.N_PTS + kx] -= factor * Flux_S_L_comm * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }

        // Right Face (1) non-conforming coarser neighbor
        if (c->neighbors[1] && c->neighbors[1]->level < c->level) {
            Cell* nc = c->neighbors[1];
            char nface = c->neighbor_faces[1];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double UR_face[4] = {};
                double sig_R_face = 0.0;
                double S_R_face = 0.0;
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_R_face += s * basis.l_R[ix];
                    if (p.ENABLE_PPR) {
                        S_R_face += c->S_field[iy * p.N_PTS + ix] * basis.l_R[ix];
                    }
                    for (int v = 0; v < 4; ++v) {
                        UR_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[ix];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                double S_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                        }
                        sig_coarse_face[ky] += nc->sigma_field[ky * p.N_PTS + kx] * weights[kx];
                        if (p.ENABLE_PPR) {
                            S_coarse_face[ky] += nc->S_field[ky * p.N_PTS + kx] * weights[kx];
                        }
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[ky][iy] * U_coarse_face[v][ky];
                    }
                    sig_neigh += P[ky][iy] * sig_coarse_face[ky];
                    if (p.ENABLE_PPR) {
                        S_neigh += P[ky][iy] * S_coarse_face[ky];
                    }
                }

                double Flux_R_comm[4];
                double Flux_S_R_comm = 0.0;
                compute_interface_flux(UR_face, U_neigh, sig_R_face, sig_neigh, S_R_face, S_neigh, c->theta_avg, nc->theta_avg, 0, Flux_R_comm, Flux_S_R_comm);

                // Update fine cell
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_R_comm[v] * basis.dgr[ix] * (2.0 / c->dx);
                    }
                    if (p.ENABLE_PPR) {
                        #pragma omp atomic
                        c->S_RHS[iy * p.N_PTS + ix] -= Flux_S_R_comm * basis.dgr[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_R_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            nc->S_RHS[ky * p.N_PTS + kx] -= factor * Flux_S_R_comm * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }
    }
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

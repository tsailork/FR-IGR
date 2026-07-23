/**
 * @file sweep_y.cpp
 * @brief Y-direction Flux Reconstruction sweep (tensor-product) on decoupled Cells.
 */

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::sweep_y() {
    // =========================================================================
    // Pass 1: Local & Conforming Sweep (highly optimized, vectorizable)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        for (int ix = 0; ix < p.N_PTS; ++ix) {

            // --- 1. Pointwise Y-flux ---
            double G_sol[MAX_PTS][4];
            double G_sol_S[MAX_PTS] = {};
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                get_flux_pointwise_cell(*c, iy, ix,
                                        nullptr, G_sol[iy],
                                        c->sigma_field[iy * p.N_PTS + ix]);
                if (p.ENABLE_PPR) {
                    double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                    double v   = c->get_U(2, iy, ix, p.N_PTS) / rho;
                    // Part A: apply y-component of acoustic pressure-gradient correction using stored gradient
                    if (p.PPR_GRAD_ADV_SCALE > 0.0) {
                        int idx = iy * p.N_PTS + ix;
                        double dP_dx = c->grad_px_field[idx];
                        double dP_dy = c->grad_py_field[idx];
                        double u     = c->get_U(1, iy, ix, p.N_PTS) / rho;
                        double press = std::max(p.POS_LIMITER_EPS,
                            (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, p.N_PTS) - 0.5*rho*(u*u+v*v)));
                        double a_loc = std::sqrt(p.GAMMA * press / rho);
                        double grad_norm = std::sqrt(dP_dx*dP_dx + dP_dy*dP_dy) + p.PPR_GRAD_EPS;
                        v += p.PPR_GRAD_ADV_SCALE * a_loc * (dP_dy / grad_norm); // +sign: push S toward high-P side
                    }
                    G_sol_S[iy] = c->S_field[iy * p.N_PTS + ix] * (p.PPR_ADV_MULT * v);
                }
            }

            // --- 2. Face-extrapolated states ---
            double UB_face[4] = {}, UT_face[4] = {};
            double sig_B_face = 0.0, sig_T_face = 0.0;
            double S_B_face = 0.0, S_T_face = 0.0;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double s = c->sigma_field[iy * p.N_PTS + ix];
                sig_B_face += s * basis.l_L[iy];
                sig_T_face += s * basis.l_R[iy];
                if (p.ENABLE_PPR) {
                    S_B_face += c->S_field[iy * p.N_PTS + ix] * basis.l_L[iy];
                    S_T_face += c->S_field[iy * p.N_PTS + ix] * basis.l_R[iy];
                }
            }
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    UB_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[iy];
                    UT_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[iy];
                }
            }

            // --- 3. Common Riemann fluxes (Local, Conforming & Boundary) ---
            double Flux_B_local[4] = {}, Flux_T_local[4] = {};
            double Flux_S_B_comm = 0.0, Flux_S_T_comm = 0.0;
            double U_neigh[4];
            double sig_neigh;

            // Bottom Face (2)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_B = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 2, ix) : nullptr;
            if (sfp_B) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_B, u_sb);
                double rho_sb = std::max(p.POS_LIMITER_EPS, u_sb[0]);
                double p_sb = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / rho_sb));
                double S_sb = rho_sb * p_sb;
                compute_interface_flux(u_sb, UB_face, sig_B_face, sig_B_face, S_sb, S_B_face, c->theta_avg, c->theta_avg, 1, Flux_B_local, Flux_S_B_comm);
            } else if (c->neighbors[2] && c->neighbors[2]->level == c->level) {
                Cell* nc = c->neighbors[2];
                char nface = c->neighbor_faces[2];
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[k * p.N_PTS + ix] * weights[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += nc->S_field[k * p.N_PTS + ix] * weights[k];
                    }
                }
                compute_interface_flux(U_neigh, UB_face, sig_neigh, sig_B_face, S_neigh, S_B_face, nc->theta_avg, c->theta_avg, 1, Flux_B_local, Flux_S_B_comm);
            } else if (c->is_boundary[2]) {
                get_neigh_state_cell(*c, ix, false,
                                     UB_face, sig_B_face, U_neigh, sig_neigh, 1);
                double S_neigh = S_B_face;
                if (p.ENABLE_PPR) {
                    double p_phan_local = S_B_face / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                    double p_phan_ghost = p_phan_local;
                    const NeighborInfo& ni = c->boundary_info[2];
                    if (ni.is_supersonic_inflow) {
                        p_phan_ghost = ni.ref_p;
                    } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                        double v_ghost_n = -U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                        if (v_ghost_n < 0.0) {
                            p_phan_ghost = ni.ref_p;
                        }
                    }
                    S_neigh = U_neigh[0] * p_phan_ghost;
                }
                compute_interface_flux(U_neigh, UB_face, sig_neigh, sig_B_face, S_neigh, S_B_face, c->theta_avg, c->theta_avg, 1, Flux_B_local, Flux_S_B_comm);
            }

            // Top Face (3)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_T = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 3, ix) : nullptr;
            if (sfp_T) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_T, u_sb);
                double rho_sb = std::max(p.POS_LIMITER_EPS, u_sb[0]);
                double p_sb = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / rho_sb));
                double S_sb = rho_sb * p_sb;
                compute_interface_flux(UT_face, u_sb, sig_T_face, sig_T_face, S_T_face, S_sb, c->theta_avg, c->theta_avg, 1, Flux_T_local, Flux_S_T_comm);
            } else if (c->neighbors[3] && c->neighbors[3]->level == c->level) {
                Cell* nc = c->neighbors[3];
                char nface = c->neighbor_faces[3];
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[k * p.N_PTS + ix] * weights[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += nc->S_field[k * p.N_PTS + ix] * weights[k];
                    }
                }
                compute_interface_flux(UT_face, U_neigh, sig_T_face, sig_neigh, S_T_face, S_neigh, c->theta_avg, nc->theta_avg, 1, Flux_T_local, Flux_S_T_comm);
            } else if (c->is_boundary[3]) {
                get_neigh_state_cell(*c, ix, true,
                                     UT_face, sig_T_face, U_neigh, sig_neigh, 1);
                double S_neigh = S_T_face;
                if (p.ENABLE_PPR) {
                    double p_phan_local = S_T_face / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                    double p_phan_ghost = p_phan_local;
                    const NeighborInfo& ni = c->boundary_info[3];
                    if (ni.is_supersonic_inflow) {
                        p_phan_ghost = ni.ref_p;
                    } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                        double v_ghost_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                        if (v_ghost_n < 0.0) {
                            p_phan_ghost = ni.ref_p;
                        }
                    }
                    S_neigh = U_neigh[0] * p_phan_ghost;
                }
                compute_interface_flux(UT_face, U_neigh, sig_T_face, sig_neigh, S_T_face, S_neigh, c->theta_avg, c->theta_avg, 1, Flux_T_local, Flux_S_T_comm);
            }

            // --- 4. Interior flux at faces ---
            double G_B[4] = {}, G_T[4] = {};
            double G_S_B = 0.0, G_S_T = 0.0;
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    G_B[v] += G_sol[iy][v] * basis.l_L[iy];
                    G_T[v] += G_sol[iy][v] * basis.l_R[iy];
                }
            }
            if (p.ENABLE_PPR) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    G_S_B += G_sol_S[iy] * basis.l_L[iy];
                    G_S_T += G_sol_S[iy] * basis.l_R[iy];
                }
            }

            // --- 5. Accumulate into RHS (single-write, extremely fast) ---
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double dg = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        dg += basis.D[iy][k] * G_sol[k][v];
                    
                    c->get_RHS(v, iy, ix, p.N_PTS) -= 
                        (dg
                         + (Flux_B_local[v] - G_B[v]) * basis.dgl[iy]
                         + (Flux_T_local[v] - G_T[v]) * basis.dgr[iy])
                        * (2.0 / c->dy);
                }
            }
            if (p.ENABLE_PPR) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double dg_S = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        dg_S += basis.D[iy][k] * G_sol_S[k];

                    c->S_RHS[iy * p.N_PTS + ix] -=
                        (dg_S
                         + (Flux_S_B_comm - G_S_B) * basis.dgl[iy]
                         + (Flux_S_T_comm - G_S_T) * basis.dgr[iy])
                        * (2.0 / c->dy);
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

        // Bottom Face (2) non-conforming coarser neighbor
        if (c->neighbors[2] && c->neighbors[2]->level < c->level) {
            Cell* nc = c->neighbors[2];
            char nface = c->neighbor_faces[2];
            int child_idx = c->ex & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double UB_face[4] = {};
                double sig_B_face = 0.0;
                double S_B_face = 0.0;
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_B_face += s * basis.l_L[iy];
                    if (p.ENABLE_PPR) {
                        S_B_face += c->S_field[iy * p.N_PTS + ix] * basis.l_L[iy];
                    }
                    for (int v = 0; v < 4; ++v) {
                        UB_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[iy];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                double S_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                        }
                        sig_coarse_face[kx] += nc->sigma_field[ky * p.N_PTS + kx] * weights[ky];
                        if (p.ENABLE_PPR) {
                            S_coarse_face[kx] += nc->S_field[ky * p.N_PTS + kx] * weights[ky];
                        }
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[kx][ix] * U_coarse_face[v][kx];
                    }
                    sig_neigh += P[kx][ix] * sig_coarse_face[kx];
                    if (p.ENABLE_PPR) {
                        S_neigh += P[kx][ix] * S_coarse_face[kx];
                    }
                }

                double Flux_B_comm[4];
                double Flux_S_B_comm = 0.0;
                compute_interface_flux(U_neigh, UB_face, sig_neigh, sig_B_face, S_neigh, S_B_face, nc->theta_avg, c->theta_avg, 1, Flux_B_comm, Flux_S_B_comm);

                // Update fine cell
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_B_comm[v] * basis.dgl[iy] * (2.0 / c->dy);
                    }
                    if (p.ENABLE_PPR) {
                        #pragma omp atomic
                        c->S_RHS[iy * p.N_PTS + ix] -= Flux_S_B_comm * basis.dgl[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_B_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            nc->S_RHS[ky * p.N_PTS + kx] -= factor * Flux_S_B_comm * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }

        // Top Face (3) non-conforming coarser neighbor
        if (c->neighbors[3] && c->neighbors[3]->level < c->level) {
            Cell* nc = c->neighbors[3];
            char nface = c->neighbor_faces[3];
            int child_idx = c->ex & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double UT_face[4] = {};
                double sig_T_face = 0.0;
                double S_T_face = 0.0;
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_T_face += s * basis.l_R[iy];
                    if (p.ENABLE_PPR) {
                        S_T_face += c->S_field[iy * p.N_PTS + ix] * basis.l_R[iy];
                    }
                    for (int v = 0; v < 4; ++v) {
                        UT_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[iy];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                double S_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                        }
                        sig_coarse_face[kx] += nc->sigma_field[ky * p.N_PTS + kx] * weights[ky];
                        if (p.ENABLE_PPR) {
                            S_coarse_face[kx] += nc->S_field[ky * p.N_PTS + kx] * weights[ky];
                        }
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[kx][ix] * U_coarse_face[v][kx];
                    }
                    sig_neigh += P[kx][ix] * sig_coarse_face[kx];
                    if (p.ENABLE_PPR) {
                        S_neigh += P[kx][ix] * S_coarse_face[kx];
                    }
                }

                double Flux_T_comm[4];
                double Flux_S_T_comm = 0.0;
                compute_interface_flux(UT_face, U_neigh, sig_T_face, sig_neigh, S_T_face, S_neigh, c->theta_avg, nc->theta_avg, 1, Flux_T_comm, Flux_S_T_comm);

                // Update fine cell
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_T_comm[v] * basis.dgr[iy] * (2.0 / c->dy);
                    }
                    if (p.ENABLE_PPR) {
                        #pragma omp atomic
                        c->S_RHS[iy * p.N_PTS + ix] -= Flux_S_T_comm * basis.dgr[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_T_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            nc->S_RHS[ky * p.N_PTS + kx] -= factor * Flux_S_T_comm * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
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
                        double u   = c->get_U(1, iz, iy, ix, N) / rho;
                        double v   = c->get_U(2, iz, iy, ix, N) / rho;
                        double w   = c->get_U(3, iz, iy, ix, N) / rho;
                        if (p.PPR_GRAD_ADV_SCALE > 0.0) {
                            int idx = iz * N2 + iy * N + ix;
                            double dP_dx = c->grad_px_field[idx];
                            double dP_dy = c->grad_py_field[idx];
                            double dP_dz = c->grad_pz_field[idx];
                            double press = std::max(p.POS_LIMITER_EPS,
                                (p.GAMMA - 1.0) * (c->get_U(4, iz, iy, ix, N) - 0.5*rho*(u*u+v*v+w*w)));
                            double a_loc = std::sqrt(p.GAMMA * press / rho);
                            double grad_norm = std::sqrt(dP_dx*dP_dx + dP_dy*dP_dy + dP_dz*dP_dz) + p.PPR_GRAD_EPS;
                            v += p.PPR_GRAD_ADV_SCALE * a_loc * (dP_dy / grad_norm);
                        }
                        G_sol_S[iy] = c->S_field[iz * N2 + iy * N + ix] * (p.PPR_ADV_MULT * v);
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
                    get_neigh_state_cell(*c, iz * N + ix, false,
                                         UB_face, sig_B_face, U_neigh, sig_neigh, 1);
                    double S_neigh = S_B_face;
                    if (p.ENABLE_PPR) {
                        double p_phan_local = S_B_face / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                        double p_phan_ghost = p_phan_local;
                        const NeighborInfo& ni = c->boundary_info[2];
                        if (ni.is_supersonic_inflow) {
                            p_phan_ghost = ni.ref_p;
                        } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                            double u_ghost_n = -U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                            if (u_ghost_n < 0.0) {
                                p_phan_ghost = ni.ref_p;
                            }
                        }
                        S_neigh = U_neigh[0] * p_phan_ghost;
                    }
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
                    get_neigh_state_cell(*c, iz * N + ix, true,
                                         UT_face, sig_T_face, U_neigh, sig_neigh, 1);
                    double S_neigh = S_T_face;
                    if (p.ENABLE_PPR) {
                        double p_phan_local = S_T_face / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                        double p_phan_ghost = p_phan_local;
                        const NeighborInfo& ni = c->boundary_info[3];
                        if (ni.is_supersonic_inflow) {
                            p_phan_ghost = ni.ref_p;
                        } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                            double u_ghost_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                            if (u_ghost_n < 0.0) {
                                p_phan_ghost = ni.ref_p;
                            }
                        }
                        S_neigh = U_neigh[0] * p_phan_ghost;
                    }
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

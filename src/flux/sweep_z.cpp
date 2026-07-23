/**
 * @file sweep_z.cpp
 * @brief Z-direction Flux Reconstruction sweep (tensor-product) on decoupled Cells.
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// =========================================================================
// SolverDim<3> 3D Inviscid Sweep Z Implementation
// =========================================================================

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
                            w += p.PPR_GRAD_ADV_SCALE * a_loc * (dP_dz / grad_norm);
                        }
                        H_sol_S[iz] = c->S_field[iz * N2 + iy * N + ix] * (p.PPR_ADV_MULT * w);
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
                    get_neigh_state_cell(*c, iy * N + ix, false,
                                         UF_face, sig_F_face, U_neigh, sig_neigh, 2);
                    double S_neigh = S_F_face;
                    if (p.ENABLE_PPR) {
                        double p_phan_local = S_F_face / std::max(p.POS_LIMITER_EPS, UF_face[0]);
                        double p_phan_ghost = p_phan_local;
                        const NeighborInfo& ni = c->boundary_info[4];
                        if (ni.is_supersonic_inflow) {
                            p_phan_ghost = ni.ref_p;
                        } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                            double u_ghost_n = -U_neigh[3] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                            if (u_ghost_n < 0.0) {
                                p_phan_ghost = ni.ref_p;
                            }
                        }
                        S_neigh = U_neigh[0] * p_phan_ghost;
                    }
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
                    get_neigh_state_cell(*c, iy * N + ix, true,
                                         UK_face, sig_K_face, U_neigh, sig_neigh, 2);
                    double S_neigh = S_K_face;
                    if (p.ENABLE_PPR) {
                        double p_phan_local = S_K_face / std::max(p.POS_LIMITER_EPS, UK_face[0]);
                        double p_phan_ghost = p_phan_local;
                        const NeighborInfo& ni = c->boundary_info[5];
                        if (ni.is_supersonic_inflow) {
                            p_phan_ghost = ni.ref_p;
                        } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                            double u_ghost_n = U_neigh[3] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                            if (u_ghost_n < 0.0) {
                                p_phan_ghost = ni.ref_p;
                            }
                        }
                        S_neigh = U_neigh[0] * p_phan_ghost;
                    }
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

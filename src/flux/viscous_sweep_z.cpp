/**
 * @file viscous_sweep_z.cpp
 * @brief Z-direction viscous flux divergence (BR2 Phase 2) on decoupled Cells.
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// =========================================================================
// SolverDim<3> 3D Viscous Sweep Z Implementation
// =========================================================================

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

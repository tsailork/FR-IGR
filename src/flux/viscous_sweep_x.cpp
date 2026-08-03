/**
 * @file viscous_sweep_x.cpp
 * @brief X-direction viscous flux divergence (BR2 Phase 2) on decoupled Cells.
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

static void compute_viscous_flux_x(const double U[4],
                                    const double dUdx[4], const double dUdy[4],
                                    double mu, double kappa, double gamma,
                                    double Fv[4], bool enable_suth, double suth_c, double pr)
{
    double rho = std::max(1e-14, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;

    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;

    double p = (gamma - 1.0) * (U[3] - 0.5 * rho * (u*u + v*v));
    if (p < 1e-14) p = 1e-14;
    double T = p / rho;

    double mu_local = mu;
    double kappa_local = kappa;
    if (enable_suth) {
        double T_norm = std::max(1e-8, T);
        mu_local = mu * (std::pow(T_norm, 1.5) * (1.0 + suth_c) / (T_norm + suth_c));
        kappa_local = mu_local * gamma / ((gamma - 1.0) * pr);
    }

    double dpdx = (gamma - 1.0) * (dUdx[3] - 0.5*(u*u+v*v)*dUdx[0]
                                    - rho*(u*dudx + v*dvdx));
    double dTdx = (dpdx - T * dUdx[0]) / rho;

    double tau_xx = mu_local * (4.0/3.0 * dudx - 2.0/3.0 * dvdy);
    double tau_xy = mu_local * (dudy + dvdx);

    double qx = -kappa_local * dTdx;

    Fv[0] = 0.0;
    Fv[1] = tau_xx;
    Fv[2] = tau_xy;
    Fv[3] = u * tau_xx + v * tau_xy - qx;
}

void Solver::viscous_sweep_x() {
    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);

    // =========================================================================
    // Pass 1: Local & Conforming Pass (highly optimized, vectorizable)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        auto get_suth_mu = [&](const double State[4]) {
            double r = std::max(1e-14, State[0]);
            double press = (p.GAMMA - 1.0) * (State[3] - 0.5 * (State[1]*State[1] + State[2]*State[2]) / r);
            if (press < 1e-14) press = 1e-14;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        auto extrapolate_neighbor_x = [&](const Cell* nc, char nface, int iy, double U_nb[4], double dUdx_nb[4], double dUdy_nb[4]) {
            const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
            for (int v = 0; v < 4; ++v) {
                U_nb[v] = 0.0;
                dUdx_nb[v] = 0.0;
                dUdy_nb[v] = 0.0;
            }
            int nc_offset = nc->cell_index * 4 * p.N_PTS * p.N_PTS;
            for (int k = 0; k < p.N_PTS; ++k) {
                int flat = iy * p.N_PTS + k;
                for (int v = 0; v < 4; ++v) {
                    U_nb[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    dUdx_nb[v] += global_grad_Ux[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[k];
                    dUdy_nb[v] += global_grad_Uy[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[k];
                }
            }
        };

        int c_offset = c->cell_index * 4 * p.N_PTS * p.N_PTS;

        for (int iy = 0; iy < p.N_PTS; ++iy) {

            // --- 1. Pointwise viscous X-flux at each solution point ---
            double Fv_sol[MAX_PTS][4];
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                int flat = iy * p.N_PTS + ix;
                double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                for (int v = 0; v < 4; ++v) {
                    U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                    dUdx_pt[v] = global_grad_Ux[c_offset + v * p.N_PTS * p.N_PTS + flat];
                    dUdy_pt[v] = global_grad_Uy[c_offset + v * p.N_PTS * p.N_PTS + flat];
                }
                compute_viscous_flux_x(U_pt, dUdx_pt, dUdy_pt,
                                        mu, kappa, p.GAMMA, Fv_sol[ix],
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
            }

            // --- 2. Face-extrapolated viscous fluxes & states ---
            double Fv_L[4] = {}, Fv_R[4] = {};
            double UL_face[4] = {}, UR_face[4] = {};
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    Fv_L[v] += Fv_sol[ix][v] * basis.l_L[ix];
                    Fv_R[v] += Fv_sol[ix][v] * basis.l_R[ix];
                    UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    UR_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[ix];
                }
            }

            // --- 3. Left common viscous flux (conforming/boundary/SFP) ---
            double Fv_L_local[4] = {};
            if (c->neighbors[0] && c->neighbors[0]->level == c->level) {
                Cell* nc = c->neighbors[0];
                char nface = c->neighbor_faces[0];
                double UL_nb[4], dUdx_nb[4], dUdy_nb[4], Fv_L_nb[4];
                extrapolate_neighbor_x(nc, nface, iy, UL_nb, dUdx_nb, dUdy_nb);
                compute_viscous_flux_x(UL_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Fv_L_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_L_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_L_face = 0.5 * (get_suth_mu(UL_face) + get_suth_mu(UL_nb));
                }
                double penalty_L = br2_eta * mu_L_face / c->dx;
                for (int v = 0; v < 4; ++v) {
                    Fv_L_local[v] = 0.5 * (Fv_L_nb[v] + Fv_L[v])
                                  + penalty_L * (UL_face[v] - UL_nb[v]);
                }
            } else if (c->is_boundary[0]) {
                double UL_nb[4], sig_dummy;
                get_neigh_state_cell(*c, iy, false, UL_face, 0.0, UL_nb, sig_dummy, 0);
                for (int v = 0; v < 4; ++v) {
                    Fv_L_local[v] = Fv_L[v];
                }
            }

            // --- 4. Right common viscous flux (conforming/boundary/SFP) ---
            double Fv_R_local[4] = {};
            if (c->neighbors[1] && c->neighbors[1]->level == c->level) {
                Cell* nc = c->neighbors[1];
                char nface = c->neighbor_faces[1];
                double UR_nb[4], dUdx_nb[4], dUdy_nb[4], Fv_R_nb[4];
                extrapolate_neighbor_x(nc, nface, iy, UR_nb, dUdx_nb, dUdy_nb);
                compute_viscous_flux_x(UR_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Fv_R_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_R_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_R_face = 0.5 * (get_suth_mu(UR_face) + get_suth_mu(UR_nb));
                }
                double penalty_R = br2_eta * mu_R_face / c->dx;
                for (int v = 0; v < 4; ++v) {
                    Fv_R_local[v] = 0.5 * (Fv_R[v] + Fv_R_nb[v])
                                  + penalty_R * (UR_nb[v] - UR_face[v]);
                }
            } else if (c->is_boundary[1]) {
                double UR_nb[4], sig_dummy;
                get_neigh_state_cell(*c, iy, true, UR_face, 0.0, UR_nb, sig_dummy, 0);
                for (int v = 0; v < 4; ++v) {
                    Fv_R_local[v] = Fv_R[v];
                }
            }

            // --- 5. Accumulate into RHS (single-write, extremely fast) ---
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double df = 0.0;
                    for (int kk = 0; kk < p.N_PTS; ++kk)
                        df += basis.D[ix][kk] * Fv_sol[kk][v];
                    
                    c->get_RHS(v, iy, ix, p.N_PTS) += 
                        (df
                         + (Fv_L_local[v] - Fv_L[v]) * basis.dgl[ix]
                         + (Fv_R_local[v] - Fv_R[v]) * basis.dgr[ix])
                        * (2.0 / c->dx);
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Restriction Pass (sparse, only active near hanging nodes)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        int c_offset = c->cell_index * 4 * p.N_PTS * p.N_PTS;

        auto get_suth_mu = [&](const double State[4]) {
            double r = std::max(1e-14, State[0]);
            double press = (p.GAMMA - 1.0) * (State[3] - 0.5 * (State[1]*State[1] + State[2]*State[2]) / r);
            if (press < 1e-14) press = 1e-14;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        // Left Face (0) non-conforming coarser neighbor
        if (c->neighbors[0] && c->neighbors[0]->level < c->level) {
            Cell* nc = c->neighbors[0];
            int nc_offset = nc->cell_index * 4 * p.N_PTS * p.N_PTS;
            char nface = c->neighbor_faces[0];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                // Pointwise viscous X-flux at fine cell left face
                double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                double Fv_L[4] = {};
                double UL_face[4] = {};
                for (int v = 0; v < 4; ++v) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    }
                }
                double Fv_sol[MAX_PTS][4];
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int flat = iy * p.N_PTS + ix;
                    double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                    for (int v = 0; v < 4; ++v) {
                        U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * p.N_PTS * p.N_PTS + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * p.N_PTS * p.N_PTS + flat];
                    }
                    compute_viscous_flux_x(U_pt, dUdx_pt, dUdy_pt,
                                            mu, kappa, p.GAMMA, Fv_sol[ix],
                                            p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                }
                for (int v = 0; v < 4; ++v) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        Fv_L[v] += Fv_sol[ix][v] * basis.l_L[ix];
                    }
                }

                // Coarser neighbor extrapolation
                double UL_nb[4] = {}, dUdx_nb[4] = {}, dUdy_nb[4] = {};
                double U_coarse_face[4][MAX_PTS] = {};
                double dUdx_coarse_face[4][MAX_PTS] = {};
                double dUdy_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        int flat = ky * p.N_PTS + kx;
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                            dUdx_coarse_face[v][ky] += global_grad_Ux[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[kx];
                            dUdy_coarse_face[v][ky] += global_grad_Uy[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[kx];
                        }
                    }
                }
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = P[ky][iy];
                    for (int v = 0; v < 4; ++v) {
                        UL_nb[v] += factor * U_coarse_face[v][ky];
                        dUdx_nb[v] += factor * dUdx_coarse_face[v][ky];
                        dUdy_nb[v] += factor * dUdy_coarse_face[v][ky];
                    }
                }

                double Fv_L_nb[4] = {};
                compute_viscous_flux_x(UL_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Fv_L_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_L_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_L_face = 0.5 * (get_suth_mu(UL_face) + get_suth_mu(UL_nb));
                }
                double penalty_L = br2_eta * mu_L_face / c->dx;
                double Fv_L_comm[4];
                for (int v = 0; v < 4; ++v) {
                    Fv_L_comm[v] = 0.5 * (Fv_L_nb[v] + Fv_L[v])
                                 + penalty_L * (UL_face[v] - UL_nb[v]);
                }

                // Update fine cell local RHS
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) += Fv_L_comm[v] * basis.dgl[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict and accumulate to coarse neighbor nc Right face
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) += factor * Fv_L_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }

        // Right Face (1) non-conforming coarser neighbor
        if (c->neighbors[1] && c->neighbors[1]->level < c->level) {
            Cell* nc = c->neighbors[1];
            int nc_offset = nc->cell_index * 4 * p.N_PTS * p.N_PTS;
            char nface = c->neighbor_faces[1];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double Fv_R[4] = {};
                double UR_face[4] = {};
                for (int v = 0; v < 4; ++v) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        UR_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[ix];
                    }
                }
                double Fv_sol[MAX_PTS][4];
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int flat = iy * p.N_PTS + ix;
                    double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                    for (int v = 0; v < 4; ++v) {
                        U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * p.N_PTS * p.N_PTS + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * p.N_PTS * p.N_PTS + flat];
                    }
                    compute_viscous_flux_x(U_pt, dUdx_pt, dUdy_pt,
                                            mu, kappa, p.GAMMA, Fv_sol[ix],
                                            p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                }
                for (int v = 0; v < 4; ++v) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        Fv_R[v] += Fv_sol[ix][v] * basis.l_R[ix];
                    }
                }

                // Coarser neighbor extrapolation
                double UR_nb[4] = {}, dUdx_nb[4] = {}, dUdy_nb[4] = {};
                double U_coarse_face[4][MAX_PTS] = {};
                double dUdx_coarse_face[4][MAX_PTS] = {};
                double dUdy_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        int flat = ky * p.N_PTS + kx;
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                            dUdx_coarse_face[v][ky] += global_grad_Ux[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[kx];
                            dUdy_coarse_face[v][ky] += global_grad_Uy[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[kx];
                        }
                    }
                }
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = P[ky][iy];
                    for (int v = 0; v < 4; ++v) {
                        UR_nb[v] += factor * U_coarse_face[v][ky];
                        dUdx_nb[v] += factor * dUdx_coarse_face[v][ky];
                        dUdy_nb[v] += factor * dUdy_coarse_face[v][ky];
                    }
                }

                double Fv_R_nb[4] = {};
                compute_viscous_flux_x(UR_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Fv_R_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_R_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_R_face = 0.5 * (get_suth_mu(UR_face) + get_suth_mu(UR_nb));
                }
                double penalty_R = br2_eta * mu_R_face / c->dx;
                double Fv_R_comm[4];
                for (int v = 0; v < 4; ++v) {
                    Fv_R_comm[v] = 0.5 * (Fv_R[v] + Fv_R_nb[v])
                                 + penalty_R * (UR_nb[v] - UR_face[v]);
                }

                // Update fine cell local RHS
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) += Fv_R_comm[v] * basis.dgr[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict and accumulate to coarse neighbor nc Left face
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) += factor * Fv_R_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }
    }
}
// =========================================================================
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

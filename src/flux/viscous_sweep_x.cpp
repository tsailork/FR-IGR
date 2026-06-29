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
            for (int k = 0; k < p.N_PTS; ++k) {
                int flat = iy * p.N_PTS + k;
                for (int v = 0; v < 4; ++v) {
                    U_nb[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    dUdx_nb[v] += nc->grad_Ux[v * p.N_PTS * p.N_PTS + flat] * weights[k];
                    dUdy_nb[v] += nc->grad_Uy[v * p.N_PTS * p.N_PTS + flat] * weights[k];
                }
            }
        };

        for (int iy = 0; iy < p.N_PTS; ++iy) {

            // --- 1. Pointwise viscous X-flux at each solution point ---
            double Fv_sol[MAX_PTS][4];
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                int flat = iy * p.N_PTS + ix;
                double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                for (int v = 0; v < 4; ++v) {
                    U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                    dUdx_pt[v] = c->grad_Ux[v * p.N_PTS * p.N_PTS + flat];
                    dUdy_pt[v] = c->grad_Uy[v * p.N_PTS * p.N_PTS + flat];
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
            char nface = c->neighbor_faces[0];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                // Pointwise viscous X-flux at fine cell left face
                double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                int flat_pt = iy * p.N_PTS; // left face ix=0
                // Wait! To get UL_face and Fv_L:
                double Fv_L[4] = {};
                double UL_face[4] = {};
                for (int v = 0; v < 4; ++v) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    }
                }
                // Compute Fv_L using extrapolated values or integrated?
                // The original code computed Fv_L by integrating Fv_sol over face.
                // We should do the exact same:
                double Fv_sol[MAX_PTS][4];
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int flat = iy * p.N_PTS + ix;
                    double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                    for (int v = 0; v < 4; ++v) {
                        U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                        dUdx_pt[v] = c->grad_Ux[v * p.N_PTS * p.N_PTS + flat];
                        dUdy_pt[v] = c->grad_Uy[v * p.N_PTS * p.N_PTS + flat];
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
                            dUdx_coarse_face[v][ky] += nc->grad_Ux[v * p.N_PTS * p.N_PTS + flat] * weights[kx];
                            dUdy_coarse_face[v][ky] += nc->grad_Uy[v * p.N_PTS * p.N_PTS + flat] * weights[kx];
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
                        dUdx_pt[v] = c->grad_Ux[v * p.N_PTS * p.N_PTS + flat];
                        dUdy_pt[v] = c->grad_Uy[v * p.N_PTS * p.N_PTS + flat];
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
                            dUdx_coarse_face[v][ky] += nc->grad_Ux[v * p.N_PTS * p.N_PTS + flat] * weights[kx];
                            dUdy_coarse_face[v][ky] += nc->grad_Uy[v * p.N_PTS * p.N_PTS + flat] * weights[kx];
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

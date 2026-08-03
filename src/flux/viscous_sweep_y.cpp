/**
 * @file viscous_sweep_y.cpp
 * @brief Y-direction viscous flux divergence (BR2 Phase 2) on decoupled Cells.
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

static void compute_viscous_flux_y(const double U[4],
                                    const double dUdx[4], const double dUdy[4],
                                    double mu, double kappa, double gamma,
                                    double Gv[4], bool enable_suth, double suth_c, double pr)
{
    double rho = std::max(1e-14, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;

    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;

    double p_val = (gamma - 1.0) * (U[3] - 0.5 * rho * (u*u + v*v));
    if (p_val < 1e-14) p_val = 1e-14;
    double T = p_val / rho;

    double mu_local = mu;
    double kappa_local = kappa;
    if (enable_suth) {
        double T_norm = std::max(1e-8, T);
        mu_local = mu * (std::pow(T_norm, 1.5) * (1.0 + suth_c) / (T_norm + suth_c));
        kappa_local = mu_local * gamma / ((gamma - 1.0) * pr);
    }

    double dpdy = (gamma - 1.0) * (dUdy[3] - 0.5*(u*u+v*v)*dUdy[0]
                                    - rho*(u*dudy + v*dvdy));
    double dTdy = (dpdy - T * dUdy[0]) / rho;

    double tau_yy = mu_local * (4.0/3.0 * dvdy - 2.0/3.0 * dudx);
    double tau_xy = mu_local * (dudy + dvdx);

    double qy = -kappa_local * dTdy;

    Gv[0] = 0.0;
    Gv[1] = tau_xy;
    Gv[2] = tau_yy;
    Gv[3] = u * tau_xy + v * tau_yy - qy;
}

void Solver::viscous_sweep_y() {
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

        auto extrapolate_neighbor_y = [&](const Cell* nc, char nface, int ix, double U_nb[4], double dUdx_nb[4], double dUdy_nb[4]) {
            const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
            for (int v = 0; v < 4; ++v) {
                U_nb[v] = 0.0;
                dUdx_nb[v] = 0.0;
                dUdy_nb[v] = 0.0;
            }
            int nc_offset = nc->cell_index * 4 * p.N_PTS * p.N_PTS;
            for (int k = 0; k < p.N_PTS; ++k) {
                int flat = k * p.N_PTS + ix;
                for (int v = 0; v < 4; ++v) {
                    U_nb[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    dUdx_nb[v] += global_grad_Ux[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[k];
                    dUdy_nb[v] += global_grad_Uy[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[k];
                }
            }
        };

        int c_offset = c->cell_index * 4 * p.N_PTS * p.N_PTS;

        for (int ix = 0; ix < p.N_PTS; ++ix) {

            // --- 1. Pointwise viscous Y-flux at each solution point ---
            double Gv_sol[MAX_PTS][4];
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                int flat = iy * p.N_PTS + ix;
                double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                for (int v = 0; v < 4; ++v) {
                    U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                    dUdx_pt[v] = global_grad_Ux[c_offset + v * p.N_PTS * p.N_PTS + flat];
                    dUdy_pt[v] = global_grad_Uy[c_offset + v * p.N_PTS * p.N_PTS + flat];
                }
                compute_viscous_flux_y(U_pt, dUdx_pt, dUdy_pt,
                                        mu, kappa, p.GAMMA, Gv_sol[iy],
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
            }

            // --- 2. Face-extrapolated viscous fluxes & states ---
            double Gv_B[4] = {}, Gv_T[4] = {};
            double UB_face[4] = {}, UT_face[4] = {};
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    Gv_B[v] += Gv_sol[iy][v] * basis.l_L[iy];
                    Gv_T[v] += Gv_sol[iy][v] * basis.l_R[iy];
                    UB_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[iy];
                    UT_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[iy];
                }
            }

            // --- 3. Bottom common viscous flux (conforming/boundary/SFP) ---
            double Gv_B_local[4] = {};
            if (c->neighbors[2] && c->neighbors[2]->level == c->level) {
                Cell* nc = c->neighbors[2];
                char nface = c->neighbor_faces[2];
                double UB_nb[4], dUdx_nb[4], dUdy_nb[4], Gv_B_nb[4];
                extrapolate_neighbor_y(nc, nface, ix, UB_nb, dUdx_nb, dUdy_nb);
                compute_viscous_flux_y(UB_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Gv_B_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_B_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_B_face = 0.5 * (get_suth_mu(UB_face) + get_suth_mu(UB_nb));
                }
                double penalty_B = br2_eta * mu_B_face / c->dy;
                for (int v = 0; v < 4; ++v) {
                    Gv_B_local[v] = 0.5 * (Gv_B_nb[v] + Gv_B[v])
                                  + penalty_B * (UB_face[v] - UB_nb[v]);
                }
            } else if (c->is_boundary[2]) {
                double UB_nb[4], sig_dummy;
                get_neigh_state_cell(*c, ix, false, UB_face, 0.0, UB_nb, sig_dummy, 1);
                for (int v = 0; v < 4; ++v) {
                    Gv_B_local[v] = Gv_B[v];
                }
            }

            // --- 4. Top common viscous flux (conforming/boundary/SFP) ---
            double Gv_T_local[4] = {};
            if (c->neighbors[3] && c->neighbors[3]->level == c->level) {
                Cell* nc = c->neighbors[3];
                char nface = c->neighbor_faces[3];
                double UT_nb[4], dUdx_nb[4], dUdy_nb[4], Gv_T_nb[4];
                extrapolate_neighbor_y(nc, nface, ix, UT_nb, dUdx_nb, dUdy_nb);
                compute_viscous_flux_y(UT_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Gv_T_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_T_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_T_face = 0.5 * (get_suth_mu(UT_face) + get_suth_mu(UT_nb));
                }
                double penalty_T = br2_eta * mu_T_face / c->dy;
                for (int v = 0; v < 4; ++v) {
                    Gv_T_local[v] = 0.5 * (Gv_T[v] + Gv_T_nb[v])
                                  + penalty_T * (UT_nb[v] - UT_face[v]);
                }
            } else if (c->is_boundary[3]) {
                double UT_nb[4], sig_dummy;
                get_neigh_state_cell(*c, ix, true, UT_face, 0.0, UT_nb, sig_dummy, 1);
                for (int v = 0; v < 4; ++v) {
                    Gv_T_local[v] = Gv_T[v];
                }
            }

            // --- 5. Accumulate into RHS (single-write, extremely fast) ---
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double dg = 0.0;
                    for (int kk = 0; kk < p.N_PTS; ++kk)
                        dg += basis.D[iy][kk] * Gv_sol[kk][v];
                    
                    c->get_RHS(v, iy, ix, p.N_PTS) += 
                        (dg
                         + (Gv_B_local[v] - Gv_B[v]) * basis.dgl[iy]
                         + (Gv_T_local[v] - Gv_T[v]) * basis.dgr[iy])
                        * (2.0 / c->dy);
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

        // Bottom Face (2) non-conforming coarser neighbor
        if (c->neighbors[2] && c->neighbors[2]->level < c->level) {
            Cell* nc = c->neighbors[2];
            int nc_offset = nc->cell_index * 4 * p.N_PTS * p.N_PTS;
            char nface = c->neighbor_faces[2];
            int child_idx = c->ex & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double Gv_B[4] = {};
                double UB_face[4] = {};
                for (int v = 0; v < 4; ++v) {
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        UB_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[iy];
                    }
                }
                double Gv_sol[MAX_PTS][4];
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    int flat = iy * p.N_PTS + ix;
                    double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                    for (int v = 0; v < 4; ++v) {
                        U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * p.N_PTS * p.N_PTS + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * p.N_PTS * p.N_PTS + flat];
                    }
                    compute_viscous_flux_y(U_pt, dUdx_pt, dUdy_pt,
                                            mu, kappa, p.GAMMA, Gv_sol[iy],
                                            p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                }
                for (int v = 0; v < 4; ++v) {
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        Gv_B[v] += Gv_sol[iy][v] * basis.l_L[iy];
                    }
                }

                // Coarser neighbor extrapolation
                double UB_nb[4] = {}, dUdx_nb[4] = {}, dUdy_nb[4] = {};
                double U_coarse_face[4][MAX_PTS] = {};
                double dUdx_coarse_face[4][MAX_PTS] = {};
                double dUdy_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        int flat = ky * p.N_PTS + kx;
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                            dUdx_coarse_face[v][kx] += global_grad_Ux[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[ky];
                            dUdy_coarse_face[v][kx] += global_grad_Uy[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[ky];
                        }
                    }
                }
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = P[kx][ix];
                    for (int v = 0; v < 4; ++v) {
                        UB_nb[v] += factor * U_coarse_face[v][kx];
                        dUdx_nb[v] += factor * dUdx_coarse_face[v][kx];
                        dUdy_nb[v] += factor * dUdy_coarse_face[v][kx];
                    }
                }

                double Gv_B_nb[4] = {};
                compute_viscous_flux_y(UB_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Gv_B_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_B_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_B_face = 0.5 * (get_suth_mu(UB_face) + get_suth_mu(UB_nb));
                }
                double penalty_B = br2_eta * mu_B_face / c->dy;
                double Gv_B_comm[4];
                for (int v = 0; v < 4; ++v) {
                    Gv_B_comm[v] = 0.5 * (Gv_B_nb[v] + Gv_B[v])
                                 + penalty_B * (UB_face[v] - UB_nb[v]);
                }

                // Update fine cell local RHS
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) += Gv_B_comm[v] * basis.dgl[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict and accumulate to coarse neighbor nc Top face
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) += factor * Gv_B_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }

        // Top Face (3) non-conforming coarser neighbor
        if (c->neighbors[3] && c->neighbors[3]->level < c->level) {
            Cell* nc = c->neighbors[3];
            int nc_offset = nc->cell_index * 4 * p.N_PTS * p.N_PTS;
            char nface = c->neighbor_faces[3];
            int child_idx = c->ex & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double Gv_T[4] = {};
                double UT_face[4] = {};
                for (int v = 0; v < 4; ++v) {
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        UT_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[iy];
                    }
                }
                double Gv_sol[MAX_PTS][4];
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    int flat = iy * p.N_PTS + ix;
                    double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                    for (int v = 0; v < 4; ++v) {
                        U_pt[v]    = c->get_U(v, iy, ix, p.N_PTS);
                        dUdx_pt[v] = global_grad_Ux[c_offset + v * p.N_PTS * p.N_PTS + flat];
                        dUdy_pt[v] = global_grad_Uy[c_offset + v * p.N_PTS * p.N_PTS + flat];
                    }
                    compute_viscous_flux_y(U_pt, dUdx_pt, dUdy_pt,
                                            mu, kappa, p.GAMMA, Gv_sol[iy],
                                            p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                }
                for (int v = 0; v < 4; ++v) {
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        Gv_T[v] += Gv_sol[iy][v] * basis.l_R[iy];
                    }
                }

                // Coarser neighbor extrapolation
                double UT_nb[4] = {}, dUdx_nb[4] = {}, dUdy_nb[4] = {};
                double U_coarse_face[4][MAX_PTS] = {};
                double dUdx_coarse_face[4][MAX_PTS] = {};
                double dUdy_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        int flat = ky * p.N_PTS + kx;
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                            dUdx_coarse_face[v][kx] += global_grad_Ux[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[ky];
                            dUdy_coarse_face[v][kx] += global_grad_Uy[nc_offset + v * p.N_PTS * p.N_PTS + flat] * weights[ky];
                        }
                    }
                }
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = P[kx][ix];
                    for (int v = 0; v < 4; ++v) {
                        UT_nb[v] += factor * U_coarse_face[v][kx];
                        dUdx_nb[v] += factor * dUdx_coarse_face[v][kx];
                        dUdy_nb[v] += factor * dUdy_coarse_face[v][kx];
                    }
                }

                double Gv_T_nb[4] = {};
                compute_viscous_flux_y(UT_nb, dUdx_nb, dUdy_nb,
                                        mu, kappa, p.GAMMA, Gv_T_nb,
                                        p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_T_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_T_face = 0.5 * (get_suth_mu(UT_face) + get_suth_mu(UT_nb));
                }
                double penalty_T = br2_eta * mu_T_face / c->dy;
                double Gv_T_comm[4];
                for (int v = 0; v < 4; ++v) {
                    Gv_T_comm[v] = 0.5 * (Gv_T[v] + Gv_T_nb[v])
                                 + penalty_T * (UT_nb[v] - UT_face[v]);
                }

                // Update fine cell local RHS
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) += Gv_T_comm[v] * basis.dgr[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict and accumulate to coarse neighbor nc Bottom face
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) += factor * Gv_T_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }
    }
}
// =========================================================================
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

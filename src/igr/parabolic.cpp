/**
 * @file parabolic.cpp
 * @brief Parabolic IGR evolution — BR2 gradient + divergence operator on decoupled Cells.
 */

#include "../core/solver.hpp"
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::compute_igr_parabolic_rhs() {
    // =====================================================================
    // Phase 1: Gradient pass (q = grad(sigma))
    // =====================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // X-gradient (qx)
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            double sigL = 0, sigR = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
                double s = c->sigma_field[iy * p.N_PTS + k];
                sigL += s * basis.l_L[k];
                sigR += s * basis.l_R[k];
            }

            double dummy_neigh[4], dummy_sig_L = 0.0, dummy_sig_R = 0.0;
            double dummy_face[4] = {0,0,0,0};

            if (c->neighbors[0]) {
                Cell* nc = c->neighbors[0];
                char nface = c->neighbor_faces[0];
                const double* w = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    dummy_sig_L += nc->sigma_field[iy * p.N_PTS + k] * w[k];
                }
            } else {
                get_neigh_state_cell(*c, iy, false, dummy_face, sigL, dummy_neigh, dummy_sig_L, 0);
                if (c->boundary_info[0].is_supersonic_inflow || c->boundary_info[0].is_characteristic) {
                    dummy_sig_L = -sigL;
                }
            }

            if (c->neighbors[1]) {
                Cell* nc = c->neighbors[1];
                char nface = c->neighbor_faces[1];
                const double* w = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    dummy_sig_R += nc->sigma_field[iy * p.N_PTS + k] * w[k];
                }
            } else {
                get_neigh_state_cell(*c, iy, true, dummy_face, sigR, dummy_neigh, dummy_sig_R, 0);
                if (c->boundary_info[1].is_supersonic_inflow || c->boundary_info[1].is_characteristic) {
                    dummy_sig_R = -sigR;
                }
            }

            double sigL_hat = 0.5 * (sigL + dummy_sig_L);
            double sigR_hat = 0.5 * (sigR + dummy_sig_R);

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double ds_dx_loc = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    ds_dx_loc += basis.D[ix][k] * c->sigma_field[iy * p.N_PTS + k];
                }
                double ds_dx = (ds_dx_loc + (sigL_hat - sigL) * basis.dgl[ix] + (sigR_hat - sigR) * basis.dgr[ix]) * (2.0 / c->dx);
                double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                c->qx_buf[iy * p.N_PTS + ix] = ds_dx / rho;
            }
        }

        // Y-gradient (qy)
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            double sigB = 0, sigT = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
                double s = c->sigma_field[k * p.N_PTS + ix];
                sigB += s * basis.l_L[k];
                sigT += s * basis.l_R[k];
            }

            double dummy_neigh[4], dummy_sig_B = 0.0, dummy_sig_T = 0.0;
            double dummy_face[4] = {0,0,0,0};

            if (c->neighbors[2]) {
                Cell* nc = c->neighbors[2];
                char nface = c->neighbor_faces[2];
                const double* w = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    dummy_sig_B += nc->sigma_field[k * p.N_PTS + ix] * w[k];
                }
            } else {
                get_neigh_state_cell(*c, ix, false, dummy_face, sigB, dummy_neigh, dummy_sig_B, 1);
                if (c->boundary_info[2].is_supersonic_inflow || c->boundary_info[2].is_characteristic) {
                    dummy_sig_B = -sigB;
                }
            }

            if (c->neighbors[3]) {
                Cell* nc = c->neighbors[3];
                char nface = c->neighbor_faces[3];
                const double* w = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    dummy_sig_T += nc->sigma_field[k * p.N_PTS + ix] * w[k];
                }
            } else {
                get_neigh_state_cell(*c, ix, true, dummy_face, sigT, dummy_neigh, dummy_sig_T, 1);
                if (c->boundary_info[3].is_supersonic_inflow || c->boundary_info[3].is_characteristic) {
                    dummy_sig_T = -sigT;
                }
            }

            double sigB_hat = 0.5 * (sigB + dummy_sig_B);
            double sigT_hat = 0.5 * (sigT + dummy_sig_T);

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double ds_dy_loc = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    ds_dy_loc += basis.D[iy][k] * c->sigma_field[k * p.N_PTS + ix];
                }
                double ds_dy = (ds_dy_loc + (sigB_hat - sigB) * basis.dgl[iy] + (sigT_hat - sigT) * basis.dgr[iy]) * (2.0 / c->dy);
                double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                c->qy_buf[iy * p.N_PTS + ix] = ds_dy / rho;
            }
        }
    }

    // =====================================================================
    // Phase 2 & 3: Fused Divergence Pass + Assembly
    // =====================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        const double epsilon = p.ALPHA_SCALE * (c->dx * c->dy);
        const double br2_factor = p.IGR_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);

        // --- PRE-CALCULATE X-INTERFACE VALUES ---
        double fhatL[MAX_PTS], fhatR[MAX_PTS], fL_int[MAX_PTS], fR_int[MAX_PTS];
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            double fL = 0, fR = 0, sigL = 0, sigR = 0, rhoL = 0, rhoR = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
                int idx = iy * p.N_PTS + k;
                double r_k = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, k, p.N_PTS));
                fL += c->qx_buf[idx] * basis.l_L[k];
                fR += c->qx_buf[idx] * basis.l_R[k];
                sigL += c->sigma_field[idx] * basis.l_L[k];
                sigR += c->sigma_field[idx] * basis.l_R[k];
                rhoL += r_k * basis.l_L[k];
                rhoR += r_k * basis.l_R[k];
            }
            fL_int[iy] = fL; fR_int[iy] = fR;

            double f_nb, sig_nb, rho_nb;

            // Left interface
            if (c->neighbors[0]) {
                Cell* nc = c->neighbors[0];
                char nface = c->neighbor_faces[0];
                const double* w = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                f_nb = 0; sig_nb = 0; rho_nb = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    int idx = iy * p.N_PTS + k;
                    double r_k = std::max(p.POS_LIMITER_EPS, nc->get_U(0, iy, k, p.N_PTS));
                    f_nb += nc->qx_buf[idx] * w[k];
                    sig_nb += nc->sigma_field[idx] * w[k];
                    rho_nb += r_k * w[k];
                }
                double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, 0.5*(rhoL + rho_nb))) / c->dx;
                fhatL[iy] = 0.5*(f_nb + fL) + eta*(sigL - sig_nb);
            } else {
                const auto& ni = c->boundary_info[0];
                if (ni.is_supersonic_inflow || ni.is_characteristic) {
                    double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, rhoL)) / c->dx;
                    fhatL[iy] = fL_int[iy] + eta * sigL;
                } else {
                    fhatL[iy] = 0.0; // Neumann zero-gradient BC on physical boundaries
                }
            }

            // Right interface
            if (c->neighbors[1]) {
                Cell* nc = c->neighbors[1];
                char nface = c->neighbor_faces[1];
                const double* w = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                f_nb = 0; sig_nb = 0; rho_nb = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    int idx = iy * p.N_PTS + k;
                    double r_k = std::max(p.POS_LIMITER_EPS, nc->get_U(0, iy, k, p.N_PTS));
                    f_nb += nc->qx_buf[idx] * w[k];
                    sig_nb += nc->sigma_field[idx] * w[k];
                    rho_nb += r_k * w[k];
                }
                double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, 0.5*(rhoR + rho_nb))) / c->dx;
                fhatR[iy] = 0.5*(f_nb + fR) + eta*(sig_nb - sigR);
            } else {
                const auto& ni = c->boundary_info[1];
                if (ni.is_supersonic_inflow || ni.is_characteristic) {
                    double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, rhoR)) / c->dx;
                    fhatR[iy] = fR_int[iy] - eta * sigR;
                } else {
                    fhatR[iy] = 0.0; // Neumann zero-gradient BC on physical boundaries
                }
            }
        }

        // --- PRE-CALCULATE Y-INTERFACE VALUES ---
        double fhatB[MAX_PTS], fhatT[MAX_PTS], fB_int[MAX_PTS], fT_int[MAX_PTS];
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            double fB = 0, fT = 0, sigB = 0, sigT = 0, rhoB = 0, rhoT = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
                int idx = k * p.N_PTS + ix;
                double r_k = std::max(p.POS_LIMITER_EPS, c->get_U(0, k, ix, p.N_PTS));
                fB += c->qy_buf[idx] * basis.l_L[k];
                fT += c->qy_buf[idx] * basis.l_R[k];
                sigB += c->sigma_field[idx] * basis.l_L[k];
                sigT += c->sigma_field[idx] * basis.l_R[k];
                rhoB += r_k * basis.l_L[k];
                rhoT += r_k * basis.l_R[k];
            }
            fB_int[ix] = fB; fT_int[ix] = fT;

            double f_nb, sig_nb, rho_nb;

            // Bottom interface
            if (c->neighbors[2]) {
                Cell* nc = c->neighbors[2];
                char nface = c->neighbor_faces[2];
                const double* w = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                f_nb = 0; sig_nb = 0; rho_nb = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    int idx = k * p.N_PTS + ix;
                    double r_k = std::max(p.POS_LIMITER_EPS, nc->get_U(0, k, ix, p.N_PTS));
                    f_nb += nc->qy_buf[idx] * w[k];
                    sig_nb += nc->sigma_field[idx] * w[k];
                    rho_nb += r_k * w[k];
                }
                double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, 0.5*(rhoB + rho_nb))) / c->dy;
                fhatB[ix] = 0.5*(f_nb + fB) + eta*(sigB - sig_nb);
            } else {
                const auto& ni = c->boundary_info[2];
                if (ni.is_supersonic_inflow || ni.is_characteristic) {
                    double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, rhoB)) / c->dy;
                    fhatB[ix] = fB_int[ix] + eta * sigB;
                } else {
                    fhatB[ix] = 0.0;
                }
            }

            // Top interface
            if (c->neighbors[3]) {
                Cell* nc = c->neighbors[3];
                char nface = c->neighbor_faces[3];
                const double* w = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                f_nb = 0; sig_nb = 0; rho_nb = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    int idx = k * p.N_PTS + ix;
                    double r_k = std::max(p.POS_LIMITER_EPS, nc->get_U(0, k, ix, p.N_PTS));
                    f_nb += nc->qy_buf[idx] * w[k];
                    sig_nb += nc->sigma_field[idx] * w[k];
                    rho_nb += r_k * w[k];
                }
                double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, 0.5*(rhoT + rho_nb))) / c->dy;
                fhatT[ix] = 0.5*(f_nb + fT) + eta*(sig_nb - sigT);
            } else {
                const auto& ni = c->boundary_info[3];
                if (ni.is_supersonic_inflow || ni.is_characteristic) {
                    double eta = br2_factor * (epsilon / std::max(p.POS_LIMITER_EPS, rhoT)) / c->dy;
                    fhatT[ix] = fT_int[ix] - eta * sigT;
                } else {
                    fhatT[ix] = 0.0;
                }
            }
        }

        // --- INTERIOR DIVERGENCE + ASSEMBLY ---
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double df_dx_loc = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    int idx = iy * p.N_PTS + k;
                    df_dx_loc += basis.D[ix][k] * c->qx_buf[idx];
                }
                double div_x = (df_dx_loc + (fhatL[iy] - fL_int[iy])*basis.dgl[ix] + (fhatR[iy] - fR_int[iy])*basis.dgr[ix]) * (2.0 / c->dx);

                double df_dy_loc = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    int idx = k * p.N_PTS + ix;
                    df_dy_loc += basis.D[iy][k] * c->qy_buf[idx];
                }
                double div_y = (df_dy_loc + (fhatB[ix] - fB_int[ix])*basis.dgl[iy] + (fhatT[ix] - fT_int[ix])*basis.dgr[iy]) * (2.0 / c->dy);

                int idx = iy * p.N_PTS + ix;
                double S = c->S_buf[idx];
                double sig = c->sigma_field[idx];
                double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                c->sigma_RHS[idx] = (1.0 / p.IGR_TAU_R) * ((S - sig) + epsilon * rho * (div_x + div_y));
                if (p.ENABLE_IB && (c->solid_mask || c->ib_mask[idx] > 0.99)) {
                    c->sigma_RHS[idx] = 0.0;
                }
            }
        }
    }
}

void SolverDim<3>::step_parabolic_igr(double dt_stage_ratio) {
    if (!p.ENABLE_IGR) return;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        int npts = p.N_PTS;
        int npts3 = npts * npts * npts;

        for (int pt = 0; pt < npts3; ++pt) {
            double S = c->S_buf[pt];
            double sig = c->sigma_field[pt];
            c->sigma_RHS[pt] = (1.0 / p.IGR_TAU_R) * (S - sig);
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        double dt_stage = dt_stage_ratio * c->element_dt;
        int npts3 = p.N_PTS * p.N_PTS * p.N_PTS;

        for (int pt = 0; pt < npts3; ++pt) {
            c->sigma_field[pt] += dt_stage * c->sigma_RHS[pt];
            c->sigma_field[pt] = std::max(0.0, c->sigma_field[pt]);
        }
    }
}

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
            const ImmersedBoundary::SurrogateFluxPoint* sfp_B = ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 2, ix);
            if (sfp_B) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_B, u_sb);
                double Flux_B_comm[4];
                if (p.ENABLE_PPR) {
                    double rho_sb = std::max(p.POS_LIMITER_EPS, u_sb[0]);
                    double p_sb = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / rho_sb));
                    double S_sb = rho_sb * p_sb;
                    solve_riemann(u_sb, UB_face, Flux_B_comm, 1, S_sb, S_B_face, c->theta_avg, c->theta_avg);
                    double v_sb_n = u_sb[2] / rho_sb;
                    double v_local_n = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                    double c_sb = std::sqrt(p.GAMMA * p_sb / rho_sb);
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UB_face[3] - 0.5 * UB_face[0] * ((UB_face[1]/UB_face[0])*(UB_face[1]/UB_face[0]) + v_local_n*v_local_n)));
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_B_face / UB_face[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UB_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_sb_n) + c_sb, std::abs(v_local_n) + c_local);
                    Flux_S_B_comm = 0.5 * p.PPR_ADV_MULT * (S_sb * v_sb_n + S_B_face * v_local_n) - 0.5 * lam * (S_B_face - S_sb);
                } else {
                    solve_riemann(u_sb, UB_face, Flux_B_comm, 1);
                }
                for (int v = 0; v < 4; ++v) Flux_B_local[v] = Flux_B_comm[v];
                double vn_sb = u_sb[2] / std::max(p.POS_LIMITER_EPS, u_sb[0]);
                double vn_loc = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                Flux_B_local[2] += sig_B_face;
                Flux_B_local[3] += 0.5 * sig_B_face * (vn_sb + vn_loc);
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
                double Flux_B_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(U_neigh, UB_face, Flux_B_comm, 1, S_neigh, S_B_face, nc->theta_avg, c->theta_avg);
                    double v_neigh_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double v_local_n = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * ((U_neigh[1]/U_neigh[0])*(U_neigh[1]/U_neigh[0]) + v_neigh_n*v_neigh_n)));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UB_face[3] - 0.5 * UB_face[0] * ((UB_face[1]/UB_face[0])*(UB_face[1]/UB_face[0]) + v_local_n*v_local_n)));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_B_face / UB_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UB_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_neigh_n) + c_neigh, std::abs(v_local_n) + c_local);
                    Flux_S_B_comm = 0.5 * p.PPR_ADV_MULT * (S_neigh * v_neigh_n + S_B_face * v_local_n) - 0.5 * lam * (S_B_face - S_neigh);
                } else {
                    solve_riemann(U_neigh, UB_face, Flux_B_comm, 1);
                }
                for (int v = 0; v < 4; ++v) Flux_B_local[v] = Flux_B_comm[v];
                double vn_neigh = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                double vn_loc = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                Flux_B_local[2] += 0.5 * (sig_neigh + sig_B_face);
                Flux_B_local[3] += 0.5 * (sig_neigh * vn_neigh + sig_B_face * vn_loc);
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
                double Flux_B_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(U_neigh, UB_face, Flux_B_comm, 1, S_neigh, S_B_face, c->theta_avg, c->theta_avg);
                    double v_neigh_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double v_local_n = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * ((U_neigh[1]/U_neigh[0])*(U_neigh[1]/U_neigh[0]) + v_neigh_n*v_neigh_n)));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UB_face[3] - 0.5 * UB_face[0] * ((UB_face[1]/UB_face[0])*(UB_face[1]/UB_face[0]) + v_local_n*v_local_n)));
                    double p_neigh_reg = p_neigh + c->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_B_face / UB_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UB_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_neigh_n) + c_neigh, std::abs(v_local_n) + c_local);
                    Flux_S_B_comm = 0.5 * p.PPR_ADV_MULT * (S_neigh * v_neigh_n + S_B_face * v_local_n) - 0.5 * lam * (S_B_face - S_neigh);
                } else {
                    solve_riemann(U_neigh, UB_face, Flux_B_comm, 1);
                }
                for (int v = 0; v < 4; ++v) Flux_B_local[v] = Flux_B_comm[v];
                double vn_neigh = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                double vn_loc = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                Flux_B_local[2] += 0.5 * (sig_neigh + sig_B_face);
                Flux_B_local[3] += 0.5 * (sig_neigh * vn_neigh + sig_B_face * vn_loc);
            }

            // Top Face (3)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_T = ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 3, ix);
            if (sfp_T) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_T, u_sb);
                double Flux_T_comm[4];
                if (p.ENABLE_PPR) {
                    double rho_sb = std::max(p.POS_LIMITER_EPS, u_sb[0]);
                    double p_sb = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / rho_sb));
                    double S_sb = rho_sb * p_sb;
                    solve_riemann(UT_face, u_sb, Flux_T_comm, 1, S_T_face, S_sb, c->theta_avg, c->theta_avg);
                    double v_sb_n = u_sb[2] / rho_sb;
                    double v_local_n = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                    double c_sb = std::sqrt(p.GAMMA * p_sb / rho_sb);
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UT_face[3] - 0.5 * UT_face[0] * ((UT_face[1]/UT_face[0])*(UT_face[1]/UT_face[0]) + v_local_n*v_local_n)));
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_T_face / UT_face[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UT_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_local_n) + c_local, std::abs(v_sb_n) + c_sb);
                    Flux_S_T_comm = 0.5 * p.PPR_ADV_MULT * (S_T_face * v_local_n + S_sb * v_sb_n) - 0.5 * lam * (S_sb - S_T_face);
                } else {
                    solve_riemann(UT_face, u_sb, Flux_T_comm, 1);
                }
                for (int v = 0; v < 4; ++v) Flux_T_local[v] = Flux_T_comm[v];
                double vn_loc = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                double vn_sb = u_sb[2] / std::max(p.POS_LIMITER_EPS, u_sb[0]);
                Flux_T_local[2] += sig_T_face;
                Flux_T_local[3] += 0.5 * sig_T_face * (vn_loc + vn_sb);
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
                double Flux_T_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(UT_face, U_neigh, Flux_T_comm, 1, S_T_face, S_neigh, c->theta_avg, nc->theta_avg);
                    double v_neigh_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double v_local_n = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * ((U_neigh[1]/U_neigh[0])*(U_neigh[1]/U_neigh[0]) + v_neigh_n*v_neigh_n)));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UT_face[3] - 0.5 * UT_face[0] * ((UT_face[1]/UT_face[0])*(UT_face[1]/UT_face[0]) + v_local_n*v_local_n)));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_T_face / UT_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UT_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_local_n) + c_local, std::abs(v_neigh_n) + c_neigh);
                    Flux_S_T_comm = 0.5 * p.PPR_ADV_MULT * (S_T_face * v_local_n + S_neigh * v_neigh_n) - 0.5 * lam * (S_neigh - S_T_face);
                } else {
                    solve_riemann(UT_face, U_neigh, Flux_T_comm, 1);
                }
                for (int v = 0; v < 4; ++v) Flux_T_local[v] = Flux_T_comm[v];
                double vn_loc = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                double vn_neigh = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                Flux_T_local[2] += 0.5 * (sig_T_face + sig_neigh);
                Flux_T_local[3] += 0.5 * (sig_T_face * vn_loc + sig_neigh * vn_neigh);
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
                double Flux_T_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(UT_face, U_neigh, Flux_T_comm, 1, S_T_face, S_neigh, c->theta_avg, c->theta_avg);
                    double v_neigh_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double v_local_n = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * ((U_neigh[1]/U_neigh[0])*(U_neigh[1]/U_neigh[0]) + v_neigh_n*v_neigh_n)));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UT_face[3] - 0.5 * UT_face[0] * ((UT_face[1]/UT_face[0])*(UT_face[1]/UT_face[0]) + v_local_n*v_local_n)));
                    double p_neigh_reg = p_neigh + c->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_T_face / UT_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UT_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_local_n) + c_local, std::abs(v_neigh_n) + c_neigh);
                    Flux_S_T_comm = 0.5 * p.PPR_ADV_MULT * (S_T_face * v_local_n + S_neigh * v_neigh_n) - 0.5 * lam * (S_neigh - S_T_face);
                } else {
                    solve_riemann(UT_face, U_neigh, Flux_T_comm, 1);
                }
                for (int v = 0; v < 4; ++v) Flux_T_local[v] = Flux_T_comm[v];
                double vn_loc = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                double vn_neigh = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                Flux_T_local[2] += 0.5 * (sig_T_face + sig_neigh);
                Flux_T_local[3] += 0.5 * (sig_T_face * vn_loc + sig_neigh * vn_neigh);
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
                if (p.ENABLE_PPR) {
                    solve_riemann(U_neigh, UB_face, Flux_B_comm, 1, S_neigh, S_B_face, nc->theta_avg, c->theta_avg);
                    double v_neigh_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double v_local_n = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * ((U_neigh[1]/U_neigh[0])*(U_neigh[1]/U_neigh[0]) + v_neigh_n*v_neigh_n)));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UB_face[3] - 0.5 * UB_face[0] * ((UB_face[1]/UB_face[0])*(UB_face[1]/UB_face[0]) + v_local_n*v_local_n)));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_B_face / UB_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UB_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_neigh_n) + c_neigh, std::abs(v_local_n) + c_local);
                    Flux_S_B_comm = 0.5 * p.PPR_ADV_MULT * (S_neigh * v_neigh_n + S_B_face * v_local_n) - 0.5 * lam * (S_B_face - S_neigh);
                } else {
                    solve_riemann(U_neigh, UB_face, Flux_B_comm, 1);
                }
                double vn_neigh = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                double vn_loc = UB_face[2] / std::max(p.POS_LIMITER_EPS, UB_face[0]);
                Flux_B_comm[2] += 0.5 * (sig_neigh + sig_B_face);
                Flux_B_comm[3] += 0.5 * (sig_neigh * vn_neigh + sig_B_face * vn_loc);

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
                if (p.ENABLE_PPR) {
                    solve_riemann(UT_face, U_neigh, Flux_T_comm, 1, S_T_face, S_neigh, c->theta_avg, nc->theta_avg);
                    double v_neigh_n = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double v_local_n = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * ((U_neigh[1]/U_neigh[0])*(U_neigh[1]/U_neigh[0]) + v_neigh_n*v_neigh_n)));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UT_face[3] - 0.5 * UT_face[0] * ((UT_face[1]/UT_face[0])*(UT_face[1]/UT_face[0]) + v_local_n*v_local_n)));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_T_face / UT_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UT_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(v_local_n) + c_local, std::abs(v_neigh_n) + c_neigh);
                    Flux_S_T_comm = 0.5 * p.PPR_ADV_MULT * (S_T_face * v_local_n + S_neigh * v_neigh_n) - 0.5 * lam * (S_neigh - S_T_face);
                } else {
                    solve_riemann(UT_face, U_neigh, Flux_T_comm, 1);
                }
                double vn_loc = UT_face[2] / std::max(p.POS_LIMITER_EPS, UT_face[0]);
                double vn_neigh = U_neigh[2] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                Flux_T_comm[2] += 0.5 * (sig_T_face + sig_neigh);
                Flux_T_comm[3] += 0.5 * (sig_T_face * vn_loc + sig_neigh * vn_neigh);

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

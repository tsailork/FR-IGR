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

            // Pre-compute pressure gradients for the entire cell if needed for acoustic advection
            if (iy == 0 && p.ENABLE_PPR && p.PPR_GRAD_ADV_SCALE > 0.0) {
                double P_buf[MAX_PTS][MAX_PTS];
                for (int ty = 0; ty < p.N_PTS; ++ty) {
                    for (int tx = 0; tx < p.N_PTS; ++tx) {
                        double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, ty, tx, p.N_PTS));
                        double u = c->get_U(1, ty, tx, p.N_PTS) / rho;
                        double v = c->get_U(2, ty, tx, p.N_PTS) / rho;
                        P_buf[ty][tx] = (p.GAMMA - 1.0) * (c->get_U(3, ty, tx, p.N_PTS) - 0.5 * rho * (u * u + v * v));
                    }
                }
                for (int ty = 0; ty < p.N_PTS; ++ty) {
                    for (int tx = 0; tx < p.N_PTS; ++tx) {
                        double dP_dx = 0.0, dP_dy = 0.0;
                        for (int k = 0; k < p.N_PTS; ++k) {
                            dP_dx += basis.D[tx][k] * P_buf[ty][k];
                            dP_dy += basis.D[ty][k] * P_buf[k][tx];
                        }
                        int idx = ty * p.N_PTS + tx;
                        c->grad_px_field[idx] = dP_dx * (2.0 / c->dx);
                        c->grad_py_field[idx] = dP_dy * (2.0 / c->dy);
                    }
                }
            }

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
                    double v   = c->get_U(2, iy, ix, p.N_PTS) / rho;
                    // Part A: compute full 2D pressure gradient and store for y-sweep reuse
                    if (p.PPR_GRAD_ADV_SCALE > 0.0) {
                        int idx = iy * p.N_PTS + ix;
                        double dP_dx = c->grad_px_field[idx];
                        double dP_dy = c->grad_py_field[idx];
                        // Apply x-component of acoustic pressure-gradient correction to advection velocity
                        double press = std::max(p.POS_LIMITER_EPS,
                            (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, p.N_PTS) - 0.5*rho*(u*u+v*v)));
                        double a_loc = std::sqrt(p.GAMMA * press / rho);
                        double grad_norm = std::sqrt(dP_dx*dP_dx + dP_dy*dP_dy) + p.PPR_GRAD_EPS;
                        u += p.PPR_GRAD_ADV_SCALE * a_loc * (dP_dx / grad_norm); // +sign: push S toward high-P side
                    }
                    F_sol_S[ix] = c->S_field[iy * p.N_PTS + ix] * (p.PPR_ADV_MULT * u);
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
                double Flux_L_comm[4];
                if (p.ENABLE_PPR) {
                    double rho_sb = std::max(p.POS_LIMITER_EPS, u_sb[0]);
                    double p_sb = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / rho_sb));
                    double S_sb = rho_sb * p_sb;
                    solve_riemann(u_sb, UL_face, Flux_L_comm, 0, S_sb, S_L_face, c->theta_avg, c->theta_avg);
                    double u_sb_n = u_sb[1] / rho_sb;
                    double u_local_n = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                    double c_sb = std::sqrt(p.GAMMA * p_sb / rho_sb);
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UL_face[3] - 0.5 * UL_face[0] * (u_local_n*u_local_n + (UL_face[2]/UL_face[0])*(UL_face[2]/UL_face[0]))));
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_L_face / UL_face[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UL_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_sb_n) + c_sb, std::abs(u_local_n) + c_local);
                    Flux_S_L_comm = 0.5 * p.PPR_ADV_MULT * (S_sb * u_sb_n + S_L_face * u_local_n) - 0.5 * lam * (S_L_face - S_sb);
                } else {
                    solve_riemann(u_sb, UL_face, Flux_L_comm, 0);
                }
                for (int v = 0; v < 4; ++v) Flux_L_local[v] = Flux_L_comm[v];
                double un_sb = u_sb[1] / std::max(p.POS_LIMITER_EPS, u_sb[0]);
                double un_loc = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                Flux_L_local[1] += sig_L_face;
                Flux_L_local[3] += 0.5 * sig_L_face * (un_sb + un_loc);
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
                double Flux_L_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0, S_neigh, S_L_face, nc->theta_avg, c->theta_avg);
                    double u_neigh_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double u_local_n = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * (u_neigh_n*u_neigh_n + (U_neigh[2]/U_neigh[0])*(U_neigh[2]/U_neigh[0]))));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UL_face[3] - 0.5 * UL_face[0] * (u_local_n*u_local_n + (UL_face[2]/UL_face[0])*(UL_face[2]/UL_face[0]))));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_L_face / UL_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UL_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_neigh_n) + c_neigh, std::abs(u_local_n) + c_local);
                    Flux_S_L_comm = 0.5 * p.PPR_ADV_MULT * (S_neigh * u_neigh_n + S_L_face * u_local_n) - 0.5 * lam * (S_L_face - S_neigh);
                } else {
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0);
                }
                for (int v = 0; v < 4; ++v) Flux_L_local[v] = Flux_L_comm[v];
                double un_neigh = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                double un_loc = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                Flux_L_local[1] += 0.5 * (sig_neigh + sig_L_face);
                Flux_L_local[3] += 0.5 * (sig_neigh * un_neigh + sig_L_face * un_loc);
            } else if (c->is_boundary[0]) {
                get_neigh_state_cell(*c, iy, false,
                                     UL_face, sig_L_face, U_neigh, sig_neigh, 0);
                double S_neigh = S_L_face;
                if (p.ENABLE_PPR) {
                    double p_phan_local = S_L_face / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                    double p_phan_ghost = p_phan_local;
                    const NeighborInfo& ni = c->boundary_info[0];
                    if (ni.is_supersonic_inflow) {
                        p_phan_ghost = ni.ref_p;
                    } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                        double u_ghost_n = -U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                        if (u_ghost_n < 0.0) {
                            p_phan_ghost = ni.ref_p;
                        }
                    }
                    S_neigh = U_neigh[0] * p_phan_ghost;
                }
                double Flux_L_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0, S_neigh, S_L_face, c->theta_avg, c->theta_avg);
                    double u_neigh_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double u_local_n = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * (u_neigh_n*u_neigh_n + (U_neigh[2]/U_neigh[0])*(U_neigh[2]/U_neigh[0]))));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UL_face[3] - 0.5 * UL_face[0] * (u_local_n*u_local_n + (UL_face[2]/UL_face[0])*(UL_face[2]/UL_face[0]))));
                    double p_neigh_reg = p_neigh + c->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_L_face / UL_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UL_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_neigh_n) + c_neigh, std::abs(u_local_n) + c_local);
                    Flux_S_L_comm = 0.5 * p.PPR_ADV_MULT * (S_neigh * u_neigh_n + S_L_face * u_local_n) - 0.5 * lam * (S_L_face - S_neigh);
                } else {
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0);
                }
                for (int v = 0; v < 4; ++v) Flux_L_local[v] = Flux_L_comm[v];
                double un_neigh = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                double un_loc = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                Flux_L_local[1] += 0.5 * (sig_neigh + sig_L_face);
                Flux_L_local[3] += 0.5 * (sig_neigh * un_neigh + sig_L_face * un_loc);
            }

            // Right Face (1)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_R = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 1, iy) : nullptr;
            if (sfp_R) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_R, u_sb);
                double Flux_R_comm[4];
                if (p.ENABLE_PPR) {
                    double rho_sb = std::max(p.POS_LIMITER_EPS, u_sb[0]);
                    double p_sb = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / rho_sb));
                    double S_sb = rho_sb * p_sb;
                    solve_riemann(UR_face, u_sb, Flux_R_comm, 0, S_R_face, S_sb, c->theta_avg, c->theta_avg);
                    double u_sb_n = u_sb[1] / rho_sb;
                    double u_local_n = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                    double c_sb = std::sqrt(p.GAMMA * p_sb / rho_sb);
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UR_face[3] - 0.5 * UR_face[0] * (u_local_n*u_local_n + (UR_face[2]/UR_face[0])*(UR_face[2]/UR_face[0]))));
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_R_face / UR_face[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UR_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_local_n) + c_local, std::abs(u_sb_n) + c_sb);
                    Flux_S_R_comm = 0.5 * p.PPR_ADV_MULT * (S_R_face * u_local_n + S_sb * u_sb_n) - 0.5 * lam * (S_sb - S_R_face);
                } else {
                    solve_riemann(UR_face, u_sb, Flux_R_comm, 0);
                }
                for (int v = 0; v < 4; ++v) Flux_R_local[v] = Flux_R_comm[v];
                double un_loc = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                double un_sb = u_sb[1] / std::max(p.POS_LIMITER_EPS, u_sb[0]);
                Flux_R_local[1] += sig_R_face;
                Flux_R_local[3] += 0.5 * sig_R_face * (un_loc + un_sb);
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
                double Flux_R_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0, S_R_face, S_neigh, c->theta_avg, nc->theta_avg);
                    double u_neigh_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double u_local_n = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * (u_neigh_n*u_neigh_n + (U_neigh[2]/U_neigh[0])*(U_neigh[2]/U_neigh[0]))));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UR_face[3] - 0.5 * UR_face[0] * (u_local_n*u_local_n + (UR_face[2]/UR_face[0])*(UR_face[2]/UR_face[0]))));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_R_face / UR_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UR_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_local_n) + c_local, std::abs(u_neigh_n) + c_neigh);
                    Flux_S_R_comm = 0.5 * p.PPR_ADV_MULT * (S_R_face * u_local_n + S_neigh * u_neigh_n) - 0.5 * lam * (S_neigh - S_R_face);
                } else {
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0);
                }
                for (int v = 0; v < 4; ++v) Flux_R_local[v] = Flux_R_comm[v];
                double un_loc = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                double un_neigh = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                Flux_R_local[1] += 0.5 * (sig_R_face + sig_neigh);
                Flux_R_local[3] += 0.5 * (sig_R_face * un_loc + sig_neigh * un_neigh);
            } else if (c->is_boundary[1]) {
                get_neigh_state_cell(*c, iy, true,
                                     UR_face, sig_R_face, U_neigh, sig_neigh, 0);
                double S_neigh = S_R_face;
                if (p.ENABLE_PPR) {
                    double p_phan_local = S_R_face / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                    double p_phan_ghost = p_phan_local;
                    const NeighborInfo& ni = c->boundary_info[1];
                    if (ni.is_supersonic_inflow) {
                        p_phan_ghost = ni.ref_p;
                    } else if (ni.is_characteristic || ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
                        double u_ghost_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                        if (u_ghost_n < 0.0) {
                            p_phan_ghost = ni.ref_p;
                        }
                    }
                    S_neigh = U_neigh[0] * p_phan_ghost;
                }
                double Flux_R_comm[4];
                if (p.ENABLE_PPR) {
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0, S_R_face, S_neigh, c->theta_avg, c->theta_avg);
                    double u_neigh_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double u_local_n = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * (u_neigh_n*u_neigh_n + (U_neigh[2]/U_neigh[0])*(U_neigh[2]/U_neigh[0]))));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UR_face[3] - 0.5 * UR_face[0] * (u_local_n*u_local_n + (UR_face[2]/UR_face[0])*(UR_face[2]/UR_face[0]))));
                    double p_neigh_reg = p_neigh + c->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg * (p_local - S_R_face / UR_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UR_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_local_n) + c_local, std::abs(u_neigh_n) + c_neigh);
                    Flux_S_R_comm = 0.5 * p.PPR_ADV_MULT * (S_R_face * u_local_n + S_neigh * u_neigh_n) - 0.5 * lam * (S_neigh - S_R_face);
                } else {
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0);
                }
                for (int v = 0; v < 4; ++v) Flux_R_local[v] = Flux_R_comm[v];
                double un_loc = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                double un_neigh = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                Flux_R_local[1] += 0.5 * (sig_R_face + sig_neigh);
                Flux_R_local[3] += 0.5 * (sig_R_face * un_loc + sig_neigh * un_neigh);
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
                if (p.ENABLE_PPR) {
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0, S_neigh, S_L_face, nc->theta_avg, c->theta_avg);
                    double u_neigh_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double u_local_n = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * (u_neigh_n*u_neigh_n + (U_neigh[2]/U_neigh[0])*(U_neigh[2]/U_neigh[0]))));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UL_face[3] - 0.5 * UL_face[0] * (u_local_n*u_local_n + (UL_face[2]/UL_face[0])*(UL_face[2]/UL_face[0]))));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_L_face / UL_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UL_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_neigh_n) + c_neigh, std::abs(u_local_n) + c_local);
                    Flux_S_L_comm = 0.5 * p.PPR_ADV_MULT * (S_neigh * u_neigh_n + S_L_face * u_local_n) - 0.5 * lam * (S_L_face - S_neigh);
                } else {
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0);
                }
                double un_neigh = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                double un_loc = UL_face[1] / std::max(p.POS_LIMITER_EPS, UL_face[0]);
                Flux_L_comm[1] += 0.5 * (sig_neigh + sig_L_face);
                Flux_L_comm[3] += 0.5 * (sig_neigh * un_neigh + sig_L_face * un_loc);

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
                if (p.ENABLE_PPR) {
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0, S_R_face, S_neigh, c->theta_avg, nc->theta_avg);
                    double u_neigh_n = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                    double u_local_n = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                    double p_neigh = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (U_neigh[3] - 0.5 * U_neigh[0] * (u_neigh_n*u_neigh_n + (U_neigh[2]/U_neigh[0])*(U_neigh[2]/U_neigh[0]))));
                    double p_local = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UR_face[3] - 0.5 * UR_face[0] * (u_local_n*u_local_n + (UR_face[2]/UR_face[0])*(UR_face[2]/UR_face[0]))));
                    double p_neigh_reg = p_neigh + nc->theta_avg * (p_neigh - S_neigh / U_neigh[0]);
                    double p_local_reg = p_local + c->theta_avg  * (p_local - S_R_face / UR_face[0]);
                    double c_neigh = std::sqrt(p.GAMMA * std::max(p_neigh, p_neigh_reg) / U_neigh[0]);
                    double c_local = std::sqrt(p.GAMMA * std::max(p_local, p_local_reg) / UR_face[0]);
                    double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(u_local_n) + c_local, std::abs(u_neigh_n) + c_neigh);
                    Flux_S_R_comm = 0.5 * p.PPR_ADV_MULT * (S_R_face * u_local_n + S_neigh * u_neigh_n) - 0.5 * lam * (S_neigh - S_R_face);
                } else {
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0);
                }
                double un_loc = UR_face[1] / std::max(p.POS_LIMITER_EPS, UR_face[0]);
                double un_neigh = U_neigh[1] / std::max(p.POS_LIMITER_EPS, U_neigh[0]);
                Flux_R_comm[1] += 0.5 * (sig_R_face + sig_neigh);
                Flux_R_comm[3] += 0.5 * (sig_R_face * un_loc + sig_neigh * un_neigh);

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

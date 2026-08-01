/**
 * @file ppr.cpp
 * @brief Implementation of Hyperbolic Non-Equilibrium Phantom Pressure Relaxation (PPR) functions.
 */

#include "ppr.hpp"

namespace PPR {

void compute_element_theta_2d(const std::vector<CellDim<2>*>& cells,
                             const Basis& basis,
                             const Parameters& p)
{
    if (!p.ENABLE_PPR) return;

    const int Np = p.N_PTS;
    const double N_factor = p.P_DEG + 1; // N + 1
    const double eps = p.POS_LIMITER_EPS;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<2>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double u_buf[MAX_PTS][MAX_PTS];
        double v_buf[MAX_PTS][MAX_PTS];
        double P_phys_buf[MAX_PTS][MAX_PTS];
        double P_phan_buf[MAX_PTS][MAX_PTS];
        double div_u_buf[MAX_PTS][MAX_PTS];

        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                double rho = std::max(eps, c->get_U(0, iy, ix, Np));
                double rhou = c->get_U(1, iy, ix, Np);
                double rhov = c->get_U(2, iy, ix, Np);
                double E = c->get_U(3, iy, ix, Np);
                double S = c->S_field[iy * Np + ix];

                u_buf[iy][ix] = rhou / rho;
                v_buf[iy][ix] = rhov / rho;

                double e_kin = 0.5 * rho * (u_buf[iy][ix]*u_buf[iy][ix] + v_buf[iy][ix]*v_buf[iy][ix]);
                P_phys_buf[iy][ix] = std::max(eps, (p.GAMMA - 1.0) * (E - e_kin));
                P_phan_buf[iy][ix] = S / rho;
            }
        }

        double a_min = 1e99;
        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                double rho = std::max(eps, c->get_U(0, iy, ix, Np));
                double a_loc = std::sqrt(p.GAMMA * P_phys_buf[iy][ix] / rho);
                a_min = std::min(a_min, a_loc);
            }
        }

        // 2. Face-Integral Gauss Divergence over element boundaries (ZS-limiter immune & 2x sensitivity)
        double u_face_L_self = 0.0, u_face_R_self = 0.0;
        double v_face_B_self = 0.0, v_face_T_self = 0.0;

        for (int iy = 0; iy < Np; ++iy) {
            double u_L = 0.0, u_R = 0.0;
            for (int ix = 0; ix < Np; ++ix) {
                u_L += basis.l_L[ix] * u_buf[iy][ix];
                u_R += basis.l_R[ix] * u_buf[iy][ix];
            }
            double wy = basis.w[iy] * 0.5;
            u_face_L_self += wy * u_L;
            u_face_R_self += wy * u_R;
        }

        for (int ix = 0; ix < Np; ++ix) {
            double v_B = 0.0, v_T = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                v_B += basis.l_L[iy] * v_buf[iy][ix];
                v_T += basis.l_R[iy] * v_buf[iy][ix];
            }
            double wx = basis.w[ix] * 0.5;
            v_face_B_self += wx * v_B;
            v_face_T_self += wx * v_T;
        }

        double u_face_L_neigh = u_face_L_self;
        double u_face_R_neigh = u_face_R_self;
        double v_face_B_neigh = v_face_B_self;
        double v_face_T_neigh = v_face_T_self;

        if (c->neighbors[0] && c->neighbors[0]->level == c->level) { // Left neighbor
            CellDim<2>* nc = c->neighbors[0];
            double u_R_sum = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                double u_R = 0.0;
                for (int ix = 0; ix < Np; ++ix) {
                    double r_node = std::max(eps, nc->get_U(0, iy, ix, Np));
                    double u_node = nc->get_U(1, iy, ix, Np) / r_node;
                    u_R += basis.l_R[ix] * u_node;
                }
                u_R_sum += (basis.w[iy] * 0.5) * u_R;
            }
            u_face_L_neigh = u_R_sum;
        }

        if (c->neighbors[1] && c->neighbors[1]->level == c->level) { // Right neighbor
            CellDim<2>* nc = c->neighbors[1];
            double u_L_sum = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                double u_L = 0.0;
                for (int ix = 0; ix < Np; ++ix) {
                    double r_node = std::max(eps, nc->get_U(0, iy, ix, Np));
                    double u_node = nc->get_U(1, iy, ix, Np) / r_node;
                    u_L += basis.l_L[ix] * u_node;
                }
                u_L_sum += (basis.w[iy] * 0.5) * u_L;
            }
            u_face_R_neigh = u_L_sum;
        }

        if (c->neighbors[2] && c->neighbors[2]->level == c->level) { // Bottom neighbor
            CellDim<2>* nc = c->neighbors[2];
            double v_T_sum = 0.0;
            for (int ix = 0; ix < Np; ++ix) {
                double v_T = 0.0;
                for (int iy = 0; iy < Np; ++iy) {
                    double r_node = std::max(eps, nc->get_U(0, iy, ix, Np));
                    double v_node = nc->get_U(2, iy, ix, Np) / r_node;
                    v_T += basis.l_R[iy] * v_node;
                }
                v_T_sum += (basis.w[ix] * 0.5) * v_T;
            }
            v_face_B_neigh = v_T_sum;
        }

        if (c->neighbors[3] && c->neighbors[3]->level == c->level) { // Top neighbor
            CellDim<2>* nc = c->neighbors[3];
            double v_B_sum = 0.0;
            for (int ix = 0; ix < Np; ++ix) {
                double v_B = 0.0;
                for (int iy = 0; iy < Np; ++iy) {
                    double r_node = std::max(eps, nc->get_U(0, iy, ix, Np));
                    double v_node = nc->get_U(2, iy, ix, Np) / r_node;
                    v_B += basis.l_L[iy] * v_node;
                }
                v_B_sum += (basis.w[ix] * 0.5) * v_B;
            }
            v_face_T_neigh = v_B_sum;
        }

        double u_face_Left   = 0.5 * (u_face_L_self + u_face_L_neigh);
        double u_face_Right  = 0.5 * (u_face_R_self + u_face_R_neigh);
        double v_face_Bottom = 0.5 * (v_face_B_self + v_face_B_neigh);
        double v_face_Top    = 0.5 * (v_face_T_self + v_face_T_neigh);

        double div_face_jump = (u_face_Right - u_face_Left) / c->dx + (v_face_Top - v_face_Bottom) / c->dy;

        // 1. Spatial velocity divergence div(u) = du/dx + dv/dy & Shock-Normal Mach estimation
        double shock_val = 0.0;
        double softmax_sum = 0.0;
        double div_u_sum = 0.0;
        double P_phys_sum = 0.0;
        double P_phan_sum = 0.0;
        double rho_sum = 0.0;
        double rhou_sum = 0.0;
        double rhov_sum = 0.0;
        double E_sum = 0.0;
        double Mn_sum = 0.0;
        double theta_fs = 0.0;

        double h_eff = std::min(c->dx, c->dy);

        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                double du_dx = 0.0, dv_dy = 0.0;
                double dP_dx = 0.0, dP_dy = 0.0;
                for (int k = 0; k < Np; ++k) {
                    du_dx += basis.D[ix][k] * u_buf[iy][k];
                    dv_dy += basis.D[iy][k] * v_buf[k][ix];
                    dP_dx += basis.D[ix][k] * P_phys_buf[iy][k];
                    dP_dy += basis.D[iy][k] * P_phys_buf[k][ix];
                }
                du_dx *= (2.0 / c->dx);
                dv_dy *= (2.0 / c->dy);
                dP_dx *= (2.0 / c->dx);
                dP_dy *= (2.0 / c->dy);

                double div_u = du_dx + dv_dy;
                div_u_buf[iy][ix] = div_u;

                double div_u_eff = std::min(div_u, div_face_jump);

                double rho = std::max(eps, c->get_U(0, iy, ix, Np));
                double a_phys = std::sqrt(p.GAMMA * P_phys_buf[iy][ix] / rho);

                // Option 1: Shock-Normal Mach estimation
                double u_loc = u_buf[iy][ix];
                double v_loc = v_buf[iy][ix];
                double grad_P = std::sqrt(dP_dx * dP_dx + dP_dy * dP_dy);
                double u_n = 0.0;
                if (grad_P > 1e-10) {
                    u_n = std::abs(u_loc * dP_dx + v_loc * dP_dy) / grad_P;
                } else {
                    u_n = std::sqrt(u_loc * u_loc + v_loc * v_loc);
                }
                double M_n_loc = u_n / a_phys;

                double indicator = div_u_eff * h_eff / (N_factor * a_min);
                double ind_val = std::max(0.0, -indicator);
                //double ind_val = std::abs(indicator);

                // Option 2a: Soft-Max Indicator
                if (p.PPR_USE_SOFTMAX_INDICATOR) {
                    double ind_pow = 0.0;
                    if (p.PPR_SOFTMAX_P == 4.0) {
                        double i2 = ind_val * ind_val;
                        ind_pow = i2 * i2;
                    } else {
                        ind_pow = std::pow(ind_val, p.PPR_SOFTMAX_P);
                    }
                    softmax_sum += (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5) * ind_pow;
                } else {
                    shock_val = std::max(shock_val, ind_val);
                }

                // Quadrature weights (sum to 1.0)
                double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                div_u_sum += w * div_u;
                P_phys_sum += w * P_phys_buf[iy][ix];
                P_phan_sum += w * P_phan_buf[iy][ix];
                Mn_sum += w * M_n_loc;

                rho_sum += w * c->get_U(0, iy, ix, Np);
                rhou_sum += w * c->get_U(1, iy, ix, Np);
                rhov_sum += w * c->get_U(2, iy, ix, Np);
                E_sum += w * c->get_U(3, iy, ix, Np);
            }
        }

        if (p.PPR_USE_SOFTMAX_INDICATOR) {
            if (p.PPR_SOFTMAX_P == 4.0) {
                shock_val = std::sqrt(std::sqrt(softmax_sum));
            } else {
                shock_val = std::pow(softmax_sum, 1.0 / p.PPR_SOFTMAX_P);
            }
        }

        // 2. Element-averaged Mach numbers
        double rho_avg = std::max(eps, rho_sum);
        double u_avg = rhou_sum / rho_avg;
        double v_avg = rhov_sum / rho_avg;
        double P_avg = std::max(eps, (p.GAMMA - 1.0) * (E_sum - 0.5 * rho_avg * (u_avg*u_avg + v_avg*v_avg)));
        double a_avg = std::sqrt(p.GAMMA * P_avg / rho_avg);
        double M_avg = std::sqrt(u_avg*u_avg + v_avg*v_avg) / a_avg;
        double M_normal_avg = Mn_sum;

        double M_mach_use = (p.PPR_USE_SHOCK_NORMAL_MACH) ? M_normal_avg : M_avg;

        // Option 4: Dynamic C_tau guide rule
        double C_tau_eff = p.PPR_C_TAU;
        if (p.PPR_USE_DYNAMIC_C_TAU || p.PPR_C_TAU <= 0.0) {
            double Mn2 = M_mach_use * M_mach_use;
            C_tau_eff = (p.PPR_N_CELLS_SHOCK / (2.0 * N_factor)) * std::sqrt(1.0 + Mn2 / (1.0 + Mn2));
        }

        // 3. Raw element theta formula
        shock_val = std::max(shock_val,theta_fs);
        double theta_raw = (p.PPR_N_CELLS_SHOCK * N_factor * (p.GAMMA + 1.0) / (8.0 * C_tau_eff)) * (1.0 + M_mach_use) * shock_val;

        // 4. Thermodynamic Energy Guard
        if (theta_raw * (P_phys_sum - P_phan_sum) * div_u_sum > 0.0) {
            theta_raw = 0.0;
        }

        c->theta_max_tmp = theta_raw;
    }

    // 5. Face-Neighbor Smooth Tapering Filter
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<2>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double max_t = c->theta_max_tmp;
        for (int f = 0; f < 4; ++f) {
            if (c->neighbors[f]) {
                max_t = std::max(max_t, c->neighbors[f]->theta_max_tmp);
            }
        }
        c->theta_avg = max_t;

        // Option 2b: Sub-Cell Linear theta representation
        if (p.PPR_USE_SUBCELL_LINEAR_THETA) {
            double ax_sum = 0.0, ay_sum = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                    ax_sum += w * max_t * basis.z[ix] * 3.0;
                    ay_sum += w * max_t * basis.z[iy] * 3.0;
                }
            }
            c->theta_ax = ax_sum;
            c->theta_ay = ay_sum;
        } else {
            c->theta_ax = 0.0;
            c->theta_ay = 0.0;
        }
    }
}

void relax_phantom_pressure_2d(CellDim<2>& cell, double dt_stage, const Basis& basis, const Parameters& p) {
    if (!p.ENABLE_PPR) return;
    (void)basis;

    const int Np = p.N_PTS;
    const double N_factor = p.P_DEG + 1;
    const double eps = p.POS_LIMITER_EPS;
    const double h_eff = std::min(cell.dx, cell.dy);
    const double dx_eff = h_eff / N_factor;

    for (int iy = 0; iy < Np; ++iy) {
        for (int ix = 0; ix < Np; ++ix) {
            int k = iy * Np + ix;
            double rho = std::max(eps, cell.get_U(0, iy, ix, Np));
            double u = cell.get_U(1, iy, ix, Np) / rho;
            double v = cell.get_U(2, iy, ix, Np) / rho;
            double E = cell.get_U(3, iy, ix, Np);

            double P_phys = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));
            double a_phys = std::sqrt(p.GAMMA * P_phys / rho);
            double speed = std::sqrt(u*u + v*v);

            double C_tau_eff = p.PPR_C_TAU;
            if (p.PPR_USE_DYNAMIC_C_TAU || p.PPR_C_TAU <= 0.0) {
                double Mn_loc = speed / a_phys;
                double Mn2 = Mn_loc * Mn_loc;
                C_tau_eff = (p.PPR_N_CELLS_SHOCK / (2.0 * N_factor)) * std::sqrt(1.0 + Mn2 / (1.0 + Mn2));
            }
            double tau = C_tau_eff * dx_eff / (a_phys + speed + 1e-12);
            double exp_factor = std::exp(-dt_stage / tau);

            double S_eq = rho * P_phys;
            cell.S_field[k] = S_eq + (cell.S_field[k] - S_eq) * exp_factor;
            if (cell.S_field[k] < eps) {
                cell.S_field[k] = eps;
            }
        }
    }
}

void apply_phantom_pressure_limiter_2d(const std::vector<CellDim<2>*>& cells, const Basis& basis, const Parameters& p) {
    if (!p.ENABLE_PPR) return;
    (void)basis;

    const int Np = p.N_PTS;
    const double eps = p.POS_LIMITER_EPS;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<2>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // Construct 3-element stencil min/max for physical pressure across solution points
        double P_nodal_min = 1e30;
        double P_nodal_max = -1e30;

        auto search_cell_P = [&](const CellDim<2>* target) {
            if (!target) return;
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    double rho = std::max(eps, target->get_U(0, iy, ix, Np));
                    double u = target->get_U(1, iy, ix, Np) / rho;
                    double v = target->get_U(2, iy, ix, Np) / rho;
                    double E = target->get_U(3, iy, ix, Np);
                    double P = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));

                    P_nodal_min = std::min(P_nodal_min, P);
                    P_nodal_max = std::max(P_nodal_max, P);
                }
            }
        };

        search_cell_P(c);
        for (int f = 0; f < 4; ++f) {
            search_cell_P(c->neighbors[f]);
        }

        double theta = c->theta_avg;
        double c_pos = p.PPR_C_POS;
        double C_max = p.PPR_C_MAX;

        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                int k = iy * Np + ix;
                double rho = std::max(eps, c->get_U(0, iy, ix, Np));
                double u = c->get_U(1, iy, ix, Np) / rho;
                double v = c->get_U(2, iy, ix, Np) / rho;
                double E = c->get_U(3, iy, ix, Np);
                double P_phys = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));
                double P_phan = c->S_field[k] / rho;

                double P_phan_max = std::min(P_nodal_max, (1.0 + (1.0 - c_pos) / (theta + 1e-12)) * P_phys);
                double P_phan_min = std::max(0.0, std::min(P_nodal_min, (1.0 - (C_max - 1.0) / (theta + 1e-12)) * P_phys));

                double P_phan_clipped = std::clamp(P_phan, P_phan_min, P_phan_max);
                c->S_field[k] = rho * P_phan_clipped;
            }
        }
    }
}

void compute_element_theta_3d(const std::vector<CellDim<3>*>& cells,
                             const Basis& basis,
                             const Parameters& p)
{
    if (!p.ENABLE_PPR) return;

    const int Np = p.N_PTS;
    const double N_factor = p.P_DEG + 1;
    const double eps = p.POS_LIMITER_EPS;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<3>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double u_buf[MAX_PTS][MAX_PTS][MAX_PTS];
        double v_buf[MAX_PTS][MAX_PTS][MAX_PTS];
        double w_buf[MAX_PTS][MAX_PTS][MAX_PTS];
        double P_phys_buf[MAX_PTS][MAX_PTS][MAX_PTS];
        double P_phan_buf[MAX_PTS][MAX_PTS][MAX_PTS];

        for (int iz = 0; iz < Np; ++iz) {
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    int idx = iz * Np * Np + iy * Np + ix;
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    u_buf[iz][iy][ix] = c->get_U(1, iz, iy, ix, Np) / rho;
                    v_buf[iz][iy][ix] = c->get_U(2, iz, iy, ix, Np) / rho;
                    w_buf[iz][iy][ix] = c->get_U(3, iz, iy, ix, Np) / rho;
                    double E = c->get_U(4, iz, iy, ix, Np);
                    double S = c->S_field[idx];

                    double u = u_buf[iz][iy][ix], v = v_buf[iz][iy][ix], w = w_buf[iz][iy][ix];
                    P_phys_buf[iz][iy][ix] = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v + w*w)));
                    P_phan_buf[iz][iy][ix] = S / rho;
                }
            }
        }

        double a_min = 1e99;
        for (int iz = 0; iz < Np; ++iz) {
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double a_loc = std::sqrt(p.GAMMA * P_phys_buf[iz][iy][ix] / rho);
                    a_min = std::min(a_min, a_loc);
                }
            }
        }

        // 2. Face-Integral Gauss Divergence in 3D (ZS-limiter immune & 2x sensitivity)
        double u_face_L_self = 0.0, u_face_R_self = 0.0;
        double v_face_B_self = 0.0, v_face_T_self = 0.0;
        double w_face_F_self = 0.0, w_face_K_self = 0.0;

        for (int iz = 0; iz < Np; ++iz) {
            for (int iy = 0; iy < Np; ++iy) {
                double u_L = 0.0, u_R = 0.0;
                for (int ix = 0; ix < Np; ++ix) {
                    u_L += basis.l_L[ix] * u_buf[iz][iy][ix];
                    u_R += basis.l_R[ix] * u_buf[iz][iy][ix];
                }
                double w_area = (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5);
                u_face_L_self += w_area * u_L;
                u_face_R_self += w_area * u_R;
            }
        }

        for (int iz = 0; iz < Np; ++iz) {
            for (int ix = 0; ix < Np; ++ix) {
                double v_B = 0.0, v_T = 0.0;
                for (int iy = 0; iy < Np; ++iy) {
                    v_B += basis.l_L[iy] * v_buf[iz][iy][ix];
                    v_T += basis.l_R[iy] * v_buf[iz][iy][ix];
                }
                double w_area = (basis.w[iz] * 0.5) * (basis.w[ix] * 0.5);
                v_face_B_self += w_area * v_B;
                v_face_T_self += w_area * v_T;
            }
        }

        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                double w_F = 0.0, w_K = 0.0;
                for (int iz = 0; iz < Np; ++iz) {
                    w_F += basis.l_L[iz] * w_buf[iz][iy][ix];
                    w_K += basis.l_R[iz] * w_buf[iz][iy][ix];
                }
                double w_area = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                w_face_F_self += w_area * w_F;
                w_face_K_self += w_area * w_K;
            }
        }

        double u_face_L_neigh = u_face_L_self, u_face_R_neigh = u_face_R_self;
        double v_face_B_neigh = v_face_B_self, v_face_T_neigh = v_face_T_self;
        double w_face_F_neigh = w_face_F_self, w_face_K_neigh = w_face_K_self;

        if (c->neighbors[0] && c->neighbors[0]->level == c->level) { // Left neighbor
            CellDim<3>* nc = c->neighbors[0];
            double sum = 0.0;
            for (int iz = 0; iz < Np; ++iz) {
                for (int iy = 0; iy < Np; ++iy) {
                    double u_R = 0.0;
                    for (int ix = 0; ix < Np; ++ix) {
                        double r_node = std::max(eps, nc->get_U(0, iz, iy, ix, Np));
                        double u_node = nc->get_U(1, iz, iy, ix, Np) / r_node;
                        u_R += basis.l_R[ix] * u_node;
                    }
                    sum += (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5) * u_R;
                }
            }
            u_face_L_neigh = sum;
        }

        if (c->neighbors[1] && c->neighbors[1]->level == c->level) { // Right neighbor
            CellDim<3>* nc = c->neighbors[1];
            double sum = 0.0;
            for (int iz = 0; iz < Np; ++iz) {
                for (int iy = 0; iy < Np; ++iy) {
                    double u_L = 0.0;
                    for (int ix = 0; ix < Np; ++ix) {
                        double r_node = std::max(eps, nc->get_U(0, iz, iy, ix, Np));
                        double u_node = nc->get_U(1, iz, iy, ix, Np) / r_node;
                        u_L += basis.l_L[ix] * u_node;
                    }
                    sum += (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5) * u_L;
                }
            }
            u_face_R_neigh = sum;
        }

        if (c->neighbors[2] && c->neighbors[2]->level == c->level) { // Bottom neighbor
            CellDim<3>* nc = c->neighbors[2];
            double sum = 0.0;
            for (int iz = 0; iz < Np; ++iz) {
                for (int ix = 0; ix < Np; ++ix) {
                    double v_T = 0.0;
                    for (int iy = 0; iy < Np; ++iy) {
                        double r_node = std::max(eps, nc->get_U(0, iz, iy, ix, Np));
                        double v_node = nc->get_U(2, iz, iy, ix, Np) / r_node;
                        v_T += basis.l_R[iy] * v_node;
                    }
                    sum += (basis.w[iz] * 0.5) * (basis.w[ix] * 0.5) * v_T;
                }
            }
            v_face_B_neigh = sum;
        }

        if (c->neighbors[3] && c->neighbors[3]->level == c->level) { // Top neighbor
            CellDim<3>* nc = c->neighbors[3];
            double sum = 0.0;
            for (int iz = 0; iz < Np; ++iz) {
                for (int ix = 0; ix < Np; ++ix) {
                    double v_B = 0.0;
                    for (int iy = 0; iy < Np; ++iy) {
                        double r_node = std::max(eps, nc->get_U(0, iz, iy, ix, Np));
                        double v_node = nc->get_U(2, iz, iy, ix, Np) / r_node;
                        v_B += basis.l_L[iy] * v_node;
                    }
                    sum += (basis.w[iz] * 0.5) * (basis.w[ix] * 0.5) * v_B;
                }
            }
            v_face_T_neigh = sum;
        }

        if (c->neighbors[4] && c->neighbors[4]->level == c->level) { // Front neighbor
            CellDim<3>* nc = c->neighbors[4];
            double sum = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    double w_K = 0.0;
                    for (int iz = 0; iz < Np; ++iz) {
                        double r_node = std::max(eps, nc->get_U(0, iz, iy, ix, Np));
                        double w_node = nc->get_U(3, iz, iy, ix, Np) / r_node;
                        w_K += basis.l_R[iz] * w_node;
                    }
                    sum += (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5) * w_K;
                }
            }
            w_face_F_neigh = sum;
        }

        if (c->neighbors[5] && c->neighbors[5]->level == c->level) { // Back neighbor
            CellDim<3>* nc = c->neighbors[5];
            double sum = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    double w_F = 0.0;
                    for (int iz = 0; iz < Np; ++iz) {
                        double r_node = std::max(eps, nc->get_U(0, iz, iy, ix, Np));
                        double w_node = nc->get_U(3, iz, iy, ix, Np) / r_node;
                        w_F += basis.l_L[iz] * w_node;
                    }
                    sum += (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5) * w_F;
                }
            }
            w_face_K_neigh = sum;
        }

        double u_face_Left   = 0.5 * (u_face_L_self + u_face_L_neigh);
        double u_face_Right  = 0.5 * (u_face_R_self + u_face_R_neigh);
        double v_face_Bottom = 0.5 * (v_face_B_self + v_face_B_neigh);
        double v_face_Top    = 0.5 * (v_face_T_self + v_face_T_neigh);
        double w_face_Front  = 0.5 * (w_face_F_self + w_face_F_neigh);
        double w_face_Back   = 0.5 * (w_face_K_self + w_face_K_neigh);

        double div_face_jump = (u_face_Right - u_face_Left)/c->dx + (v_face_Top - v_face_Bottom)/c->dy + (w_face_Back - w_face_Front)/c->dz;

        double shock_val = 0.0;
        double softmax_sum = 0.0;
        double div_u_sum = 0.0;
        double P_phys_sum = 0.0;
        double P_phan_sum = 0.0;
        double rho_sum = 0.0, rhou_sum = 0.0, rhov_sum = 0.0, rhow_sum = 0.0, E_sum = 0.0;
        double Mn_sum = 0.0;
        double theta_min = 1.0;

        double h_eff = std::min({c->dx, c->dy, c->dz});

        for (int iz = 0; iz < Np; ++iz) {
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    double du_dx = 0.0, dv_dy = 0.0, dw_dz = 0.0;
                    double dP_dx = 0.0, dP_dy = 0.0, dP_dz = 0.0;
                    for (int k = 0; k < Np; ++k) {
                        du_dx += basis.D[ix][k] * u_buf[iz][iy][k];
                        dv_dy += basis.D[iy][k] * v_buf[iz][k][ix];
                        dw_dz += basis.D[iz][k] * w_buf[k][iy][ix];

                        dP_dx += basis.D[ix][k] * P_phys_buf[iz][iy][k];
                        dP_dy += basis.D[iy][k] * P_phys_buf[iz][k][ix];
                        dP_dz += basis.D[iz][k] * P_phys_buf[k][iy][ix];
                    }
                    du_dx *= (2.0 / c->dx);
                    dv_dy *= (2.0 / c->dy);
                    dw_dz *= (2.0 / c->dz);
                    dP_dx *= (2.0 / c->dx);
                    dP_dy *= (2.0 / c->dy);
                    dP_dz *= (2.0 / c->dz);
                    double div_u = du_dx + dv_dy + dw_dz;
                    double div_u_eff = std::min(div_u, div_face_jump);

                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double u = u_buf[iz][iy][ix];
                    double v = v_buf[iz][iy][ix];
                    double w_vel = w_buf[iz][iy][ix];
                    double E = c->get_U(4, iz, iy, ix, Np);
                    double P_phys = P_phys_buf[iz][iy][ix];
                    double P_phan = P_phan_buf[iz][iy][ix];
                    double a_phys = std::sqrt(p.GAMMA * P_phys / rho);

                    // Option 1: Shock-Normal Mach estimation in 3D
                    double grad_P = std::sqrt(dP_dx * dP_dx + dP_dy * dP_dy + dP_dz * dP_dz);
                    double u_n = 0.0;
                    if (grad_P > 1e-10) {
                        u_n = std::abs(u * dP_dx + v * dP_dy + w_vel * dP_dz) / grad_P;
                    } else {
                        u_n = std::sqrt(u*u + v*v + w_vel*w_vel);
                    }
                    double M_n_loc = u_n / a_phys;

                    double indicator = div_u_eff * h_eff / (N_factor * a_min);
                    double ind_val = std::max(0.0, -indicator);

                    // Option 2a: Soft-Max Indicator
                    double weight = (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                    if (p.PPR_USE_SOFTMAX_INDICATOR) {
                        double ind_pow = 0.0;
                        if (p.PPR_SOFTMAX_P == 4.0) {
                            double i2 = ind_val * ind_val;
                            ind_pow = i2 * i2;
                        } else {
                            ind_pow = std::pow(ind_val, p.PPR_SOFTMAX_P);
                        }
                        softmax_sum += weight * ind_pow;
                    } else {
                        shock_val = std::max(shock_val, ind_val);
                    }

                    div_u_sum += weight * div_u;
                    P_phys_sum += weight * P_phys;
                    P_phan_sum += weight * P_phan;
                    Mn_sum += weight * M_n_loc;

                    rho_sum += weight * rho;
                    rhou_sum += weight * (rho * u);
                    rhov_sum += weight * (rho * v);
                    rhow_sum += weight * (rho * w_vel);
                    E_sum += weight * E;
                }
            }
        }

        if (p.PPR_USE_SOFTMAX_INDICATOR) {
            if (p.PPR_SOFTMAX_P == 4.0) {
                shock_val = std::sqrt(std::sqrt(softmax_sum));
            } else {
                shock_val = std::pow(softmax_sum, 1.0 / p.PPR_SOFTMAX_P);
            }
        }

        double rho_avg = std::max(eps, rho_sum);
        double u_avg = rhou_sum / rho_avg;
        double v_avg = rhov_sum / rho_avg;
        double w_avg = rhow_sum / rho_avg;
        double P_avg = std::max(eps, (p.GAMMA - 1.0) * (E_sum - 0.5 * rho_avg * (u_avg*u_avg + v_avg*v_avg + w_avg*w_avg)));
        double a_avg = std::sqrt(p.GAMMA * P_avg / rho_avg);
        double M_avg = std::sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg) / a_avg;
        double M_normal_avg = Mn_sum;

        double M_mach_use = (p.PPR_USE_SHOCK_NORMAL_MACH) ? M_normal_avg : M_avg;

        // Option 4: Dynamic C_tau guide rule
        double C_tau_eff = p.PPR_C_TAU;
        if (p.PPR_USE_DYNAMIC_C_TAU || p.PPR_C_TAU <= 0.0) {
            double Mn2 = M_mach_use * M_mach_use;
            C_tau_eff = (p.PPR_N_CELLS_SHOCK / (2.0 * N_factor)) * std::sqrt(1.0 + Mn2 / (1.0 + Mn2));
        }

        double theta_raw = theta_min + (p.PPR_N_CELLS_SHOCK * N_factor * (p.GAMMA + 1.0) / (8.0 * C_tau_eff)) * (1.0 + M_mach_use) * shock_val;
        if (theta_raw * (P_phys_sum - P_phan_sum) * div_u_sum > 0.0) {
            theta_raw = 0.0;
        }

        c->theta_max_tmp = theta_raw;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<3>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double max_t = c->theta_max_tmp;
        for (int f = 0; f < 6; ++f) {
            if (c->neighbors[f]) {
                max_t = std::max(max_t, 0.5 * c->neighbors[f]->theta_max_tmp);
            }
        }
        c->theta_avg = max_t;

        // Option 2b: Sub-Cell Linear theta representation in 3D
        if (p.PPR_USE_SUBCELL_LINEAR_THETA) {
            double ax_sum = 0.0, ay_sum = 0.0, az_sum = 0.0;
            for (int iz = 0; iz < Np; ++iz) {
                for (int iy = 0; iy < Np; ++iy) {
                    for (int ix = 0; ix < Np; ++ix) {
                        double weight = (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                        ax_sum += weight * max_t * basis.z[ix] * 3.0;
                        ay_sum += weight * max_t * basis.z[iy] * 3.0;
                        az_sum += weight * max_t * basis.z[iz] * 3.0;
                    }
                }
            }
            c->theta_ax = ax_sum;
            c->theta_ay = ay_sum;
            c->theta_az = az_sum;
        } else {
            c->theta_ax = 0.0;
            c->theta_ay = 0.0;
            c->theta_az = 0.0;
        }
    }
}

void relax_phantom_pressure_3d(CellDim<3>& cell, double dt_stage, const Basis& basis, const Parameters& p) {
    if (!p.ENABLE_PPR) return;
    (void)basis;

    const int Np = p.N_PTS;
    const double N_factor = p.P_DEG + 1;
    const double eps = p.POS_LIMITER_EPS;
    const double h_eff = std::min({cell.dx, cell.dy, cell.dz});
    const double dx_eff = h_eff / N_factor;

    for (int iz = 0; iz < Np; ++iz) {
        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                int k = iz * Np * Np + iy * Np + ix;
                double rho = std::max(eps, cell.get_U(0, iz, iy, ix, Np));
                double u = cell.get_U(1, iz, iy, ix, Np) / rho;
                double v = cell.get_U(2, iz, iy, ix, Np) / rho;
                double w = cell.get_U(3, iz, iy, ix, Np) / rho;
                double E = cell.get_U(4, iz, iy, ix, Np);

                double P_phys = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v + w*w)));
                double a_phys = std::sqrt(p.GAMMA * P_phys / rho);
                double speed = std::sqrt(u*u + v*v + w*w);
                double C_tau_eff = p.PPR_C_TAU;
                if (p.PPR_USE_DYNAMIC_C_TAU || p.PPR_C_TAU <= 0.0) {
                    double Mn_loc = speed / a_phys;
                    double Mn2 = Mn_loc * Mn_loc;
                    C_tau_eff = (p.PPR_N_CELLS_SHOCK / (2.0 * N_factor)) * std::sqrt(1.0 + Mn2 / (1.0 + Mn2));
                }
                double tau = C_tau_eff * dx_eff / (a_phys + speed + 1e-12);
                double exp_factor = std::exp(-dt_stage / tau);

                double S_eq = rho * P_phys;
                cell.S_field[k] = S_eq + (cell.S_field[k] - S_eq) * exp_factor;
                if (cell.S_field[k] < eps) {
                    cell.S_field[k] = eps;
                }
            }
        }
    }
}

void apply_phantom_pressure_limiter_3d(const std::vector<CellDim<3>*>& cells, const Basis& basis, const Parameters& p) {
    if (!p.ENABLE_PPR) return;
    (void)basis;

    const int Np = p.N_PTS;
    const double eps = p.POS_LIMITER_EPS;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<3>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double P_nodal_min = 1e30;
        double P_nodal_max = -1e30;

        auto search_cell_P = [&](const CellDim<3>* target) {
            if (!target) return;
            for (int iz = 0; iz < Np; ++iz) {
                for (int iy = 0; iy < Np; ++iy) {
                    for (int ix = 0; ix < Np; ++ix) {
                        double rho = std::max(eps, target->get_U(0, iz, iy, ix, Np));
                        double u = target->get_U(1, iz, iy, ix, Np) / rho;
                        double v = target->get_U(2, iz, iy, ix, Np) / rho;
                        double w = target->get_U(3, iz, iy, ix, Np) / rho;
                        double E = target->get_U(4, iz, iy, ix, Np);
                        double P = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v + w*w)));

                        P_nodal_min = std::min(P_nodal_min, P);
                        P_nodal_max = std::max(P_nodal_max, P);
                    }
                }
            }
        };

        search_cell_P(c);
        for (int f = 0; f < 6; ++f) {
            search_cell_P(c->neighbors[f]);
        }

        double theta = c->theta_avg;
        double c_pos = p.PPR_C_POS;
        double C_max = p.PPR_C_MAX;

        for (int iz = 0; iz < Np; ++iz) {
            for (int iy = 0; iy < Np; ++iy) {
                for (int ix = 0; ix < Np; ++ix) {
                    int k = iz * Np * Np + iy * Np + ix;
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double u = c->get_U(1, iz, iy, ix, Np) / rho;
                    double v = c->get_U(2, iz, iy, ix, Np) / rho;
                    double w = c->get_U(3, iz, iy, ix, Np) / rho;
                    double E = c->get_U(4, iz, iy, ix, Np);
                    double P_phys = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v + w*w)));
                    double P_phan = c->S_field[k] / rho;

                    double P_phan_max = std::min(P_nodal_max, (1.0 + (1.0 - c_pos) / (theta + 1e-12)) * P_phys);
                    double P_phan_min = std::max(0.0, std::min(P_nodal_min, (1.0 - (C_max - 1.0) / (theta + 1e-12)) * P_phys));

                    double P_phan_clipped = std::clamp(P_phan, P_phan_min, P_phan_max);
                    c->S_field[k] = rho * P_phan_clipped;
                }
            }
        }
    }
}

} // namespace PPR

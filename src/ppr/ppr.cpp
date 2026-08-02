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

    // Pass 1: Local self-face pre-computation (100% cache-local parallel pass)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<2>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double u_L_self = 0.0, u_R_self = 0.0;
        for (int iy = 0; iy < Np; ++iy) {
            double u_L = 0.0, u_R = 0.0;
            for (int ix = 0; ix < Np; ++ix) {
                double rho = std::max(eps, c->get_U(0, iy, ix, Np));
                double u_node = c->get_U(1, iy, ix, Np) / rho;
                u_L += basis.l_L[ix] * u_node;
                u_R += basis.l_R[ix] * u_node;
            }
            double wy = basis.w[iy] * 0.5;
            u_L_self += wy * u_L;
            u_R_self += wy * u_R;
        }

        double v_B_self = 0.0, v_T_self = 0.0;
        for (int ix = 0; ix < Np; ++ix) {
            double v_B = 0.0, v_T = 0.0;
            for (int iy = 0; iy < Np; ++iy) {
                double rho = std::max(eps, c->get_U(0, iy, ix, Np));
                double v_node = c->get_U(2, iy, ix, Np) / rho;
                v_B += basis.l_L[iy] * v_node;
                v_T += basis.l_R[iy] * v_node;
            }
            double wx = basis.w[ix] * 0.5;
            v_B_self += wx * v_B;
            v_T_self += wx * v_T;
        }

        c->face_u_L = u_L_self;
        c->face_u_R = u_R_self;
        c->face_v_B = v_B_self;
        c->face_v_T = v_T_self;
    }

    // Pass 2: Theta computation using O(1) scalar neighbor lookups
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<2>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double u_buf[MAX_PTS][MAX_PTS];
        double v_buf[MAX_PTS][MAX_PTS];
        double P_phys_buf[MAX_PTS][MAX_PTS];
        double P_phan_buf[MAX_PTS][MAX_PTS];

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

        // Fast O(1) scalar face lookups (ZERO neighbor loops!)
        double u_face_L_self = c->face_u_L;
        double u_face_R_self = c->face_u_R;
        double v_face_B_self = c->face_v_B;
        double v_face_T_self = c->face_v_T;

        double u_face_L_neigh = (c->neighbors[0] && c->neighbors[0]->level == c->level) ? c->neighbors[0]->face_u_R : u_face_L_self;
        double u_face_R_neigh = (c->neighbors[1] && c->neighbors[1]->level == c->level) ? c->neighbors[1]->face_u_L : u_face_R_self;
        double v_face_B_neigh = (c->neighbors[2] && c->neighbors[2]->level == c->level) ? c->neighbors[2]->face_v_T : v_face_B_self;
        double v_face_T_neigh = (c->neighbors[3] && c->neighbors[3]->level == c->level) ? c->neighbors[3]->face_v_B : v_face_T_self;

        double u_face_Left   = 0.5 * (u_face_L_self + u_face_L_neigh);
        double u_face_Right  = 0.5 * (u_face_R_self + u_face_R_neigh);
        double v_face_Bottom = 0.5 * (v_face_B_self + v_face_B_neigh);
        double v_face_Top    = 0.5 * (v_face_T_self + v_face_T_neigh);

        double div_face_jump = (u_face_Right - u_face_Left) / c->dx + (v_face_Top - v_face_Bottom) / c->dy;

        // Strategy 2: Un-Limited Face Riemann Jump Sensor (100% P0-Immune)
        double jump_L = std::max(0.0, u_face_L_neigh - u_face_L_self);
        double jump_R = std::max(0.0, u_face_R_self - u_face_R_neigh);
        double jump_B = std::max(0.0, v_face_B_neigh - v_face_B_self);
        double jump_T = std::max(0.0, v_face_T_self - v_face_T_neigh);
        double max_face_jump = std::max({jump_L, jump_R, jump_B, jump_T});
        double riemann_jump_ind = max_face_jump / (N_factor * a_min);

        // 1. Spatial velocity divergence div(u) = du/dx + dv/dy & Shock-Normal Mach estimation
        double shock_val = 0.0;
        double div_u_sum = 0.0;
        double rho_sum = 0.0;
        double rhou_sum = 0.0;
        double rhov_sum = 0.0;
        double E_sum = 0.0;
        double Mn_sum = 0.0;
        double curl_u_sum = 0.0;
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
                ind_val = std::max(ind_val, riemann_jump_ind);
                //double ind_val = std::abs(indicator);

                shock_val = std::max(shock_val, ind_val);

                // Compute vorticity for Ducros filter
                double du_dy = 0.0, dv_dx = 0.0;
                for (int k = 0; k < Np; ++k) {
                    du_dy += basis.D[iy][k] * u_buf[k][ix];
                    dv_dx += basis.D[ix][k] * v_buf[iy][k];
                }
                du_dy *= (2.0 / c->dy);
                dv_dx *= (2.0 / c->dx);
                double vort = dv_dx - du_dy;

                // Quadrature weights (sum to 1.0)
                double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                div_u_sum += w * div_u;
                Mn_sum += w * M_n_loc;
                curl_u_sum += w * (vort * vort);

                rho_sum += w * c->get_U(0, iy, ix, Np);
                rhou_sum += w * c->get_U(1, iy, ix, Np);
                rhov_sum += w * c->get_U(2, iy, ix, Np);
                E_sum += w * c->get_U(3, iy, ix, Np);
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

        // 1. Refined Kinematic Sensor I_shock = Ducros * S(phi_eff)
        double h_node = h_eff / N_factor;
        double phi_comp = -h_node * std::min(0.0, div_u_sum) / (a_avg + 1e-12);
        double phi_face = theta_fs;
        double phi_eff = std::max({shock_val, phi_comp, phi_face});

        double Ducros_ratio = 1.0;
        if (p.PPR_USE_DUCROS_SENSOR) {
            double div_sq = div_u_sum * div_u_sum;
            Ducros_ratio = div_sq / (div_sq + curl_u_sum + 1e-12);
        }

        double noise_floor = p.PPR_SENSOR_NOISE_FLOOR;
        double saturation = p.PPR_SENSOR_SATURATION;
        double s_range = std::max(1e-6, saturation - noise_floor);
        double s_phi = std::min(1.0, std::max(0.0, (phi_eff - noise_floor) / s_range));
        double I_shock = Ducros_ratio * s_phi;

        // 2. Adaptive Relaxation Timescale Ratio C_tau & Master Law Coupling Intensity theta_e
        if (p.PPR_CONSTANT_MODE) {
            // Constant-mode: user-specified uniform theta and C_tau everywhere
            c->C_tau_cell = p.PPR_CONSTANT_C_TAU_VAL;
            c->theta_max_tmp = p.PPR_CONSTANT_THETA;
        } else {
        double C_tau_0 = (p.PPR_C_TAU != 0.0) ? std::abs(p.PPR_C_TAU) : 0.25;
        double C_tau_base = C_tau_0;// * std::max(0.5, p.PPR_N_CELLS_SHOCK);
        double theta_target = (p.PPR_N_CELLS_SHOCK * N_factor * (p.GAMMA + 1.0) / (8.0 * std::max(0.01, C_tau_0))) * (1.0 + M_mach_use) * phi_eff;
        double theta_e = theta_target * I_shock;

        // 3. Three-State Relaxation Controller & Von Neumann Ceiling
        double C_tau_vonNeumann = C_tau_base;
        if (p.PPR_USE_VON_NEUMANN_CEILING) {
            C_tau_vonNeumann = std::min(C_tau_base, 0.90 / (theta_e + 1.0));
        }
        double C_tau_blend = (1.0 - I_shock) * C_tau_vonNeumann + I_shock * C_tau_base;

        c->C_tau_cell = C_tau_blend;
        c->theta_max_tmp = theta_e;
        } // end adaptive mode
    }

    if (p.PPR_CONSTANT_MODE) {
        // Constant mode: just copy theta_max_tmp -> theta_avg, no expansion needed
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            CellDim<2>* c = cells[i];
            if (p.ENABLE_MULTIRATE && !c->element_active) continue;
            c->theta_avg = c->theta_max_tmp;
            c->theta_ax = 0.0;
            c->theta_ay = 0.0;
        }
    } else {
    // 4. Face-Neighbor Smooth Tapering Filter
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<2>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double max_t = c->theta_max_tmp;
        double max_c_tau = c->C_tau_cell;
        if (p.PPR_USE_STENCIL_EXPANSION) {
            for (int f = 0; f < 4; ++f) {
                if (c->neighbors[f]) {
                    max_t = std::max(max_t, c->neighbors[f]->theta_max_tmp);
                    max_c_tau = std::max(max_c_tau, c->neighbors[f]->C_tau_cell);
                }
            }
        }
        c->theta_avg = max_t;
        c->C_tau_cell = max_c_tau;
        c->theta_ax = 0.0;
        c->theta_ay = 0.0;
    }
    } // end adaptive stencil expansion
}

void relax_phantom_pressure_2d(CellDim<2>& cell, double dt_stage, const Basis& basis, const Parameters& p) {
    if (!p.ENABLE_PPR) return;
    (void)basis;

    const int Np = p.N_PTS;
    const double N_factor = p.P_DEG + 1;
    const double eps = p.POS_LIMITER_EPS;
    const double h_eff = std::min(cell.dx, cell.dy) / N_factor;

    for (int iy = 0; iy < Np; ++iy) {
        for (int ix = 0; ix < Np; ++ix) {
            int k = iy * Np + ix;
            double rho = std::max(eps, cell.get_U(0, iy, ix, Np));
            double u = cell.get_U(1, iy, ix, Np) / rho;
            double v = cell.get_U(2, iy, ix, Np) / rho;
            double E = cell.get_U(3, iy, ix, Np);

            double P_phys = std::max(eps, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));
            double P_phan = cell.S_field[k] / rho;
            double div_u = (cell.get_U(1, iy, ix, Np) - cell.get_U(1, iy, std::max(0, ix-1), Np)) / h_eff;

            // Thermodynamic Instant-Thermalization Guard
            bool anti_dissipative = (cell.theta_avg * (P_phys - P_phan) * div_u > 0.0) ||
                                    (div_u < 0.0 && P_phan > P_phys);
            if (p.PPR_USE_ENERGY_GUARD && anti_dissipative) {
                // Instantly thermalize P_phan -> P_phys in 1 RK stage
                double S_eq = rho * P_phys;
                cell.S_field[k] = S_eq;
                continue;
            }

            double a_phys = std::sqrt(p.GAMMA * P_phys / rho);
            double speed = std::sqrt(u*u + v*v);

            double C_tau_eff = cell.C_tau_cell;
            double tau = C_tau_eff * h_eff / (a_phys + speed + 1e-12);
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
    if (!p.PPR_USE_LIMITER) return;
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

                double P_phan_max = (1.0 + (1.0 - c_pos) / (theta + 1e-12)) * P_phys;
                double P_phan_min = std::max(0.0, (1.0 - (C_max - 1.0) / (theta + 1e-12)) * P_phys);

                if (p.PPR_USE_SPATIAL_CLAMP) {
                    P_phan_max = std::min(P_nodal_max, P_phan_max);
                    P_phan_min = std::min(P_nodal_min, P_phan_min);
                }

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

    // Pass 1: Local self-face pre-computation in 3D (100% cache-local parallel pass)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<3>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double u_L_self = 0.0, u_R_self = 0.0;
        for (int iz = 0; iz < Np; ++iz) {
            for (int iy = 0; iy < Np; ++iy) {
                double u_L = 0.0, u_R = 0.0;
                for (int ix = 0; ix < Np; ++ix) {
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double u_node = c->get_U(1, iz, iy, ix, Np) / rho;
                    u_L += basis.l_L[ix] * u_node;
                    u_R += basis.l_R[ix] * u_node;
                }
                double w_area = (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5);
                u_L_self += w_area * u_L;
                u_R_self += w_area * u_R;
            }
        }

        double v_B_self = 0.0, v_T_self = 0.0;
        for (int iz = 0; iz < Np; ++iz) {
            for (int ix = 0; ix < Np; ++ix) {
                double v_B = 0.0, v_T = 0.0;
                for (int iy = 0; iy < Np; ++iy) {
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double v_node = c->get_U(2, iz, iy, ix, Np) / rho;
                    v_B += basis.l_L[iy] * v_node;
                    v_T += basis.l_R[iy] * v_node;
                }
                double w_area = (basis.w[iz] * 0.5) * (basis.w[ix] * 0.5);
                v_B_self += w_area * v_B;
                v_T_self += w_area * v_T;
            }
        }

        double w_F_self = 0.0, w_K_self = 0.0;
        for (int iy = 0; iy < Np; ++iy) {
            for (int ix = 0; ix < Np; ++ix) {
                double w_F = 0.0, w_K = 0.0;
                for (int iz = 0; iz < Np; ++iz) {
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double w_node = c->get_U(3, iz, iy, ix, Np) / rho;
                    w_F += basis.l_L[iz] * w_node;
                    w_K += basis.l_R[iz] * w_node;
                }
                double w_area = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                w_F_self += w_area * w_F;
                w_K_self += w_area * w_K;
            }
        }

        c->face_u_L = u_L_self; c->face_u_R = u_R_self;
        c->face_v_B = v_B_self; c->face_v_T = v_T_self;
        c->face_w_F = w_F_self; c->face_w_K = w_K_self;
    }

    // Pass 2: Theta computation using O(1) scalar neighbor lookups in 3D
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
                    double rho = std::max(eps, c->get_U(0, iz, iy, ix, Np));
                    double rhou = c->get_U(1, iz, iy, ix, Np);
                    double rhov = c->get_U(2, iz, iy, ix, Np);
                    double rhow = c->get_U(3, iz, iy, ix, Np);
                    double E    = c->get_U(4, iz, iy, ix, Np);
                    double S    = c->S_field[iz * Np * Np + iy * Np + ix];

                    u_buf[iz][iy][ix] = rhou / rho;
                    v_buf[iz][iy][ix] = rhov / rho;
                    w_buf[iz][iy][ix] = rhow / rho;

                    double e_kin = 0.5 * rho * (u_buf[iz][iy][ix]*u_buf[iz][iy][ix] +
                                                v_buf[iz][iy][ix]*v_buf[iz][iy][ix] +
                                                w_buf[iz][iy][ix]*w_buf[iz][iy][ix]);
                    P_phys_buf[iz][iy][ix] = std::max(eps, (p.GAMMA - 1.0) * (E - e_kin));
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

        // Fast O(1) scalar face lookups in 3D (ZERO neighbor loops!)
        double u_face_L_self = c->face_u_L, u_face_R_self = c->face_u_R;
        double v_face_B_self = c->face_v_B, v_face_T_self = c->face_v_T;
        double w_face_F_self = c->face_w_F, w_face_K_self = c->face_w_K;

        double u_face_L_neigh = (c->neighbors[0] && c->neighbors[0]->level == c->level) ? c->neighbors[0]->face_u_R : u_face_L_self;
        double u_face_R_neigh = (c->neighbors[1] && c->neighbors[1]->level == c->level) ? c->neighbors[1]->face_u_L : u_face_R_self;
        double v_face_B_neigh = (c->neighbors[2] && c->neighbors[2]->level == c->level) ? c->neighbors[2]->face_v_T : v_face_B_self;
        double v_face_T_neigh = (c->neighbors[3] && c->neighbors[3]->level == c->level) ? c->neighbors[3]->face_v_B : v_face_T_self;
        double w_face_F_neigh = (c->neighbors[4] && c->neighbors[4]->level == c->level) ? c->neighbors[4]->face_w_K : w_face_F_self;
        double w_face_K_neigh = (c->neighbors[5] && c->neighbors[5]->level == c->level) ? c->neighbors[5]->face_w_F : w_face_K_self;

        double u_face_Left   = 0.5 * (u_face_L_self + u_face_L_neigh);
        double u_face_Right  = 0.5 * (u_face_R_self + u_face_R_neigh);
        double v_face_Bottom = 0.5 * (v_face_B_self + v_face_B_neigh);
        double v_face_Top    = 0.5 * (v_face_T_self + v_face_T_neigh);
        double w_face_Front  = 0.5 * (w_face_F_self + w_face_F_neigh);
        double w_face_Back   = 0.5 * (w_face_K_self + w_face_K_neigh);

        double div_face_jump = (u_face_Right - u_face_Left)/c->dx + (v_face_Top - v_face_Bottom)/c->dy + (w_face_Back - w_face_Front)/c->dz;

        // Strategy 2: Un-Limited Face Riemann Jump Sensor in 3D (100% P0-Immune)
        double jump_L = std::max(0.0, u_face_L_neigh - u_face_L_self);
        double jump_R = std::max(0.0, u_face_R_self - u_face_R_neigh);
        double jump_B = std::max(0.0, v_face_B_neigh - v_face_B_self);
        double jump_T = std::max(0.0, v_face_T_self - v_face_T_neigh);
        double jump_F = std::max(0.0, w_face_F_neigh - w_face_F_self);
        double jump_K = std::max(0.0, w_face_K_self - w_face_K_neigh);
        double max_face_jump = std::max({jump_L, jump_R, jump_B, jump_T, jump_F, jump_K});
        double riemann_jump_ind = max_face_jump / (N_factor * a_min);

        double shock_val = 0.0;
        double div_u_sum = 0.0;
        double rho_sum = 0.0, rhou_sum = 0.0, rhov_sum = 0.0, rhow_sum = 0.0, E_sum = 0.0;
        double Mn_sum = 0.0;
        double curl_u_sum = 0.0;

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
                    ind_val = std::max(ind_val, riemann_jump_ind);

                    double weight = (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                    shock_val = std::max(shock_val, ind_val);

                    // 3D Vorticity components
                    double dw_dy = 0.0, dv_dz = 0.0;
                    double du_dz = 0.0, dw_dx = 0.0;
                    double dv_dx = 0.0, du_dy = 0.0;
                    for (int k = 0; k < Np; ++k) {
                        dw_dy += basis.D[iy][k] * w_buf[iz][k][ix];
                        dv_dz += basis.D[iz][k] * v_buf[k][iy][ix];
                        du_dz += basis.D[iz][k] * u_buf[k][iy][ix];
                        dw_dx += basis.D[ix][k] * w_buf[iz][iy][k];
                        dv_dx += basis.D[ix][k] * v_buf[iz][iy][k];
                        du_dy += basis.D[iy][k] * u_buf[iz][k][ix];
                    }
                    dw_dy *= (2.0 / c->dy); dv_dz *= (2.0 / c->dz);
                    du_dz *= (2.0 / c->dz); dw_dx *= (2.0 / c->dx);
                    dv_dx *= (2.0 / c->dx); du_dy *= (2.0 / c->dy);

                    double om_x = dw_dy - dv_dz;
                    double om_y = du_dz - dw_dx;
                    double om_z = dv_dx - du_dy;

                    div_u_sum += weight * div_u;
                    Mn_sum += weight * M_n_loc;
                    curl_u_sum += weight * (om_x*om_x + om_y*om_y + om_z*om_z);

                    rho_sum += weight * rho;
                    rhou_sum += weight * (rho * u);
                    rhov_sum += weight * (rho * v);
                    rhow_sum += weight * (rho * w_vel);
                    E_sum += weight * E;
                }
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

        // 1. Refined Kinematic Sensor I_shock = Ducros * S(phi_eff) in 3D
        double h_node = h_eff / N_factor;
        double phi_comp = -h_node * std::min(0.0, div_u_sum) / (a_avg + 1e-12);
        double phi_eff = std::max(shock_val, phi_comp);

        double Ducros_ratio = 1.0;
        if (p.PPR_USE_DUCROS_SENSOR) {
            double div_sq = div_u_sum * div_u_sum;
            Ducros_ratio = div_sq / (div_sq + curl_u_sum + 1e-12);
        }

        double noise_floor = p.PPR_SENSOR_NOISE_FLOOR;
        double saturation = p.PPR_SENSOR_SATURATION;
        double s_range = std::max(1e-6, saturation - noise_floor);
        double s_phi = std::min(1.0, std::max(0.0, (phi_eff - noise_floor) / s_range));
        double I_shock = Ducros_ratio * s_phi;

        // 2. Adaptive Relaxation Timescale Ratio C_tau & Master Law Coupling Intensity theta_e
        if (p.PPR_CONSTANT_MODE) {
            // Constant-mode: user-specified uniform theta and C_tau everywhere
            c->C_tau_cell = p.PPR_CONSTANT_C_TAU_VAL;
            c->theta_max_tmp = p.PPR_CONSTANT_THETA;
        } else {
        double C_tau_0 = (p.PPR_C_TAU != 0.0) ? std::abs(p.PPR_C_TAU) : 0.25;
        double C_tau_base = C_tau_0;// * std::max(0.5, p.PPR_N_CELLS_SHOCK);
        double theta_target = (p.PPR_N_CELLS_SHOCK * N_factor * (p.GAMMA + 1.0) / (8.0 * std::max(0.01, C_tau_0))) * (1.0 + M_mach_use) * phi_eff;
        double theta_e = theta_target * I_shock;

        // 3. Three-State Relaxation Controller & Von Neumann Ceiling in 3D
        double C_tau_vonNeumann = C_tau_base;
        if (p.PPR_USE_VON_NEUMANN_CEILING) {
            C_tau_vonNeumann = std::min(C_tau_base, 0.90 / (theta_e + 1.0));
        }
        double C_tau_blend = (1.0 - I_shock) * C_tau_vonNeumann + I_shock * C_tau_base;

        c->C_tau_cell = C_tau_blend;
        c->theta_max_tmp = theta_e;
        } // end adaptive mode
    }

    if (p.PPR_CONSTANT_MODE) {
        // Constant mode: just copy theta_max_tmp -> theta_avg, no expansion needed
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            CellDim<3>* c = cells[i];
            if (p.ENABLE_MULTIRATE && !c->element_active) continue;
            c->theta_avg = c->theta_max_tmp;
            c->theta_ax = 0.0;
            c->theta_ay = 0.0;
            c->theta_az = 0.0;
        }
    } else {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        CellDim<3>* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        double max_t = c->theta_max_tmp;
        double max_c_tau = c->C_tau_cell;
        if (p.PPR_USE_STENCIL_EXPANSION) {
            for (int f = 0; f < 6; ++f) {
                if (c->neighbors[f]) {
                    max_t = std::max(max_t, 0.5 * c->neighbors[f]->theta_max_tmp);
                    max_c_tau = std::max(max_c_tau, c->neighbors[f]->C_tau_cell);
                }
            }
        }
        c->theta_avg = max_t;
        c->C_tau_cell = max_c_tau;
        c->theta_ax = 0.0;
        c->theta_ay = 0.0;
        c->theta_az = 0.0;
    }
    } // end adaptive stencil expansion
}

void relax_phantom_pressure_3d(CellDim<3>& cell, double dt_stage, const Basis& basis, const Parameters& p) {
    if (!p.ENABLE_PPR) return;
    (void)basis;

    const int Np = p.N_PTS;
    const double N_factor = p.P_DEG + 1;
    const double eps = p.POS_LIMITER_EPS;
    const double h_eff = std::min({cell.dx, cell.dy, cell.dz}) / N_factor;

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
                double P_phan = cell.S_field[k] / rho;
                double div_u = (cell.get_U(1, iz, iy, ix, Np) - cell.get_U(1, iz, iy, std::max(0, ix-1), Np)) / h_eff;

                // Thermodynamic Instant-Thermalization Guard in 3D
                bool anti_dissipative = (cell.theta_avg * (P_phys - P_phan) * div_u > 0.0) ||
                                        (div_u < 0.0 && P_phan > P_phys);
                if (p.PPR_USE_ENERGY_GUARD && anti_dissipative) {
                    // Instantly thermalize P_phan -> P_phys in 1 RK stage
                    double S_eq = rho * P_phys;
                    cell.S_field[k] = S_eq;
                    continue;
                }

                double a_phys = std::sqrt(p.GAMMA * P_phys / rho);
                double speed = std::sqrt(u*u + v*v + w*w);

                double C_tau_eff = cell.C_tau_cell;
                double tau = C_tau_eff * h_eff / (a_phys + speed + 1e-12);
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
    if (!p.PPR_USE_LIMITER) return;
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

                    double P_phan_max = (1.0 + (1.0 - c_pos) / (theta + 1e-12)) * P_phys;
                    double P_phan_min = std::max(0.0, (1.0 - (C_max - 1.0) / (theta + 1e-12)) * P_phys);

                    if (p.PPR_USE_SPATIAL_CLAMP) {
                        P_phan_max = std::min(P_nodal_max, P_phan_max);
                        P_phan_min = std::min(P_nodal_min, P_phan_min);
                    }

                    double P_phan_clipped = std::clamp(P_phan, P_phan_min, P_phan_max);
                    c->S_field[k] = rho * P_phan_clipped;
                }
            }
        }
    }
}

} // namespace PPR

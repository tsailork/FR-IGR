/**
 * @file ib_gcm.cpp
 * @brief Implementations of High-Order Ghost-Cell Method for Flux Reconstruction (GCM-FR).
 */

#include "ib_gcm.hpp"
#include "ib_wall_function.hpp"
#include "ib_motion.hpp"
#include "../core/solver.hpp"
#include "../limiters/limiter_modal.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fr::ib {

bool GhostNodeIbSolver::sample_fluid_state(const Solver& solver, const double pos[3], double state_out[5]) const {
    double px = pos[0];
    double py = pos[1];
    int npts = solver.p.N_PTS;

    const Cell* c = nullptr;
    for (size_t b_idx = 0; b_idx < solver.blocks.size(); ++b_idx) {
        const auto& b = solver.blocks[b_idx];
        double x_max = b.x_min + b.nx * b.dx;
        double y_max = b.y_min + b.ny * b.dy;
        if (px >= b.x_min && px <= x_max && py >= b.y_min && py <= y_max) {
            int ex = std::clamp(static_cast<int>((px - b.x_min) / (b.nx * b.dx) * b.nx), 0, b.nx - 1);
            int ey = std::clamp(static_cast<int>((py - b.y_min) / (b.ny * b.dy) * b.ny), 0, b.ny - 1);
            if (b_idx < solver.block_cells.size() &&
                ey < static_cast<int>(solver.block_cells[b_idx].size()) &&
                ex < static_cast<int>(solver.block_cells[b_idx][ey].size())) {
                c = solver.block_cells[b_idx][ey][ex];
                break;
            }
        }
    }

    if (!c) {
        c = solver.find_leaf_cell(0, px, py);
    }
    if (!c) return false;

    // Map physical (x, y) to reference coordinates xi, eta in [-1, 1]
    double dx = c->dx;
    double dy = c->dy;
    double xi = (dx > 1e-14 ? 2.0 * (px - c->x_min) / dx - 1.0 : 0.0);
    double eta = (dy > 1e-14 ? 2.0 * (py - c->y_min) / dy - 1.0 : 0.0);

    // Compute 1D Lagrange polynomial weights on stack array
    double l_xi[MAX_PTS];
    double l_eta[MAX_PTS];
    for (int i = 0; i < npts; ++i) {
        l_xi[i] = 1.0;
        l_eta[i] = 1.0;
        for (int m = 0; m < npts; ++m) {
            if (m != i) {
                double denom = solver.basis.z[i] - solver.basis.z[m];
                if (std::abs(denom) > 1e-14) {
                    l_xi[i] *= (xi - solver.basis.z[m]) / denom;
                    l_eta[i] *= (eta - solver.basis.z[m]) / denom;
                }
            }
        }
    }

    // Interpolate conservative variables (rho, rhou, rhov, E)
    for (int v = 0; v < 5; ++v) {
        state_out[v] = 0.0;
    }

    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            double w = l_eta[iy] * l_xi[ix];
            state_out[0] += w * c->get_U(0, iy, ix, npts);
            state_out[1] += w * c->get_U(1, iy, ix, npts);
            state_out[2] += w * c->get_U(2, iy, ix, npts);
            state_out[3] += w * c->get_U(3, iy, ix, npts);
        }
    }
    return true;
}

void GhostNodeIbSolver::update_geometry_and_masks(Solver& solver, double time) {
    if (!solver.p.ENABLE_IB) return;
    clear();

    int p_deg = solver.p.P_DEG;
    int npts = p_deg + 1;
    int k_probes = (solver.p.IB_GCM_ORDER >= 1 ? solver.p.IB_GCM_ORDER + 1 : npts);

    double min_phi_all = 1e30;
    double max_phi_all = -1e30;
    int solid_node_count = 0;

    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        Cell* c = solver.cells[c_idx];
        if (!c) continue;

        double dx = c->dx;
        double dy = c->dy;
        double h_cell = std::max(dx, dy);

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                // Get node physical coordinates
                double x_node = c->x_min + 0.5 * dx * (1.0 + solver.basis.z[ix]);
                double y_node = c->y_min + 0.5 * dy * (1.0 + solver.basis.z[iy]);

                double phi = solver.get_ib_sdf_at_time(x_node, y_node, time);
                if (phi < min_phi_all) min_phi_all = phi;
                if (phi > max_phi_all) max_phi_all = phi;

                if (phi <= 0.0) { // Solid ghost node
                    solid_node_count++;
                    GhostNodeInfo info;
                    info.cell_idx = c_idx;
                    info.iy = iy;
                    info.ix = ix;
                    info.ghost_pos[0] = x_node;
                    info.ghost_pos[1] = y_node;
                    info.ghost_pos[2] = 0.0;
                    info.dist = std::abs(phi);

                    double nx = 0.0, ny = 0.0;
                    solver.get_ib_sdf_gradient_at_time(x_node, y_node, time, nx, ny);
                    info.normal[0] = nx;
                    info.normal[1] = ny;
                    info.normal[2] = 0.0;

                    // Establish K_PROBES sampling locations along normal ray into fluid domain.
                    double s_mirror = 2.0 * info.dist;
                    double delta_s = 0.5 * h_cell;

                    for (int k = 0; k < k_probes; ++k) {
                        double dist_k = s_mirror + k * delta_s;
                        std::array<double, 3> pk = {
                            x_node + dist_k * nx,
                            y_node + dist_k * ny,
                            0.0
                        };
                        info.probe_pts.push_back(pk);
                    }

                    ghost_nodes_.push_back(info);
                }
            }
        }
    }
    std::cout << "[GCM DIAG] min_phi=" << min_phi_all << " max_phi=" << max_phi_all << " solid_nodes=" << solid_node_count << " at t=" << time << "\n";
}

void GhostNodeIbSolver::apply_ghost_nodal_states(Solver& solver, double time) {
    if (!solver.p.ENABLE_IB || ghost_nodes_.empty()) return;

    int p_deg = solver.p.P_DEG;
    int npts = p_deg + 1;
    double gamma = solver.p.GAMMA;

    #pragma omp parallel for schedule(static)
    for (size_t g_idx = 0; g_idx < ghost_nodes_.size(); ++g_idx) {
        const auto& info = ghost_nodes_[g_idx];
        Cell* c = solver.cells[info.cell_idx];
        if (!c) continue;

        // Sample fluid probing states along normal ray
        size_t K = info.probe_pts.size();
        std::vector<std::array<double, 5>> probe_states(K);

        for (size_t k = 0; k < K; ++k) {
            double st[5] = {0.0};
            if (!sample_fluid_state(solver, info.probe_pts[k].data(), st)) {
                // Fallback to ghost cell average if probe point is outside domain
                for (int v = 0; v < CellDim<2>::N_VARS; ++v) {
                    st[v] = c->get_U(v, info.iy, info.ix, npts);
                }
            }
            for (int v = 0; v < 5; ++v) probe_states[k][v] = st[v];
        }

        // Primary probe fluid state (closest probe point)
        double rho_p = probe_states[0][0];
        double rhou_p = probe_states[0][1];
        double rhov_p = probe_states[0][2];
        double E_p = probe_states[0][3];

        double inv_rho_p = 1.0 / std::max(1e-12, rho_p);
        double u_p = rhou_p * inv_rho_p;
        double v_p = rhov_p * inv_rho_p;

        // Solid body motion velocity targets via DynamicMotionHandler
        auto motion = DynamicMotionHandler::evaluate_motion(solver.p, time, info.ghost_pos);
        double u_b = motion.vel[0];
        double v_b = motion.vel[1];

        double d_g = info.dist;
        double s_g = -d_g;

        // High-Order Lagrange Normal Extrapolation Q(s) matching solver polynomial degree P_DEG
        double u_g = 0.0, v_g = 0.0, rho_g = 0.0, P_g = 0.0;

        double nx = info.normal[0], ny = info.normal[1];
        double norm_len = std::sqrt(nx * nx + ny * ny);
        if (norm_len > 1e-12) { nx /= norm_len; ny /= norm_len; }
        else { nx = 0.0; ny = 1.0; }

        if (p_deg == 1 || K < 2) {
            // Linear Extrapolation P1 (Inviscid Slip Wall)
            double rho_p0 = probe_states[0][0];
            double inv_rho0 = 1.0 / std::max(1e-12, rho_p0);
            double u_p0 = probe_states[0][1] * inv_rho0;
            double v_p0 = probe_states[0][2] * inv_rho0;
            double P_p0 = (gamma - 1.0) * (probe_states[0][3] - 0.5 * rho_p0 * (u_p0 * u_p0 + v_p0 * v_p0));

            rho_g = rho_p0;
            double un_p = u_p0 * nx + v_p0 * ny;
            double un_b = u_b * nx + v_b * ny;
            double un_g = 2.0 * un_b - un_p;

            bool is_no_slip = (solver.p.ENABLE_NS && solver.p.IB_THERMAL_TYPE == "ISOTHERMAL");
            if (is_no_slip) {
                // Viscous No-Slip Wall: Invert both normal and tangential velocities
                u_g = 2.0 * u_b - u_p0;
                v_g = 2.0 * v_b - v_p0;
            } else {
                // Inviscid Slip Wall: Reflect normal velocity, preserve tangential velocity
                u_g = u_p0 + (un_g - un_p) * nx;
                v_g = v_p0 + (un_g - un_p) * ny;
            }
            P_g = std::max(solver.p.POS_LIMITER_EPS, P_p0);
        } else if (p_deg == 2 || K == 2) {
            // Quadratic Extrapolation P2
            double s0 = d_g;
            double s1 = d_g + 0.5 * std::max(c->dx, c->dy);

            double rho_p0 = probe_states[0][0], rho_p1 = probe_states[1][0];
            double u_p0 = probe_states[0][1] / std::max(1e-12, rho_p0);
            double u_p1 = probe_states[1][1] / std::max(1e-12, rho_p1);
            double v_p0 = probe_states[0][2] / std::max(1e-12, rho_p0);
            double v_p1 = probe_states[1][2] / std::max(1e-12, rho_p1);

            double P_p0 = (gamma - 1.0) * (probe_states[0][3] - 0.5 * rho_p0 * (u_p0 * u_p0 + v_p0 * v_p0));
            double P_p1 = (gamma - 1.0) * (probe_states[1][3] - 0.5 * rho_p1 * (u_p1 * u_p1 + v_p1 * v_p1));

            // Quadratic u(s) through (0, u_b), (s0, u_p0), (s1, u_p1)
            double L_w = (s_g - s0) * (s_g - s1) / ((0 - s0) * (0 - s1));
            double L_0 = s_g * (s_g - s1) / (s0 * (s0 - s1));
            double L_1 = s_g * (s_g - s0) / (s1 * (s1 - s0));

            u_g = u_b * L_w + u_p0 * L_0 + u_p1 * L_1;
            v_g = v_b * L_w + v_p0 * L_0 + v_p1 * L_1;

            // Reflect normal component for Euler slip wall
            double un_g_raw = u_g * nx + v_g * ny;
            double un_p0 = u_p0 * nx + v_p0 * ny;
            double un_b = u_b * nx + v_b * ny;
            double un_g_refl = 2.0 * un_b - un_p0;
            u_g = u_g + (un_g_refl - un_g_raw) * nx;
            v_g = v_g + (un_g_refl - un_g_raw) * ny;

            // Zero normal derivative for density and pressure: R(s) = A s^2 + C
            double denom_s = std::max(1e-12, s1 * s1 - s0 * s0);
            double A_rho = (rho_p1 - rho_p0) / denom_s;
            double C_rho = rho_p0 - A_rho * s0 * s0;
            rho_g = std::max(solver.p.POS_LIMITER_EPS, A_rho * s_g * s_g + C_rho);

            double A_P = (P_p1 - P_p0) / denom_s;
            double C_P = P_p0 - A_P * s0 * s0;
            P_g = std::max(solver.p.POS_LIMITER_EPS, A_P * s_g * s_g + C_P);
        } else {
            // Cubic Extrapolation P3 matching P_DEG = 3
            double s_pts[4] = {0.0, d_g, d_g + 0.5 * std::max(c->dx, c->dy), d_g + std::max(c->dx, c->dy)};
            double u_pts[4] = {u_b, 0.0, 0.0, 0.0};
            double v_pts[4] = {v_b, 0.0, 0.0, 0.0};

            for (size_t k = 0; k < 3 && k < K; ++k) {
                double r_k = probe_states[k][0];
                double inv_r = 1.0 / std::max(1e-12, r_k);
                u_pts[k+1] = probe_states[k][1] * inv_r;
                v_pts[k+1] = probe_states[k][2] * inv_r;
            }

            u_g = 0.0; v_g = 0.0;
            for (int m = 0; m < 4; ++m) {
                double L_m = 1.0;
                for (int j = 0; j < 4; ++j) {
                    if (j != m) L_m *= (s_g - s_pts[j]) / (s_pts[m] - s_pts[j]);
                }
                u_g += u_pts[m] * L_m;
                v_g += v_pts[m] * L_m;
            }
            rho_g = probe_states[0][0];
            double u_p0 = probe_states[0][1] / std::max(1e-12, rho_g);
            double v_p0 = probe_states[0][2] / std::max(1e-12, rho_g);
            P_g = std::max(solver.p.POS_LIMITER_EPS, (gamma - 1.0) * (probe_states[0][3] - 0.5 * rho_g * (u_p0 * u_p0 + v_p0 * v_p0)));

            double un_g_raw = u_g * nx + v_g * ny;
            double un_p0 = u_p0 * nx + v_p0 * ny;
            double un_b = u_b * nx + v_b * ny;
            double un_g_refl = 2.0 * un_b - un_p0;
            u_g = u_g + (un_g_refl - un_g_raw) * nx;
            v_g = v_g + (un_g_refl - un_g_raw) * ny;
        }

        if (solver.p.ENABLE_IB_WALL_FUNCTION) {
            double rho_p0 = probe_states[0][0];
            double u_p0 = probe_states[0][1] / std::max(1e-12, rho_p0);
            double v_p0 = probe_states[0][2] / std::max(1e-12, rho_p0);
            WallFunctionModel wf_model(solver.p.IB_WF_KAPPA, solver.p.IB_WF_B);
            double u_rel_x = u_p0 - u_b;
            double u_rel_y = v_p0 - v_b;
            double u_parallel = std::sqrt(u_rel_x * u_rel_x + u_rel_y * u_rel_y);
            double delta_wf = std::max(1e-5, info.dist);
            double T_wf = (gamma - 1.0) * (probe_states[0][3] - 0.5 * rho_p0 * (u_p0 * u_p0 + v_p0 * v_p0)) / std::max(1e-12, rho_p0);
            double T_wall = solver.p.IB_TEMPERATURE;
            double mu_wf = 1.0 / std::max(1.0, solver.p.RE);

            auto wf_res = wf_model.solve_wall_function(delta_wf, rho_p0, u_parallel, T_wf, T_wall, mu_wf, gamma,
                                                       solver.p.IB_WF_MAX_ITER, solver.p.IB_WF_TOL);

            if (u_parallel > 1e-12) {
                double scale = wf_res.u_wall_eff / u_parallel;
                u_g = u_b + scale * u_rel_x;
                v_g = v_b + scale * u_rel_y;
            }
        }

        double max_speed = 1.5 * (std::abs(solver.p.U_INF) + std::abs(solver.p.V_INF) + 1.0);
        double speed_g = std::sqrt(u_g * u_g + v_g * v_g);
        if (speed_g > max_speed) {
            double scale = max_speed / speed_g;
            u_g *= scale;
            v_g *= scale;
        }

        double rhou_g = rho_g * u_g;
        double rhov_g = rho_g * v_g;
        double E_g = 0.0;

        if (solver.p.IB_THERMAL_TYPE == "ADIABATIC") {
            E_g = P_g / (gamma - 1.0) + 0.5 * rho_g * (u_g * u_g + v_g * v_g);
        } else {
            double T_w = solver.p.IB_TEMPERATURE;
            E_g = rho_g * T_w / (gamma - 1.0) + 0.5 * rho_g * (u_g * u_g + v_g * v_g);
        }

        // Set ghost nodal conservative states
        c->get_U(0, info.iy, info.ix, npts) = rho_g;
        c->get_U(1, info.iy, info.ix, npts) = rhou_g;
        c->get_U(2, info.iy, info.ix, npts) = rhov_g;
        c->get_U(3, info.iy, info.ix, npts) = E_g;
    }
}

void GhostNodeIbSolver::handle_freshly_cleared_nodes(Solver& solver) {
    if (!solver.p.ENABLE_IB || !solver.p.ENABLE_IB_CLEARED_NODE_HANDLER) return;

    // Detect nodes transitioning from solid (phi^n <= 0) to fluid (phi^{n+1} > 0)
    int p_deg = solver.p.P_DEG;
    int npts = p_deg + 1;

    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        Cell* c = solver.cells[c_idx];
        if (!c) continue;

        double dx = c->dx;
        double dy = c->dy;

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double x_node = c->x_min + 0.5 * dx * (1.0 + solver.basis.z[ix]);
                double y_node = c->y_min + 0.5 * dy * (1.0 + solver.basis.z[iy]);

                double phi_now = solver.get_ib_sdf_at_time(x_node, y_node, solver.current_time);
                double phi_prev = solver.get_ib_sdf_at_time(x_node, y_node, std::max(0.0, solver.current_time - c->element_dt));

                if (phi_prev <= 0.0 && phi_now > 0.0) { // Freshly cleared fluid node
                    double nx = 0.0, ny = 0.0;
                    solver.get_ib_sdf_gradient_at_time(x_node, y_node, solver.current_time, nx, ny);

                    double probe_pos[3] = {
                        x_node + solver.p.IB_GCM_PROBE_DISTANCE_SCALE * dx * nx,
                        y_node + solver.p.IB_GCM_PROBE_DISTANCE_SCALE * dy * ny,
                        0.0
                    };

                    double st[5] = {0.0};
                    if (sample_fluid_state(solver, probe_pos, st)) {
                        c->get_U(0, iy, ix, npts) = st[0];
                        c->get_U(1, iy, ix, npts) = st[1];
                        c->get_U(2, iy, ix, npts) = st[2];
                        c->get_U(3, iy, ix, npts) = st[3];
                    }
                }
            }
        }
    }
}

void GhostNodeIbSolver::apply_ib_interface_damping(Solver& solver) {
    if (!solver.p.ENABLE_IB) return;
    if (solver.p.P_DEG < 1) return;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < solver.cells.size(); ++i) {
        Cell* c = solver.cells[i];
        if (!c) continue;

        // Only filter IB-cut cells having solution points on both sides of IB
        if (c->is_ib_cut_cell) {
            double damping = (solver.p.P_DEG >= 3 ? 0.3 : 0.5);
            Limiters::apply_ib_modal_filter(*c, solver.basis, solver.p, damping);
        }
    }
}

} // namespace fr::ib

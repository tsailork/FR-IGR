/**
 * @file solver.cpp
 * @brief Solver constructor and compute_rhs dispatcher.
 *
 * Implements the initialization logic for the high-order `Solver` structure, 
 * including string parsing for boundary condition setups and the main 
 * RHS assembly pipeline (`compute_rhs`).
 * 
 * @see Solver
 */
#include "solver.hpp"
#include "geometry.hpp"
#include <stdexcept>
#include <unordered_set>
#include <queue>
#include <map>
#include <algorithm>
#include "../ib/sbm_geometry.hpp"
#include "../igr/ducros_sensor.hpp"

/**
 * @brief Parse a boundary condition string from domain.grid into a NeighborInfo metadata struct.
 *
 * Converts a configuration string into strongly typed boolean flags and reference values.
 * Supported formats:
 * - `"WALL"` or `"WALL_SLIP"` — inviscid slip wall (reflects normal velocity).
 * - `"INFLOW"`            — freestream inflow.
 * - `"TRANSMISSIVE"`      — copy-out / zero-gradient.
 * - `"WALL_NOSLIP"`       — adiabatic no-slip wall (\f$ u=v=0 \f$, \f$ dT/dn=0 \f$).
 * - `"WALL_NOSLIP:T"`     — isothermal no-slip wall at temperature \f$ T \f$.
 * - `"WALL_MOVING:V"`     — adiabatic moving wall, tangential velocity \f$ V \f$.
 * - `"WALL_MOVING:V:T"`   — isothermal moving wall, velocity \f$ V \f$, temperature \f$ T \f$.
 * - `"ID:FACE"`           — block-to-block connectivity (e.g., `"1:L"`).
 *
 * @param[in] bc The boundary condition string to parse.
 * @param[out] ni The NeighborInfo structure to populate.
 */
static void parse_bc_string(const std::string& bc_in, NeighborInfo& ni) {
    std::string bc = bc_in;
    ni.refine = true;

    // Check for refinement flag suffixes
    std::vector<std::pair<std::string, bool>> suffixes = {
        {":NOREFINED", false},
        {":NOREFINEMENT", false},
        {":NO_REFINE", false},
        {":NOREFINE", false},
        {":REFINED", true},
        {":REFINE", true}
    };
    for (const auto& suffix_pair : suffixes) {
        const std::string& suf = suffix_pair.first;
        if (bc.size() >= suf.size() && bc.compare(bc.size() - suf.size(), suf.size(), suf) == 0) {
            ni.refine = suffix_pair.second;
            bc = bc.substr(0, bc.size() - suf.size());
            break;
        }
    }

    if (bc == "WALL" || bc == "WALL_SLIP") {
        ni.is_wall = true;
    } else if (bc.rfind("INFLOW_SUPERSONIC:", 0) == 0) {
        ni.is_supersonic_inflow = true;
        std::string params = bc.substr(18);
        size_t p1 = params.find(':');
        size_t p2 = params.find(':', p1 + 1);
        size_t p3 = params.find(':', p2 + 1);
        if (p1 != std::string::npos && p2 != std::string::npos && p3 != std::string::npos) {
            ni.ref_rho = std::stod(params.substr(0, p1));
            ni.ref_u   = std::stod(params.substr(p1 + 1, p2 - p1 - 1));
            ni.ref_v   = std::stod(params.substr(p2 + 1, p3 - p2 - 1));
            ni.ref_p   = std::stod(params.substr(p3 + 1));
        }
    } else if (bc == "OUTFLOW_SUPERSONIC") {
        ni.is_supersonic_outflow = true;
    } else if (bc.rfind("CHARACTERISTIC:", 0) == 0) {
        ni.is_characteristic = true;
        std::string params = bc.substr(15);
        size_t p1 = params.find(':');
        size_t p2 = params.find(':', p1 + 1);
        size_t p3 = params.find(':', p2 + 1);
        if (p1 != std::string::npos && p2 != std::string::npos && p3 != std::string::npos) {
            ni.ref_rho = std::stod(params.substr(0, p1));
            ni.ref_u   = std::stod(params.substr(p1 + 1, p2 - p1 - 1));
            ni.ref_v   = std::stod(params.substr(p2 + 1, p3 - p2 - 1));
            ni.ref_p   = std::stod(params.substr(p3 + 1));
        }
    } else if (bc == "WALL_NOSLIP") {
        ni.is_noslip_wall = true;
    } else if (bc.rfind("WALL_NOSLIP:", 0) == 0) {
        // WALL_NOSLIP:T_wall — isothermal no-slip
        ni.is_noslip_wall = true;
        ni.is_isothermal = true;
        ni.wall_temperature = std::stod(bc.substr(12));
    } else if (bc.rfind("WALL_MOVING:", 0) == 0) {
        // WALL_MOVING:velocity  or  WALL_MOVING:velocity:T_wall
        ni.is_moving_wall = true;
        std::string params = bc.substr(12);
        size_t colon = params.find(':');
        if (colon != std::string::npos) {
            ni.wall_velocity = std::stod(params.substr(0, colon));
            ni.is_isothermal = true;
            ni.wall_temperature = std::stod(params.substr(colon + 1));
        } else {
            ni.wall_velocity = std::stod(params);
        }
    } else if (bc.rfind("TOTAL_PRESSURE_COMP:", 0) == 0) {
        ni.is_total_pressure_comp = true;
        ni.ref_p = std::stod(bc.substr(20));
    } else if (bc.rfind("TOTAL_PRESSURE_INCOMP:", 0) == 0) {
        ni.is_total_pressure_incomp = true;
        ni.ref_p = std::stod(bc.substr(22));
    } else if (bc.rfind("STATIC_PRESSURE:", 0) == 0) {
        ni.is_static_pressure = true;
        ni.ref_p = std::stod(bc.substr(16));
    } else if (bc.find(':') != std::string::npos) {
        size_t sep = bc.find(':');
        ni.id = std::stoi(bc.substr(0, sep));
        ni.face = bc[sep + 1];
    } else {
        ni.is_supersonic_outflow = true;
    }
}

SolverDim<2>::SolverDim(const Parameters& params)
    : p(params), basis(p.P_DEG)
{
    current_time = p.RESTART_TIME;
    blocks.clear();
    for (const auto& config : p.blocks) {
        blocks.emplace_back(config, p.N_PTS);
    }
    
    // Pre-parse connectivity
    for (auto& b : blocks) {
        parse_bc_string(b.bc_l, b.ni_l);
        parse_bc_string(b.bc_r, b.ni_r);
        parse_bc_string(b.bc_b, b.ni_b);
        parse_bc_string(b.bc_t, b.ni_t);
    }

    initialize_cells();
    setup_cell_connectivity();

    // Apply initial refinement based on walls and manual zones
    flag_refinement_coarsening();

    // Compute initial IB mask field
    if (p.ENABLE_IB) {
        update_ib_mask_field(current_time);
        if (p.IB_METHOD == "SBM") {
            ImmersedBoundary::initialize_sbm_geometry(*this);
        }
    }
}

/// Pre-compute element-average adaptive theta for PPR and store in each cell's theta_avg.
void Solver::compute_ppr_theta_avg() {
    if (!p.ENABLE_PPR) return;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        if (!p.PPR_ADAPTIVE_THETA) {
            c->theta_max_tmp = p.PPR_THETA;
            continue;
        }
        double theta_max_cell = 0.0;
        const double dmax = std::max(1.001, p.PPR_DIV_ND_MAX);
        
        // Pre-compute u and v arrays to avoid O(N^3) division by rho
        double u_buf[MAX_PTS][MAX_PTS];
        double v_buf[MAX_PTS][MAX_PTS];
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                u_buf[iy][ix] = c->get_U(1, iy, ix, p.N_PTS) / rho;
                v_buf[iy][ix] = c->get_U(2, iy, ix, p.N_PTS) / rho;
            }
        }

        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double theta;
                if (!p.PPR_THETA_SCHEDULE.empty() && p.PPR_THETA_SCHEDULE.size() == p.PPR_SENS_SCHEDULE.size()) {
                    double sensor_val = Sensors::compute_combined_ppr_sensor(*c, iy, ix, basis, p, u_buf, v_buf);
                    theta = Sensors::interpolate_schedule(sensor_val, p.PPR_SENS_SCHEDULE, p.PPR_THETA_SCHEDULE);
                } else {
                    // Fallback to 3-point non-dimensional divergence interpolation
                    double du_dx = 0.0, dv_dy = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        du_dx += basis.D[ix][k] * u_buf[iy][k];
                        dv_dy += basis.D[iy][k] * v_buf[k][ix];
                    }
                    du_dx *= (2.0 / c->dx);
                    dv_dy *= (2.0 / c->dy);
                    double div_u = du_dx + dv_dy;

                    double rho   = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                    double u_loc = u_buf[iy][ix];
                    double v_loc = v_buf[iy][ix];
                    double P_loc = std::max(p.POS_LIMITER_EPS,
                        (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, p.N_PTS) - 0.5 * rho * (u_loc*u_loc + v_loc*v_loc)));
                    double a_loc  = std::sqrt(p.GAMMA * P_loc / rho);
                    double h_loc  = std::min(c->dx, c->dy);
                    double div_nd = -div_u * h_loc / (a_loc * (p.P_DEG + 1));

                    if      (div_nd <= 0.0)   theta = p.PPR_THETA_MIN;
                    else if (div_nd <  1.0)   theta = p.PPR_THETA_MIN + (p.PPR_THETA_MID - p.PPR_THETA_MIN) * div_nd;
                    else if (div_nd <  dmax)  theta = p.PPR_THETA_MID + (p.PPR_THETA_MAX - p.PPR_THETA_MID) * (div_nd - 1.0) / (dmax - 1.0);
                    else                      theta = p.PPR_THETA_MAX;
                }
                theta_max_cell = std::max(theta_max_cell, theta);
            }
        }
        c->theta_max_tmp = theta_max_cell;
    }

    if (p.PPR_SMOOTH_THETA) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            if (p.ENABLE_MULTIRATE && !c->element_active) continue;
            double max_t = c->theta_max_tmp;
            for (int f = 0; f < 4; ++f) {
                if (c->neighbors[f]) {
                    max_t = std::max(max_t, c->neighbors[f]->theta_max_tmp);
                }
            }
            c->theta_avg = max_t;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            if (p.ENABLE_MULTIRATE && !cells[i]->element_active) continue;
            cells[i]->theta_avg = cells[i]->theta_max_tmp;
        }
    }
}

/// Assemble the full right-hand side: IGR + inviscid sweeps + viscous fluxes.
void Solver::compute_rhs() {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        std::fill(c->RHS.begin(), c->RHS.end(), 0.0);
        if (p.ENABLE_PPR) {
            std::fill(c->S_RHS.begin(), c->S_RHS.end(), 0.0);
        }
    }

    compute_entropic_pressure();
    compute_ppr_theta_avg();
    sweep_x();
    sweep_y();
    if (p.ENABLE_NS) {
        compute_gradients();
        viscous_sweep_x();
        viscous_sweep_y();
    }
    if (p.ENABLE_IB && p.IB_METHOD == "VPM_EXPLICIT") {
        apply_ib_explicit();
    }

    // Blank the RHS for elements that are fully inside the immersed body.
    // These cells have their state held at freestream by initialize_sbm_geometry;
    // zeroing the RHS ensures the SSP-RK3 update never modifies them.
    if (p.ENABLE_IB) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            if (p.ENABLE_MULTIRATE && !c->element_active) continue;
            if (c->solid_mask) {
                std::fill(c->RHS.begin(), c->RHS.end(), 0.0);
                if (p.ENABLE_PPR) {
                    std::fill(c->S_RHS.begin(), c->S_RHS.end(), 0.0);
                }
            }
        }
    }
}

void Solver::compute_local_dt() {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        double max_lambda = 1e-10;
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double rho = std::max(1e-12, c->get_U(0, iy, ix, p.N_PTS));
                double u   = c->get_U(1, iy, ix, p.N_PTS) / rho;
                double v   = c->get_U(2, iy, ix, p.N_PTS) / rho;
                double press = (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, p.N_PTS) - 0.5 * rho * (u * u + v * v));
                double press_safe = press;
                if (p.ENABLE_PPR) {
                    double theta_cfl = (p.PPR_ADAPTIVE_THETA) ? c->theta_avg : p.PPR_THETA;
                    double p_phan = c->S_field[iy * p.N_PTS + ix] / rho;
                    double p_reg = press + theta_cfl * (press - p_phan);
                    press_safe = std::max(press, p_reg);
                }
                if (press_safe < 1e-12) press_safe = 1e-12;
                double sound_speed = std::sqrt(p.GAMMA * press_safe / rho);
                max_lambda = std::max({max_lambda, std::abs(u) + sound_speed, std::abs(v) + sound_speed});
                if (p.ENABLE_PPR) {
                    double s_wave_x = p.PPR_ADV_MULT * (std::abs(u) + p.PPR_GRAD_ADV_SCALE * sound_speed);
                    double s_wave_y = p.PPR_ADV_MULT * (std::abs(v) + p.PPR_GRAD_ADV_SCALE * sound_speed);
                    max_lambda = std::max({max_lambda, s_wave_x, s_wave_y});
                }
            }
        }
        double h = std::min(c->dx, c->dy);
        double dt_conv = 0.5 * p.CFL * h / (max_lambda * (p.P_DEG + 1) * (p.P_DEG + 1));
        double dt_elem = dt_conv;

        if (p.ENABLE_IGR && p.IGR_TYPE == "PARABOLIC") {
            if (p.IGR_SUB_ITERS > 0) {
                double alpha_safe = std::max(1e-10, p.ALPHA_SCALE);
                double dt_diff  = 0.5 * p.IGR_TAU_R / (alpha_safe * (1.0 + p.IGR_BR2_ETA) * (2 * p.P_DEG + 1) * (2 * p.P_DEG + 1));
                double dt_relax = 0.5 * p.IGR_TAU_R;
                double dt_limit = std::min(dt_diff, dt_relax);
                dt_elem = std::min(dt_elem, p.IGR_SUB_ITERS * dt_limit);
            }
        }

        if (p.ENABLE_NS) {
            double nu = 1.0 / p.RE;
            double h2 = h * h;
            double denom = (2 * p.P_DEG + 1) * (2 * p.P_DEG + 1);
            double dt_visc = 0.25 * p.CFL * h2 / (nu * denom);
            dt_elem = std::min(dt_elem, dt_visc);
        }

        c->element_dt = dt_elem;
    }
}

// =========================================================================
// Cell-level solver integration
// =========================================================================

static inline uint64_t dilate_1d(uint32_t x) {
    uint64_t val = x;
    val = (val | (val << 16)) & 0x0000FFFF0000FFFFULL;
    val = (val | (val << 8))  & 0x00FF00FF00FF00FFULL;
    val = (val | (val << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    val = (val | (val << 2))  & 0x3333333333333333ULL;
    val = (val | (val << 1))  & 0x5555555555555555ULL;
    return val;
}

static inline uint64_t morton_encode_2d(uint32_t x, uint32_t y) {
    return dilate_1d(x) | (dilate_1d(y) << 1);
}

static inline uint32_t undilate_1d(uint64_t val) {
    uint64_t x = val & 0x5555555555555555ULL;
    x = (x | (x >> 1))  & 0x3333333333333333ULL;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFFULL;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFFULL;
    return (uint32_t)x;
}

static inline void morton_decode_2d(uint64_t morton, uint32_t& x, uint32_t& y) {
    x = undilate_1d(morton);
    y = undilate_1d(morton >> 1);
}

static inline uint64_t dilate_1d_3d(uint32_t x) {
    uint64_t val = x & 0x1FFFFFULL; // Max 21 bits
    val = (val | (val << 32)) & 0x1F00000000FFFFULL;
    val = (val | (val << 16)) & 0x1F0000FF0000FFULL;
    val = (val | (val << 8))  & 0x100F00F00F00F00FULL;
    val = (val | (val << 4))  & 0x10c30c30c30c30c3ULL;
    val = (val | (val << 2))  & 0x1249249249249249ULL;
    return val;
}

static inline uint32_t undilate_1d_3d(uint64_t val) {
    uint64_t x = val & 0x1249249249249249ULL;
    x = (x | (x >> 2))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x >> 4))  & 0x100F00F00F00F00FULL;
    x = (x | (x >> 8))  & 0x1F0000FF0000FFULL;
    x = (x | (x >> 16)) & 0x1F00000000FFFFULL;
    x = (x | (x >> 32)) & 0x1FFFFFULL;
    return (uint32_t)x;
}

static inline uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    return dilate_1d_3d(x) | (dilate_1d_3d(y) << 1) | (dilate_1d_3d(z) << 2);
}

static inline void morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = undilate_1d_3d(morton);
    y = undilate_1d_3d(morton >> 1);
    z = undilate_1d_3d(morton >> 2);
}

SolverDim<2>::~SolverDim() {
    for (Cell* c : cells) {
        delete c;
    }
    cells.clear();
}

uint64_t Solver::get_morton_id(int block_id, int level, uint32_t ex, uint32_t ey) const {
    const int L_max = 14;
    const int N_b = 5;
    uint32_t x_fine = ex << (L_max - level);
    uint32_t y_fine = ey << (L_max - level);
    return ((uint64_t)block_id << 55) | (morton_encode_2d(x_fine, y_fine) << N_b) | (uint64_t)level;
}

bool Solver::is_ancestor(uint64_t ancestor_id, uint64_t key_id) const {
    int block_id_A = (ancestor_id >> 55);
    int block_id_K = (key_id >> 55);
    if (block_id_A != block_id_K) return false;

    int L_A = ancestor_id & 0x1F;
    int L_K = key_id & 0x1F;
    if (L_A >= L_K) return false;

    uint64_t raw_A = (ancestor_id >> 5) & 0x3FFFFFFFFFFFFULL;
    uint64_t raw_K = (key_id >> 5) & 0x3FFFFFFFFFFFFULL;

    uint32_t x_fine_A, y_fine_A;
    uint32_t x_fine_K, y_fine_K;
    morton_decode_2d(raw_A, x_fine_A, y_fine_A);
    morton_decode_2d(raw_K, x_fine_K, y_fine_K);

    uint32_t ex_A = x_fine_A >> (14 - L_A);
    uint32_t ey_A = y_fine_A >> (14 - L_A);
    uint32_t ex_K = x_fine_K >> (14 - L_K);
    uint32_t ey_K = y_fine_K >> (14 - L_K);

    return ((ex_K >> (L_K - L_A)) == ex_A) && ((ey_K >> (L_K - L_A)) == ey_A);
}

void Solver::enforce_21_ratio() {
    for (Cell* c : cells) {
        int bid = c->block_id;
        int ey = c->ey;
        int ex = c->ex;
        int L = c->level;

        const Block* b_ptr = nullptr;
        for (const auto& bk : blocks) {
            if (bk.id == bid) { b_ptr = &bk; break; }
        }
        if (!b_ptr) continue;
        const Block& b = *b_ptr;

        for (int f = 0; f < 4; ++f) {
            int nid = -1;
            int ex_neigh = -1;
            int ey_neigh = -1;
            const NeighborInfo* ni = nullptr;

            if (f == 0) {
                if (ex > 0) { nid = bid; ex_neigh = ex - 1; ey_neigh = ey; }
                else {
                    ni = &b.ni_l;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = (ni->face == 'L') ? 0 : nb->nx * (1 << L) - 1;
                            ey_neigh = ey;
                        }
                    }
                }
            } else if (f == 1) {
                if (ex < b.nx * (1 << L) - 1) { nid = bid; ex_neigh = ex + 1; ey_neigh = ey; }
                else {
                    ni = &b.ni_r;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = (ni->face == 'L') ? 0 : nb->nx * (1 << L) - 1;
                            ey_neigh = ey;
                        }
                    }
                }
            } else if (f == 2) {
                if (ey > 0) { nid = bid; ex_neigh = ex; ey_neigh = ey - 1; }
                else {
                    ni = &b.ni_b;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = ex;
                            ey_neigh = (ni->face == 'B') ? 0 : nb->ny * (1 << L) - 1;
                        }
                    }
                }
            } else if (f == 3) {
                if (ey < b.ny * (1 << L) - 1) { nid = bid; ex_neigh = ex; ey_neigh = ey + 1; }
                else {
                    ni = &b.ni_t;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = ex;
                            ey_neigh = (ni->face == 'B') ? 0 : nb->ny * (1 << L) - 1;
                        }
                    }
                }
            }

            if (nid != -1) {
                uint64_t K = get_morton_id(nid, L, ex_neigh, ey_neigh);
                auto it = std::lower_bound(cells.begin(), cells.end(), K, [](const Cell* cell, uint64_t val) {
                    return cell->morton_id < val;
                });
                if (it != cells.end() && (*it)->morton_id == K) {
                    // Conforming matches are fine
                } else if (it != cells.begin()) {
                    Cell* prev_c = *(it - 1);
                    if (is_ancestor(prev_c->morton_id, K)) {
                        int diff = L - prev_c->level;
                        if (diff > 1) {
                            throw std::runtime_error("2:1 Level Limit Violation: cell at level " + std::to_string(L) +
                                                     " adjacent to cell at level " + std::to_string(prev_c->level));
                        }
                    }
                }
            }
        }
    }
}

void Solver::initialize_cells() {
    int max_block_id = -1;
    for (const auto& b : blocks) {
        if (b.id > max_block_id) max_block_id = b.id;
    }
    block_cells.resize(max_block_id + 1);

    for (const auto& b : blocks) {
        block_cells[b.id] = std::vector<std::vector<Cell*>>(b.ny, std::vector<Cell*>(b.nx, nullptr));
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                Cell* c = new Cell(p.N_PTS, &p);
                c->block_id = b.id;
                c->ey = ey;
                c->ex = ex;
                c->dx = b.dx;
                c->dy = b.dy;
                c->x_min = b.x_min + ex * b.dx;
                c->y_min = b.y_min + ey * b.dy;
                c->x_center = c->x_min + 0.5 * b.dx;
                c->y_center = c->y_min + 0.5 * b.dy;
                c->level = 0;
                c->morton_id = get_morton_id(b.id, 0, ex, ey);
                c->element_time = current_time;
                c->element_dt = 0.0;
                c->element_active = true;

                block_cells[b.id][ey][ex] = c;
                cells.push_back(c);
            }
        }
    }

    // Sort cells by Morton ID
    std::sort(cells.begin(), cells.end(), [](const Cell* a, const Cell* b) {
        return a->morton_id < b->morton_id;
    });

    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i]->cell_index = static_cast<int>(i);
    }

    if (p.ENABLE_NS) {
        global_grad_Ux.assign(N_VARS * cells.size() * p.N_PTS * p.N_PTS, 0.0);
        global_grad_Uy.assign(N_VARS * cells.size() * p.N_PTS * p.N_PTS, 0.0);
    }
}

void Solver::setup_cell_connectivity() {
    for (Cell* c : cells) {
        int bid = c->block_id;
        int ey = c->ey;
        int ex = c->ex;
        int L = c->level;

        const Block* b_ptr = nullptr;
        for (const auto& bk : blocks) {
            if (bk.id == bid) { b_ptr = &bk; break; }
        }
        if (!b_ptr) continue;
        const Block& b = *b_ptr;

        for (int f = 0; f < 4; ++f) {
            c->neighbors[f] = nullptr;
            c->neighbor_faces[f] = ' ';
            c->is_boundary[f] = false;

            int nid = -1;
            int ex_neigh = -1;
            int ey_neigh = -1;
            bool physical_boundary = false;
            const NeighborInfo* ni = nullptr;

            if (f == 0) {
                if (ex > 0) {
                    nid = bid;
                    ex_neigh = ex - 1;
                    ey_neigh = ey;
                } else {
                    ni = &b.ni_l;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = (ni->face == 'L') ? 0 : nb->nx * (1 << L) - 1;
                            ey_neigh = ey;
                        }
                    } else {
                        physical_boundary = true;
                    }
                }
            } else if (f == 1) {
                if (ex < b.nx * (1 << L) - 1) {
                    nid = bid;
                    ex_neigh = ex + 1;
                    ey_neigh = ey;
                } else {
                    ni = &b.ni_r;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = (ni->face == 'L') ? 0 : nb->nx * (1 << L) - 1;
                            ey_neigh = ey;
                        }
                    } else {
                        physical_boundary = true;
                    }
                }
            } else if (f == 2) {
                if (ey > 0) {
                    nid = bid;
                    ex_neigh = ex;
                    ey_neigh = ey - 1;
                } else {
                    ni = &b.ni_b;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = ex;
                            ey_neigh = (ni->face == 'B') ? 0 : nb->ny * (1 << L) - 1;
                        }
                    } else {
                        physical_boundary = true;
                    }
                }
            } else if (f == 3) {
                if (ey < b.ny * (1 << L) - 1) {
                    nid = bid;
                    ex_neigh = ex;
                    ey_neigh = ey + 1;
                } else {
                    ni = &b.ni_t;
                    if (ni->id != -1) {
                        nid = ni->id;
                        const Block* nb = nullptr;
                        for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                        if (nb) {
                            ex_neigh = ex;
                            ey_neigh = (ni->face == 'B') ? 0 : nb->ny * (1 << L) - 1;
                        }
                    } else {
                        physical_boundary = true;
                    }
                }
            }

            if (physical_boundary) {
                c->neighbors[f] = nullptr;
                c->is_boundary[f] = true;
                c->boundary_info[f] = *ni;
            } else if (nid != -1) {
                uint64_t K = get_morton_id(nid, L, ex_neigh, ey_neigh);
                auto it = std::lower_bound(cells.begin(), cells.end(), K, [](const Cell* cell, uint64_t val) {
                    return cell->morton_id < val;
                });

                if (it != cells.end() && (*it)->morton_id == K) {
                    c->neighbors[f] = *it;
                    if (nid == bid) {
                        c->neighbor_faces[f] = (f == 0) ? 'R' : ((f == 1) ? 'L' : ((f == 2) ? 'T' : 'B'));
                    } else {
                        c->neighbor_faces[f] = ni->face;
                    }
                } else if (it != cells.begin()) {
                    Cell* prev_c = *(it - 1);
                    if (is_ancestor(prev_c->morton_id, K)) {
                        c->neighbors[f] = prev_c;
                        if (nid == bid) {
                            c->neighbor_faces[f] = (f == 0) ? 'R' : ((f == 1) ? 'L' : ((f == 2) ? 'T' : 'B'));
                        } else {
                            c->neighbor_faces[f] = ni->face;
                        }
                    }
                }
            }
        }
    }
    // Perform 2:1 ratio validation check
    enforce_21_ratio();
}



void Solver::get_neigh_state_cell(const Cell& c, int node_idx, bool is_right_or_top,
                                  const double* face_state, double sig_face,
                                  double* neigh_state, double& sig_neigh, int dir) const
{
    sig_neigh = 0.0;
    for (int v = 0; v < 4; ++v) neigh_state[v] = 0.0;

    int face_idx = (dir == 0) ? (is_right_or_top ? 1 : 0) : (is_right_or_top ? 3 : 2);
    const NeighborInfo& ni = c.boundary_info[face_idx];

    if (ni.is_noslip_wall || ni.is_moving_wall) {
        double u_w = 0.0, v_w = 0.0;
        if (ni.is_moving_wall) {
            if (dir == 0) v_w = ni.wall_velocity;
            else          u_w = ni.wall_velocity;
        }
        build_viscous_wall_ghost(face_state, neigh_state, u_w, v_w, p.GAMMA,
                                 ni.is_isothermal, ni.wall_temperature);
        sig_neigh = sig_face;
    } else if (ni.is_wall) {
        for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
        if (dir == 0) neigh_state[1] = -face_state[1];
        else          neigh_state[2] = -face_state[2];
        sig_neigh = sig_face;
    } else if (ni.is_supersonic_inflow) {
        neigh_state[0] = ni.ref_rho;
        neigh_state[1] = ni.ref_rho * ni.ref_u;
        neigh_state[2] = ni.ref_rho * ni.ref_v;
        neigh_state[3] = ni.ref_p / (p.GAMMA - 1.0) + 0.5 * ni.ref_rho * (ni.ref_u*ni.ref_u + ni.ref_v*ni.ref_v);
        sig_neigh = 0.0;
    } else if (ni.is_characteristic) {
        double ref_state[4];
        ref_state[0] = ni.ref_rho;
        ref_state[1] = ni.ref_rho * ni.ref_u;
        ref_state[2] = ni.ref_rho * ni.ref_v;
        ref_state[3] = ni.ref_p / (p.GAMMA - 1.0) + 0.5 * ni.ref_rho * (ni.ref_u*ni.ref_u + ni.ref_v*ni.ref_v);
        
        double n_x = 0.0, n_y = 0.0;
        if (dir == 0) n_x = is_right_or_top ? 1.0 : -1.0;
        else          n_y = is_right_or_top ? 1.0 : -1.0;
        
        build_characteristic_ghost(face_state, ref_state, n_x, n_y, p.GAMMA, neigh_state);
        sig_neigh = 0.0;
    } else if (ni.is_total_pressure_comp) {
        build_total_pressure_comp_ghost(face_state, ni.ref_p, p.GAMMA, neigh_state);
        sig_neigh = sig_face;
    } else if (ni.is_total_pressure_incomp) {
        build_total_pressure_incomp_ghost(face_state, ni.ref_p, p.GAMMA, neigh_state);
        sig_neigh = sig_face;
    } else if (ni.is_static_pressure) {
        build_static_pressure_ghost(face_state, ni.ref_p, p.GAMMA, neigh_state);
        sig_neigh = sig_face;
    } else if (ni.is_supersonic_outflow) {
        for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
        sig_neigh = sig_face;
    } else {
        for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
        sig_neigh = sig_face;
    }
}

void Solver::get_flux_pointwise_cell(const Cell& c, int iy, int ix,
                                     double* F, double* G, double sigma) const
{
    double rho   = std::max(p.POS_LIMITER_EPS, c.get_U(0, iy, ix, p.N_PTS));
    double u     = c.get_U(1, iy, ix, p.N_PTS) / rho;
    double v     = c.get_U(2, iy, ix, p.N_PTS) / rho;
    double E     = c.get_U(3, iy, ix, p.N_PTS);
    double press = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));
    if (p.ENABLE_PPR) {
        double P_phan = c.S_field[iy * p.N_PTS + ix] / rho;
        double theta_cfl = (p.PPR_ADAPTIVE_THETA) ? c.theta_avg : p.PPR_THETA;
        if (press - P_phan < 0.0) {
            double theta_safe = (press - p.POS_LIMITER_EPS) / (P_phan - press);
            theta_cfl = std::min(theta_cfl, std::max(0.0, theta_safe));
        }
        double P_reg  = press + theta_cfl * (press - P_phan);
        press = std::max(p.POS_LIMITER_EPS, P_reg);
    }

    if (F) {
        F[0] = rho * u;
        F[1] = rho * u * u + press + sigma;
        F[2] = rho * u * v;
        F[3] = (E + press + sigma) * u;
    }
    if (G) {
        G[0] = rho * v;
        G[1] = rho * v * u;
        G[2] = rho * v * v + press + sigma;
        G[3] = (E + press + sigma) * v;
    }
}

// =========================================================================
// Quadtree AMR: Helper Prolongation / Restriction Functions
// =========================================================================

static void prolong_2d(const std::vector<double>& parent_arr, std::vector<double>& child_arr,
                       const std::vector<std::vector<double>>& PX, const std::vector<std::vector<double>>& PY,
                       int n_vars, int N) {
    for (int v = 0; v < n_vars; ++v) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                double val = 0.0;
                for (int py = 0; py < N; ++py) {
                    for (int px = 0; px < N; ++px) {
                        double p_val = parent_arr[v * N * N + py * N + px];
                        val += p_val * PY[py][iy] * PX[px][ix];
                    }
                }
                child_arr[v * N * N + iy * N + ix] = val;
            }
        }
    }
}

static void prolong_2d_scalar(const std::vector<double>& parent_arr, std::vector<double>& child_arr,
                              const std::vector<std::vector<double>>& PX, const std::vector<std::vector<double>>& PY,
                              int N) {
    for (int iy = 0; iy < N; ++iy) {
        for (int ix = 0; ix < N; ++ix) {
            double val = 0.0;
            for (int py = 0; py < N; ++py) {
                for (int px = 0; px < N; ++px) {
                    double p_val = parent_arr[py * N + px];
                    val += p_val * PY[py][iy] * PX[px][ix];
                }
            }
            child_arr[iy * N + ix] = val;
        }
    }
}

static void restrict_2d(const std::vector<double>& c0_arr, const std::vector<double>& c1_arr,
                        const std::vector<double>& c2_arr, const std::vector<double>& c3_arr,
                        std::vector<double>& parent_arr,
                        const std::vector<std::vector<double>>& R1, const std::vector<std::vector<double>>& R2,
                        int n_vars, int N) {
    for (int v = 0; v < n_vars; ++v) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                double val = 0.0;
                // Child 0: R1 in Y, R1 in X
                for (int qy = 0; qy < N; ++qy) {
                    for (int qx = 0; qx < N; ++qx) {
                        val += c0_arr[v * N * N + qy * N + qx] * R1[qy][iy] * R1[qx][ix];
                    }
                }
                // Child 1: R1 in Y, R2 in X
                for (int qy = 0; qy < N; ++qy) {
                    for (int qx = 0; qx < N; ++qx) {
                        val += c1_arr[v * N * N + qy * N + qx] * R1[qy][iy] * R2[qx][ix];
                    }
                }
                // Child 2: R2 in Y, R1 in X
                for (int qy = 0; qy < N; ++qy) {
                    for (int qx = 0; qx < N; ++qx) {
                        val += c2_arr[v * N * N + qy * N + qx] * R2[qy][iy] * R1[qx][ix];
                    }
                }
                // Child 3: R2 in Y, R2 in X
                for (int qy = 0; qy < N; ++qy) {
                    for (int qx = 0; qx < N; ++qx) {
                        val += c3_arr[v * N * N + qy * N + qx] * R2[qy][iy] * R2[qx][ix];
                    }
                }
                parent_arr[v * N * N + iy * N + ix] = val;
            }
        }
    }
}

static void restrict_2d_scalar(const std::vector<double>& c0_arr, const std::vector<double>& c1_arr,
                               const std::vector<double>& c2_arr, const std::vector<double>& c3_arr,
                               std::vector<double>& parent_arr,
                               const std::vector<std::vector<double>>& R1, const std::vector<std::vector<double>>& R2,
                               int N) {
    for (int iy = 0; iy < N; ++iy) {
        for (int ix = 0; ix < N; ++ix) {
            double val = 0.0;
            // Child 0
            for (int qy = 0; qy < N; ++qy) {
                for (int qx = 0; qx < N; ++qx) {
                    val += c0_arr[qy * N + qx] * R1[qy][iy] * R1[qx][ix];
                }
            }
            // Child 1
            for (int qy = 0; qy < N; ++qy) {
                for (int qx = 0; qx < N; ++qx) {
                    val += c1_arr[qy * N + qx] * R1[qy][iy] * R2[qx][ix];
                }
            }
            // Child 2
            for (int qy = 0; qy < N; ++qy) {
                for (int qx = 0; qx < N; ++qx) {
                    val += c2_arr[qy * N + qx] * R2[qy][iy] * R1[qx][ix];
                }
            }
            // Child 3
            for (int qy = 0; qy < N; ++qy) {
                for (int qx = 0; qx < N; ++qx) {
                    val += c3_arr[qy * N + qx] * R2[qy][iy] * R2[qx][ix];
                }
            }
            parent_arr[iy * N + ix] = val;
        }
    }
}

static void prolong_3d(const std::vector<double>& parent_arr, std::vector<double>& child_arr,
                       const std::vector<std::vector<double>>& PX,
                       const std::vector<std::vector<double>>& PY,
                       const std::vector<std::vector<double>>& PZ,
                       int n_vars, int N) {
    int N3 = N * N * N;
    int N2 = N * N;
    for (int v = 0; v < n_vars; ++v) {
        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    double val = 0.0;
                    for (int pz = 0; pz < N; ++pz) {
                        for (int py = 0; py < N; ++py) {
                            for (int px = 0; px < N; ++px) {
                                double p_val = parent_arr[v * N3 + pz * N2 + py * N + px];
                                val += p_val * PZ[pz][iz] * PY[py][iy] * PX[px][ix];
                            }
                        }
                    }
                    child_arr[v * N3 + iz * N2 + iy * N + ix] = val;
                }
            }
        }
    }
}

static void prolong_3d_scalar(const std::vector<double>& parent_arr, std::vector<double>& child_arr,
                              const std::vector<std::vector<double>>& PX,
                              const std::vector<std::vector<double>>& PY,
                              const std::vector<std::vector<double>>& PZ,
                              int N) {
    int N2 = N * N;
    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                double val = 0.0;
                for (int pz = 0; pz < N; ++pz) {
                    for (int py = 0; py < N; ++py) {
                        for (int px = 0; px < N; ++px) {
                            double p_val = parent_arr[pz * N2 + py * N + px];
                            val += p_val * PZ[pz][iz] * PY[py][iy] * PX[px][ix];
                        }
                    }
                }
                child_arr[iz * N2 + iy * N + ix] = val;
            }
        }
    }
}

static void restrict_3d(const std::vector<const std::vector<double>*>& children_arrs,
                        std::vector<double>& parent_arr,
                        const std::vector<std::vector<double>>& R1, const std::vector<std::vector<double>>& R2,
                        int n_vars, int N) {
    int N3 = N * N * N;
    int N2 = N * N;
    for (int v = 0; v < n_vars; ++v) {
        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    double val = 0.0;
                    for (int c_idx = 0; c_idx < 8; ++c_idx) {
                        const auto& RZ = (c_idx / 4 == 0) ? R1 : R2;
                        const auto& RY = ((c_idx / 2) % 2 == 0) ? R1 : R2;
                        const auto& RX = (c_idx % 2 == 0) ? R1 : R2;
                        const auto& child_arr = *(children_arrs[c_idx]);
                        for (int qz = 0; qz < N; ++qz) {
                            for (int qy = 0; qy < N; ++qy) {
                                for (int qx = 0; qx < N; ++qx) {
                                    val += child_arr[v * N3 + qz * N2 + qy * N + qx] * RZ[qz][iz] * RY[qy][iy] * RX[qx][ix];
                                }
                            }
                        }
                    }
                    parent_arr[v * N3 + iz * N2 + iy * N + ix] = val;
                }
            }
        }
    }
}

static void restrict_3d_scalar(const std::vector<const std::vector<double>*>& children_arrs,
                               std::vector<double>& parent_arr,
                               const std::vector<std::vector<double>>& R1, const std::vector<std::vector<double>>& R2,
                               int N) {
    int N2 = N * N;
    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                double val = 0.0;
                for (int c_idx = 0; c_idx < 8; ++c_idx) {
                    const auto& RZ = (c_idx / 4 == 0) ? R1 : R2;
                    const auto& RY = ((c_idx / 2) % 2 == 0) ? R1 : R2;
                    const auto& RX = (c_idx % 2 == 0) ? R1 : R2;
                    const auto& child_arr = *(children_arrs[c_idx]);
                    for (int qz = 0; qz < N; ++qz) {
                        for (int qy = 0; qy < N; ++qy) {
                            for (int qx = 0; qx < N; ++qx) {
                                val += child_arr[qz * N2 + qy * N + qx] * RZ[qz][iz] * RY[qy][iy] * RX[qx][ix];
                            }
                        }
                    }
                }
                parent_arr[iz * N2 + iy * N + ix] = val;
            }
        }
    }
}

// =========================================================================
// Quadtree AMR: Member Function Implementations
// =========================================================================

Cell* Solver::find_leaf_cell(int block_id, double x, double y) const {
    for (Cell* c : cells) {
        if (c->block_id == block_id) {
            double xmax = c->x_min + c->dx;
            double ymax = c->y_min + c->dy;
            const double eps = 1e-12;
            if (x >= c->x_min - eps && x <= xmax + eps && y >= c->y_min - eps && y <= ymax + eps) {
                return c;
            }
        }
    }
    return nullptr;
}

void Solver::split_cell(Cell* parent, std::vector<Cell*>& new_cells) {
    int N = p.N_PTS;
    int L = parent->level;
    int bid = parent->block_id;
    int ex = parent->ex;
    int ey = parent->ey;

    new_cells.clear();
    new_cells.resize(4, nullptr);

    double dx_c = 0.5 * parent->dx;
    double dy_c = 0.5 * parent->dy;

    for (int c_idx = 0; c_idx < 4; ++c_idx) {
        Cell* child = new Cell(N, &p);
        child->block_id = bid;
        child->level = L + 1;
        child->dx = dx_c;
        child->dy = dy_c;

        int ex_c = 2 * ex + (c_idx % 2);
        int ey_c = 2 * ey + (c_idx / 2);
        child->ex = ex_c;
        child->ey = ey_c;

        child->x_min = parent->x_min + (c_idx % 2) * dx_c;
        child->y_min = parent->y_min + (c_idx / 2) * dy_c;
        child->x_center = child->x_min + 0.5 * dx_c;
        child->y_center = child->y_min + 0.5 * dy_c;
        child->morton_id = get_morton_id(bid, L + 1, ex_c, ey_c);

        child->element_time = parent->element_time;
        child->element_dt = parent->element_dt;
        child->element_active = parent->element_active;
        child->solid_mask = parent->solid_mask;
        child->s_min_val = parent->s_min_val;

        const auto& PX = (c_idx % 2 == 0) ? basis.P1 : basis.P2;
        const auto& PY = (c_idx / 2 == 0) ? basis.P1 : basis.P2;

        prolong_2d(parent->U, child->U, PX, PY, 4, N);
        if (p.ENABLE_IGR) {
            prolong_2d_scalar(parent->sigma_field, child->sigma_field, PX, PY, N);
            prolong_2d_scalar(parent->sigma_old, child->sigma_old, PX, PY, N);
            prolong_2d_scalar(parent->S_buf, child->S_buf, PX, PY, N);
        }
        prolong_2d(parent->U_old, child->U_old, PX, PY, 4, N);
        if (p.ENABLE_MULTIRATE) {
            prolong_2d(parent->U_accum, child->U_accum, PX, PY, 4, N);
        }

        new_cells[c_idx] = child;
    }
}

void Solver::merge_cells(const std::vector<Cell*>& siblings, Cell*& parent) {
    if (siblings.size() != 4) {
        throw std::runtime_error("merge_cells requires exactly 4 sibling cells");
    }

    int N = p.N_PTS;

    Cell* c0 = nullptr;
    Cell* c1 = nullptr;
    Cell* c2 = nullptr;
    Cell* c3 = nullptr;

    for (Cell* sib : siblings) {
        int ex_mod = sib->ex % 2;
        int ey_mod = sib->ey % 2;
        if (ex_mod == 0 && ey_mod == 0) c0 = sib;
        else if (ex_mod == 1 && ey_mod == 0) c1 = sib;
        else if (ex_mod == 0 && ey_mod == 1) c2 = sib;
        else if (ex_mod == 1 && ey_mod == 1) c3 = sib;
    }

    if (!c0 || !c1 || !c2 || !c3) {
        throw std::runtime_error("siblings are not complete or correctly aligned");
    }

    int parent_level = c0->level - 1;
    int parent_ex = c0->ex / 2;
    int parent_ey = c0->ey / 2;

    parent = new Cell(N, &p);
    parent->block_id = c0->block_id;
    parent->level = parent_level;
    parent->dx = 2.0 * c0->dx;
    parent->dy = 2.0 * c0->dy;
    parent->ex = parent_ex;
    parent->ey = parent_ey;
    parent->x_min = c0->x_min;
    parent->y_min = c0->y_min;
    parent->x_center = parent->x_min + 0.5 * parent->dx;
    parent->y_center = parent->y_min + 0.5 * parent->dy;
    parent->morton_id = get_morton_id(c0->block_id, parent_level, parent_ex, parent_ey);

    parent->element_time = c0->element_time;
    parent->element_dt = c0->element_dt;
    parent->element_active = c0->element_active;
    parent->solid_mask = c0->solid_mask;
    parent->s_min_val = c0->s_min_val;

    restrict_2d(c0->U, c1->U, c2->U, c3->U, parent->U, basis.R1, basis.R2, 4, N);
    if (p.ENABLE_IGR) {
        restrict_2d_scalar(c0->sigma_field, c1->sigma_field, c2->sigma_field, c3->sigma_field, parent->sigma_field, basis.R1, basis.R2, N);
        restrict_2d_scalar(c0->sigma_old, c1->sigma_old, c2->sigma_old, c3->sigma_old, parent->sigma_old, basis.R1, basis.R2, N);
        restrict_2d_scalar(c0->S_buf, c1->S_buf, c2->S_buf, c3->S_buf, parent->S_buf, basis.R1, basis.R2, N);
    }
    restrict_2d(c0->U_old, c1->U_old, c2->U_old, c3->U_old, parent->U_old, basis.R1, basis.R2, 4, N);
    if (p.ENABLE_MULTIRATE) {
        restrict_2d(c0->U_accum, c1->U_accum, c2->U_accum, c3->U_accum, parent->U_accum, basis.R1, basis.R2, 4, N);
    }
}

void Solver::update_tree(const std::vector<int>& target_levels) {
    if (cells.size() != target_levels.size()) return;

    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i]->target_level = target_levels[i];
    }

    bool overall_changed = true;
    int global_iter = 0;
    const int max_global_iters = 10;

    while (overall_changed && global_iter < max_global_iters) {
        overall_changed = false;
        global_iter++;

        // 1. Build adjacency list for current cells
        std::map<Cell*, std::vector<Cell*>> adj;
        for (Cell* c : cells) {
            for (int f = 0; f < 4; ++f) {
                if (c->neighbors[f]) {
                    adj[c].push_back(c->neighbors[f]);
                    adj[c->neighbors[f]].push_back(c);
                }
            }
        }

        // 2. Smooth target levels using adjacency
        std::map<Cell*, size_t> cell_to_idx;
        for (size_t i = 0; i < cells.size(); ++i) {
            cell_to_idx[cells[i]] = i;
        }

        bool smooth_changed = true;
        while (smooth_changed) {
            smooth_changed = false;
            for (size_t i = 0; i < cells.size(); ++i) {
                Cell* c = cells[i];
                int current_target = c->target_level;
                for (Cell* nb : adj[c]) {
                    if (nb->target_level < current_target - 1) {
                        nb->target_level = current_target - 1;
                        smooth_changed = true;
                    }
                }
            }
        }

        // 3. Identify splits and merges
        std::unordered_set<Cell*> split_set;
        std::unordered_set<Cell*> merge_set;

        // Group active cells by parent Morton ID
        std::map<uint64_t, std::vector<Cell*>> siblings_map;
        for (Cell* c : cells) {
            if (c->level > 0) {
                uint64_t parent_key = get_morton_id(c->block_id, c->level - 1, c->ex / 2, c->ey / 2);
                siblings_map[parent_key].push_back(c);
            }
        }

        for (Cell* c : cells) {
            if (c->level < c->target_level) {
                split_set.insert(c);
            }
        }

        for (auto const& [parent_key, sibs] : siblings_map) {
            if (sibs.size() == 4) {
                bool all_coarsen = true;
                for (Cell* sib : sibs) {
                    if (sib->target_level >= sib->level || split_set.count(sib)) {
                        all_coarsen = false;
                        break;
                    }
                }
                if (all_coarsen) {
                    for (Cell* sib : sibs) {
                        merge_set.insert(sib);
                    }
                }
            }
        }

        if (split_set.empty() && merge_set.empty()) {
            break;
        }

        // 4. Perform splits and merges
        std::vector<Cell*> next_cells;
        std::unordered_set<Cell*> deleted_set;

        for (Cell* c : cells) {
            if (deleted_set.count(c)) continue;

            if (split_set.count(c)) {
                std::vector<Cell*> children;
                split_cell(c, children);
                for (Cell* child : children) {
                    child->target_level = c->target_level;
                    next_cells.push_back(child);
                }
                deleted_set.insert(c);
                delete c;
                overall_changed = true;
            } else if (merge_set.count(c)) {
                uint64_t parent_key = get_morton_id(c->block_id, c->level - 1, c->ex / 2, c->ey / 2);
                const auto& sibs = siblings_map[parent_key];
                
                if (c->ex % 2 == 0 && c->ey % 2 == 0) {
                    Cell* parent = nullptr;
                    merge_cells(sibs, parent);
                    
                    int max_target = 0;
                    for (Cell* sib : sibs) {
                        max_target = std::max(max_target, sib->target_level);
                        deleted_set.insert(sib);
                    }
                    parent->target_level = max_target;
                    next_cells.push_back(parent);
                    
                    for (Cell* sib : sibs) {
                        delete sib;
                    }
                    overall_changed = true;
                }
            } else {
                next_cells.push_back(c);
            }
        }

        cells = next_cells;

        std::sort(cells.begin(), cells.end(), [](const Cell* a, const Cell* b) {
            return a->morton_id < b->morton_id;
        });

        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i]->cell_index = static_cast<int>(i);
        }

        if (p.ENABLE_NS) {
            global_grad_Ux.assign(N_VARS * cells.size() * p.N_PTS * p.N_PTS, 0.0);
            global_grad_Uy.assign(N_VARS * cells.size() * p.N_PTS * p.N_PTS, 0.0);
        }

        setup_cell_connectivity();
    }
}

void Solver::flag_refinement_coarsening() {
    std::vector<int> target_levels(cells.size(), 0);

    std::map<Cell*, std::vector<Cell*>> adj;
    for (Cell* c : cells) {
        for (int f = 0; f < 4; ++f) {
            if (c->neighbors[f]) {
                adj[c].push_back(c->neighbors[f]);
                adj[c->neighbors[f]].push_back(c);
            }
        }
    }

    if (p.WALL_REFINEMENT_LEVEL > 0 && p.WALL_REFINEMENT_CELLS > 0) {
        std::vector<Cell*> wall_cells;
        for (Cell* c : cells) {
            bool touches_wall = false;
            for (int f = 0; f < 4; ++f) {
                if (c->neighbors[f] == nullptr && c->is_boundary[f]) {
                    const NeighborInfo& ni = c->boundary_info[f];
                    if (ni.refine && (ni.is_wall || ni.is_noslip_wall || ni.is_moving_wall || ni.is_isothermal)) {
                        touches_wall = true;
                        break;
                    }
                }
            }
            if (touches_wall) {
                wall_cells.push_back(c);
            }
        }

        std::map<Cell*, int> dist;
        std::queue<Cell*> q;
        for (Cell* wc : wall_cells) {
            dist[wc] = 1;
            q.push(wc);
        }

        while (!q.empty()) {
            Cell* u = q.front();
            q.pop();
            int d = dist[u];
            if (d < p.WALL_REFINEMENT_CELLS) {
                for (Cell* v : adj[u]) {
                    if (dist.find(v) == dist.end()) {
                        dist[v] = d + 1;
                        q.push(v);
                    }
                }
            }
        }

        for (size_t i = 0; i < cells.size(); ++i) {
            if (dist.find(cells[i]) != dist.end()) {
                target_levels[i] = std::max(target_levels[i], p.WALL_REFINEMENT_LEVEL);
            }
        }
    }

    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        double xmin = c->x_min;
        double xmax = c->x_min + c->dx;
        double ymin = c->y_min;
        double ymax = c->y_min + c->dy;

        for (const auto& zone : p.refinement_zones) {
            bool intersects = false;
            if (zone.shape == "CIRCLE") {
                Geometry::Circle circle;
                circle.cx = zone.center_x;
                circle.cy = zone.center_y;
                circle.r = zone.radius;
                intersects = circle.intersects_aabb(xmin, xmax, ymin, ymax);
            } else if (zone.shape == "NACA") {
                Geometry::Naca naca;
                naca.x_le = zone.center_x;
                naca.y_le = zone.center_y;
                naca.chord = zone.radius;
                naca.naca_code = zone.naca_code;
                naca.aoa_deg = zone.aoa;
                intersects = naca.intersects_aabb(xmin, xmax, ymin, ymax);
            } else if (zone.shape == "POLYGON" || zone.shape == "BOX" || zone.shape == "MULTI") {
                if (zone.shape == "BOX" && zone.poly_x.empty()) {
                    double box_xmin = zone.center_x - 0.5 * zone.width;
                    double box_xmax = zone.center_x + 0.5 * zone.width;
                    double box_ymin = zone.center_y - 0.5 * zone.height;
                    double box_ymax = zone.center_y + 0.5 * zone.height;
                    intersects = !(xmax < box_xmin || xmin > box_xmax || ymax < box_ymin || ymin > box_ymax);
                } else {
                    Geometry::Polygon poly;
                    poly.x = zone.poly_x;
                    poly.y = zone.poly_y;
                    intersects = poly.intersects_aabb(xmin, xmax, ymin, ymax);
                }
            }

            if (intersects) {
                target_levels[i] = std::max(target_levels[i], zone.target_level);
            }
        }
    }

    // Coarsening optimization: If a cell (or its coarser ancestor) is entirely inside the 
    // Immersed Boundary, override its target_level to the coarsest possible level L
    // that is also entirely within the IB.
    if (p.ENABLE_IB) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            for (int L = 0; L <= c->level; ++L) {
                int diff = c->level - L;
                double dx_L = c->dx * (1 << diff);
                double dy_L = c->dy * (1 << diff);
                double xmin_L = c->x_min - (c->ex & ((1 << diff) - 1)) * c->dx;
                double ymin_L = c->y_min - (c->ey & ((1 << diff) - 1)) * c->dy;
                double xc_L = xmin_L + 0.5 * dx_L;
                double yc_L = ymin_L + 0.5 * dy_L;
                double half_diag = 0.5 * std::sqrt(dx_L * dx_L + dy_L * dy_L);
                
                double sdf = get_ib_sdf_at_time(xc_L, yc_L, current_time);
                if (sdf < -half_diag) {
                    target_levels[i] = L;
                    break; // Found the coarsest level
                }
            }
        }
    }

    update_tree(target_levels);
}

// =========================================================================
// SolverDim<3> Member Function Implementations
// =========================================================================

SolverDim<3>::SolverDim(const Parameters& params)
    : p(params), basis(p.P_DEG)
{
    current_time = p.RESTART_TIME;
    blocks.clear();
    for (const auto& config : p.blocks) {
        blocks.emplace_back(config, p.N_PTS);
    }

    initialize_cells();
    setup_cell_connectivity();
}

SolverDim<3>::~SolverDim() {
    for (Cell3D* c : cells) {
        delete c;
    }
    cells.clear();
}

void SolverDim<3>::initialize_cells() {
    for (Cell3D* c : cells) {
        delete c;
    }
    cells.clear();

    block_cells.clear();
    block_cells.resize(blocks.size());

    for (const auto& b : blocks) {
        block_cells[b.id].resize(b.nz);
        for (int ez = 0; ez < b.nz; ++ez) {
            block_cells[b.id][ez].resize(b.ny);
            for (int ey = 0; ey < b.ny; ++ey) {
                block_cells[b.id][ez][ey].resize(b.nx, nullptr);
            }
        }

        for (int ez = 0; ez < b.nz; ++ez) {
            for (int ey = 0; ey < b.ny; ++ey) {
                for (int ex = 0; ex < b.nx; ++ex) {
                    Cell3D* c = new Cell3D(p.N_PTS, &p);
                    c->block_id = b.id;
                    c->ex = ex;
                    c->ey = ey;
                    c->ez = ez;
                    c->dx = b.dx;
                    c->dy = b.dy;
                    c->dz = b.dz;
                    c->x_min = b.x_min + ex * b.dx;
                    c->y_min = b.y_min + ey * b.dy;
                    c->z_min = b.z_min + ez * b.dz;
                    c->x_center = c->x_min + 0.5 * b.dx;
                    c->y_center = c->y_min + 0.5 * b.dy;
                    c->z_center = c->z_min + 0.5 * b.dz;
                    c->level = 0;
                    c->morton_id = get_morton_id(b.id, 0, ex, ey, ez);
                    c->element_time = current_time;
                    c->element_dt = 0.0;
                    c->element_active = true;

                    block_cells[b.id][ez][ey][ex] = c;
                    cells.push_back(c);
                }
            }
        }
    }

    std::sort(cells.begin(), cells.end(), [](const Cell3D* a, const Cell3D* b) {
        return a->morton_id < b->morton_id;
    });

    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i]->cell_index = static_cast<int>(i);
    }

    if (p.ENABLE_NS) {
        global_grad_Ux.assign(Cell3D::N_VARS * cells.size() * p.N_PTS * p.N_PTS * p.N_PTS, 0.0);
        global_grad_Uy.assign(Cell3D::N_VARS * cells.size() * p.N_PTS * p.N_PTS * p.N_PTS, 0.0);
        global_grad_Uz.assign(Cell3D::N_VARS * cells.size() * p.N_PTS * p.N_PTS * p.N_PTS, 0.0);
    }
}

uint64_t SolverDim<3>::get_morton_id(int block_id, int level, uint32_t ex, uint32_t ey, uint32_t ez) const {
    const int L_max = 10;
    const int N_b = 5;
    uint32_t x_fine = ex << (L_max - level);
    uint32_t y_fine = ey << (L_max - level);
    uint32_t z_fine = ez << (L_max - level);
    return ((uint64_t)block_id << 55) | (morton_encode_3d(x_fine, y_fine, z_fine) << N_b) | (uint64_t)level;
}

bool SolverDim<3>::is_ancestor(uint64_t ancestor_id, uint64_t key_id) const {
    int block_id_A = (ancestor_id >> 55);
    int block_id_K = (key_id >> 55);
    if (block_id_A != block_id_K) return false;

    int L_A = ancestor_id & 0x1F;
    int L_K = key_id & 0x1F;
    if (L_A >= L_K) return false;

    uint64_t raw_A = (ancestor_id >> 5) & 0x3FFFFFFFFFFFFULL;
    uint64_t raw_K = (key_id >> 5) & 0x3FFFFFFFFFFFFULL;

    uint32_t x_fine_A, y_fine_A, z_fine_A;
    uint32_t x_fine_K, y_fine_K, z_fine_K;
    morton_decode_3d(raw_A, x_fine_A, y_fine_A, z_fine_A);
    morton_decode_3d(raw_K, x_fine_K, y_fine_K, z_fine_K);

    uint32_t ex_A = x_fine_A >> (10 - L_A);
    uint32_t ey_A = y_fine_A >> (10 - L_A);
    uint32_t ez_A = z_fine_A >> (10 - L_A);
    uint32_t ex_K = x_fine_K >> (10 - L_K);
    uint32_t ey_K = y_fine_K >> (10 - L_K);
    uint32_t ez_K = z_fine_K >> (10 - L_K);

    return ((ex_K >> (L_K - L_A)) == ex_A) &&
           ((ey_K >> (L_K - L_A)) == ey_A) &&
           ((ez_K >> (L_K - L_A)) == ez_A);
}

void SolverDim<3>::enforce_21_ratio() {
    for (Cell3D* c : cells) {
        int bid = c->block_id;
        int ex = c->ex;
        int ey = c->ey;
        int ez = c->ez;
        int L = c->level;

        const Block3D* b_ptr = nullptr;
        for (const auto& bk : blocks) {
            if (bk.id == bid) { b_ptr = &bk; break; }
        }
        if (!b_ptr) continue;
        const Block3D& b = *b_ptr;

        for (int f = 0; f < 6; ++f) {
            int nid = -1;
            int ex_neigh = -1;
            int ey_neigh = -1;
            int ez_neigh = -1;
            const NeighborInfo* ni = nullptr;

            if (f == 0) {
                if (ex > 0) { nid = bid; ex_neigh = ex - 1; ey_neigh = ey; ez_neigh = ez; }
                else { ni = &b.ni_l; if (ni->id != -1) nid = ni->id; }
            } else if (f == 1) {
                if (ex < b.nx * (1 << L) - 1) { nid = bid; ex_neigh = ex + 1; ey_neigh = ey; ez_neigh = ez; }
                else { ni = &b.ni_r; if (ni->id != -1) nid = ni->id; }
            } else if (f == 2) {
                if (ey > 0) { nid = bid; ex_neigh = ex; ey_neigh = ey - 1; ez_neigh = ez; }
                else { ni = &b.ni_b; if (ni->id != -1) nid = ni->id; }
            } else if (f == 3) {
                if (ey < b.ny * (1 << L) - 1) { nid = bid; ex_neigh = ex; ey_neigh = ey + 1; ez_neigh = ez; }
                else { ni = &b.ni_t; if (ni->id != -1) nid = ni->id; }
            } else if (f == 4) {
                if (ez > 0) { nid = bid; ex_neigh = ex; ey_neigh = ey; ez_neigh = ez - 1; }
                else { ni = &b.ni_f; if (ni->id != -1) nid = ni->id; }
            } else if (f == 5) {
                if (ez < b.nz * (1 << L) - 1) { nid = bid; ex_neigh = ex; ey_neigh = ey; ez_neigh = ez + 1; }
                else { ni = &b.ni_k; if (ni->id != -1) nid = ni->id; }
            }

            if (nid != -1) {
                if (ni) {
                    const Block3D* nb = nullptr;
                    for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                    if (nb) {
                        if (f == 0 || f == 1) {
                            ex_neigh = (ni->face == 'L') ? 0 : nb->nx * (1 << L) - 1;
                            ey_neigh = ey;
                            ez_neigh = ez;
                        } else if (f == 2 || f == 3) {
                            ex_neigh = ex;
                            ey_neigh = (ni->face == 'B') ? 0 : nb->ny * (1 << L) - 1;
                            ez_neigh = ez;
                        } else {
                            ex_neigh = ex;
                            ey_neigh = ey;
                            ez_neigh = (ni->face == 'F') ? 0 : nb->nz * (1 << L) - 1;
                        }
                    }
                }

                for (int p_level = 0; p_level <= L + 2; ++p_level) {
                    int scale = (L >= p_level) ? (1 << (L - p_level)) : 1;
                    int div = (p_level > L) ? (1 << (p_level - L)) : 1;
                    uint64_t target_id = get_morton_id(nid, p_level, (L >= p_level) ? (ex_neigh / scale) : (ex_neigh * div),
                                                                     (L >= p_level) ? (ey_neigh / scale) : (ey_neigh * div),
                                                                     (L >= p_level) ? (ez_neigh / scale) : (ez_neigh * div));
                    auto it = std::lower_bound(cells.begin(), cells.end(), target_id, [](const Cell3D* cell, uint64_t val) {
                        return cell->morton_id < val;
                    });
                    if (it != cells.end() && (*it)->morton_id == target_id) {
                        if ((*it)->level > L + 1) {
                            c->target_level = (*it)->level - 1;
                        }
                        break;
                    }
                }
            }
        }
    }
}

void SolverDim<3>::setup_cell_connectivity() {
    for (Cell3D* c : cells) {
        int bid = c->block_id;
        int ex = c->ex;
        int ey = c->ey;
        int ez = c->ez;
        int L = c->level;

        const Block3D* b_ptr = nullptr;
        for (const auto& bk : blocks) {
            if (bk.id == bid) { b_ptr = &bk; break; }
        }
        if (!b_ptr) continue;
        const Block3D& b = *b_ptr;

        for (int f = 0; f < 6; ++f) {
            c->neighbors[f] = nullptr;
            c->neighbor_faces[f] = ' ';
            c->is_boundary[f] = false;

            int nid = -1;
            int ex_neigh = -1;
            int ey_neigh = -1;
            int ez_neigh = -1;
            bool physical_boundary = false;
            const NeighborInfo* ni = nullptr;

            if (f == 0) {
                if (ex > 0) { nid = bid; ex_neigh = ex - 1; ey_neigh = ey; ez_neigh = ez; }
                else { ni = &b.ni_l; if (ni->id != -1) { nid = ni->id; } else { physical_boundary = true; } }
            } else if (f == 1) {
                if (ex < b.nx * (1 << L) - 1) { nid = bid; ex_neigh = ex + 1; ey_neigh = ey; ez_neigh = ez; }
                else { ni = &b.ni_r; if (ni->id != -1) { nid = ni->id; } else { physical_boundary = true; } }
            } else if (f == 2) {
                if (ey > 0) { nid = bid; ex_neigh = ex; ey_neigh = ey - 1; ez_neigh = ez; }
                else { ni = &b.ni_b; if (ni->id != -1) { nid = ni->id; } else { physical_boundary = true; } }
            } else if (f == 3) {
                if (ey < b.ny * (1 << L) - 1) { nid = bid; ex_neigh = ex; ey_neigh = ey + 1; ez_neigh = ez; }
                else { ni = &b.ni_t; if (ni->id != -1) { nid = ni->id; } else { physical_boundary = true; } }
            } else if (f == 4) {
                if (ez > 0) { nid = bid; ex_neigh = ex; ey_neigh = ey; ez_neigh = ez - 1; }
                else { ni = &b.ni_f; if (ni->id != -1) { nid = ni->id; } else { physical_boundary = true; } }
            } else if (f == 5) {
                if (ez < b.nz * (1 << L) - 1) { nid = bid; ex_neigh = ex; ey_neigh = ey; ez_neigh = ez + 1; }
                else { ni = &b.ni_k; if (ni->id != -1) { nid = ni->id; } else { physical_boundary = true; } }
            }

            if (physical_boundary) {
                c->is_boundary[f] = true;
                if (f == 0)      c->boundary_info[f] = b.ni_l;
                else if (f == 1) c->boundary_info[f] = b.ni_r;
                else if (f == 2) c->boundary_info[f] = b.ni_b;
                else if (f == 3) c->boundary_info[f] = b.ni_t;
                else if (f == 4) c->boundary_info[f] = b.ni_f;
                else if (f == 5) c->boundary_info[f] = b.ni_k;
            } else if (nid != -1) {
                if (ni) {
                    const Block3D* nb = nullptr;
                    for (const auto& bk : blocks) { if (bk.id == nid) { nb = &bk; break; } }
                    if (nb) {
                        if (f == 0 || f == 1) {
                            ex_neigh = (ni->face == 'L') ? 0 : nb->nx * (1 << L) - 1;
                            ey_neigh = ey;
                            ez_neigh = ez;
                        } else if (f == 2 || f == 3) {
                            ex_neigh = ex;
                            ey_neigh = (ni->face == 'B') ? 0 : nb->ny * (1 << L) - 1;
                            ez_neigh = ez;
                        } else {
                            ex_neigh = ex;
                            ey_neigh = ey;
                            ez_neigh = (ni->face == 'F') ? 0 : nb->nz * (1 << L) - 1;
                        }
                    }
                }

                uint64_t target_id = get_morton_id(nid, L, ex_neigh, ey_neigh, ez_neigh);
                auto it = std::lower_bound(cells.begin(), cells.end(), target_id, [](const Cell3D* cell, uint64_t val) {
                    return cell->morton_id < val;
                });

                if (it != cells.end() && (*it)->morton_id == target_id) {
                    c->neighbors[f] = *it;
                    c->neighbor_faces[f] = (f == 0) ? 'R' : (f == 1) ? 'L' : (f == 2) ? 'T' : (f == 3) ? 'B' : (f == 4) ? 'K' : 'F';
                } else {
                    bool found = false;
                    for (int p_level = L - 1; p_level >= 0; --p_level) {
                        int scale = 1 << (L - p_level);
                        uint64_t p_id = get_morton_id(nid, p_level, ex_neigh / scale, ey_neigh / scale, ez_neigh / scale);
                        auto it2 = std::lower_bound(cells.begin(), cells.end(), p_id, [](const Cell3D* cell, uint64_t val) {
                            return cell->morton_id < val;
                        });
                        if (it2 != cells.end() && (*it2)->morton_id == p_id) {
                            c->neighbors[f] = *it2;
                            c->neighbor_faces[f] = (f == 0) ? 'R' : (f == 1) ? 'L' : (f == 2) ? 'T' : (f == 3) ? 'B' : (f == 4) ? 'K' : 'F';
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

void SolverDim<3>::split_cell(Cell3D* parent, std::vector<Cell3D*>& new_cells) {
    int N = p.N_PTS;
    int L = parent->level;
    int bid = parent->block_id;
    int ex = parent->ex;
    int ey = parent->ey;
    int ez = parent->ez;

    new_cells.clear();
    new_cells.resize(8, nullptr);

    double dx_c = 0.5 * parent->dx;
    double dy_c = 0.5 * parent->dy;
    double dz_c = 0.5 * parent->dz;

    for (int c_idx = 0; c_idx < 8; ++c_idx) {
        Cell3D* child = new Cell3D(N, &p);
        child->block_id = bid;
        child->level = L + 1;
        child->dx = dx_c;
        child->dy = dy_c;
        child->dz = dz_c;

        int ex_c = 2 * ex + (c_idx % 2);
        int ey_c = 2 * ey + ((c_idx / 2) % 2);
        int ez_c = 2 * ez + (c_idx / 4);
        child->ex = ex_c;
        child->ey = ey_c;
        child->ez = ez_c;

        child->x_min = parent->x_min + (c_idx % 2) * dx_c;
        child->y_min = parent->y_min + ((c_idx / 2) % 2) * dy_c;
        child->z_min = parent->z_min + (c_idx / 4) * dz_c;
        child->x_center = child->x_min + 0.5 * dx_c;
        child->y_center = child->y_min + 0.5 * dy_c;
        child->z_center = child->z_min + 0.5 * dz_c;
        child->morton_id = get_morton_id(bid, L + 1, ex_c, ey_c, ez_c);

        child->element_time = parent->element_time;
        child->element_dt = parent->element_dt;
        child->element_active = parent->element_active;
        child->solid_mask = parent->solid_mask;
        child->s_min_val = parent->s_min_val;

        const auto& PX = (c_idx % 2 == 0) ? basis.P1 : basis.P2;
        const auto& PY = ((c_idx / 2) % 2 == 0) ? basis.P1 : basis.P2;
        const auto& PZ = (c_idx / 4 == 0) ? basis.P1 : basis.P2;

        prolong_3d(parent->U, child->U, PX, PY, PZ, Cell3D::N_VARS, N);
        if (p.ENABLE_IGR) {
            prolong_3d_scalar(parent->sigma_field, child->sigma_field, PX, PY, PZ, N);
            prolong_3d_scalar(parent->sigma_old, child->sigma_old, PX, PY, PZ, N);
            prolong_3d_scalar(parent->S_buf, child->S_buf, PX, PY, PZ, N);
        }
        prolong_3d(parent->U_old, child->U_old, PX, PY, PZ, Cell3D::N_VARS, N);
        if (p.ENABLE_MULTIRATE) {
            prolong_3d(parent->U_accum, child->U_accum, PX, PY, PZ, Cell3D::N_VARS, N);
        }

        new_cells[c_idx] = child;
    }
}

void SolverDim<3>::merge_cells(const std::vector<Cell3D*>& siblings, Cell3D*& parent) {
    if (siblings.size() != 8) {
        throw std::runtime_error("merge_cells requires exactly 8 sibling cells in 3D");
    }

    int N = p.N_PTS;

    std::vector<Cell3D*> sorted_sibs(8, nullptr);
    for (Cell3D* sib : siblings) {
        int ex_mod = sib->ex % 2;
        int ey_mod = sib->ey % 2;
        int ez_mod = sib->ez % 2;
        int c_idx = ez_mod * 4 + ey_mod * 2 + ex_mod;
        sorted_sibs[c_idx] = sib;
    }

    for (int i = 0; i < 8; ++i) {
        if (!sorted_sibs[i]) {
            throw std::runtime_error("siblings are not complete or correctly aligned in 3D");
        }
    }

    Cell3D* c0 = sorted_sibs[0];
    int parent_level = c0->level - 1;
    int parent_ex = c0->ex / 2;
    int parent_ey = c0->ey / 2;
    int parent_ez = c0->ez / 2;

    parent = new Cell3D(N, &p);
    parent->block_id = c0->block_id;
    parent->level = parent_level;
    parent->dx = 2.0 * c0->dx;
    parent->dy = 2.0 * c0->dy;
    parent->dz = 2.0 * c0->dz;
    parent->ex = parent_ex;
    parent->ey = parent_ey;
    parent->ez = parent_ez;
    parent->x_min = c0->x_min;
    parent->y_min = c0->y_min;
    parent->z_min = c0->z_min;
    parent->x_center = parent->x_min + 0.5 * parent->dx;
    parent->y_center = parent->y_min + 0.5 * parent->dy;
    parent->z_center = parent->z_min + 0.5 * parent->dz;
    parent->morton_id = get_morton_id(c0->block_id, parent_level, parent_ex, parent_ey, parent_ez);

    parent->element_time = c0->element_time;
    parent->element_dt = c0->element_dt;
    parent->element_active = c0->element_active;
    parent->solid_mask = c0->solid_mask;
    parent->s_min_val = c0->s_min_val;

    std::vector<const std::vector<double>*> u_ptrs(8), sig_ptrs(8), sig_old_ptrs(8), s_buf_ptrs(8), u_old_ptrs(8), u_accum_ptrs(8);
    for (int i = 0; i < 8; ++i) {
        u_ptrs[i] = &sorted_sibs[i]->U;
        sig_ptrs[i] = &sorted_sibs[i]->sigma_field;
        sig_old_ptrs[i] = &sorted_sibs[i]->sigma_old;
        s_buf_ptrs[i] = &sorted_sibs[i]->S_buf;
        u_old_ptrs[i] = &sorted_sibs[i]->U_old;
        u_accum_ptrs[i] = &sorted_sibs[i]->U_accum;
    }

    restrict_3d(u_ptrs, parent->U, basis.R1, basis.R2, Cell3D::N_VARS, N);
    if (p.ENABLE_IGR) {
        restrict_3d_scalar(sig_ptrs, parent->sigma_field, basis.R1, basis.R2, N);
        restrict_3d_scalar(sig_old_ptrs, parent->sigma_old, basis.R1, basis.R2, N);
        restrict_3d_scalar(s_buf_ptrs, parent->S_buf, basis.R1, basis.R2, N);
    }
    restrict_3d(u_old_ptrs, parent->U_old, basis.R1, basis.R2, Cell3D::N_VARS, N);
    if (p.ENABLE_MULTIRATE) {
        restrict_3d(u_accum_ptrs, parent->U_accum, basis.R1, basis.R2, Cell3D::N_VARS, N);
    }
}

void SolverDim<3>::update_tree(const std::vector<int>& target_levels) {
    if (cells.size() != target_levels.size()) return;

    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i]->target_level = target_levels[i];
    }

    bool overall_changed = true;
    int global_iter = 0;
    const int max_global_iters = 10;

    while (overall_changed && global_iter < max_global_iters) {
        overall_changed = false;
        global_iter++;

        std::map<Cell3D*, std::vector<Cell3D*>> adj;
        for (Cell3D* c : cells) {
            for (int f = 0; f < 6; ++f) {
                if (c->neighbors[f]) {
                    adj[c].push_back(c->neighbors[f]);
                }
            }
        }

        bool tree_changed = true;
        int propagation_iter = 0;
        while (tree_changed && propagation_iter < 100) {
            tree_changed = false;
            propagation_iter++;

            for (Cell3D* c : cells) {
                for (Cell3D* neigh : adj[c]) {
                    if (c->target_level > neigh->target_level + 1) {
                        neigh->target_level = c->target_level - 1;
                        tree_changed = true;
                    }
                }
            }
        }

        std::vector<Cell3D*> next_cells;
        bool changes_applied = false;

        std::map<uint64_t, std::vector<Cell3D*>> parent_groups;
        for (Cell3D* c : cells) {
            if (c->target_level < c->level) {
                int parent_ex = c->ex / 2;
                int parent_ey = c->ey / 2;
                int parent_ez = c->ez / 2;
                uint64_t parent_id = get_morton_id(c->block_id, c->level - 1, parent_ex, parent_ey, parent_ez);
                parent_groups[parent_id].push_back(c);
            }
        }

        std::vector<bool> processed(cells.size(), false);

        for (Cell3D* c : cells) {
            if (processed[c->cell_index]) continue;

            if (c->target_level > c->level) {
                std::vector<Cell3D*> children;
                split_cell(c, children);
                next_cells.insert(next_cells.end(), children.begin(), children.end());
                processed[c->cell_index] = true;
                delete c;
                changes_applied = true;
                overall_changed = true;
            } else if (c->target_level < c->level) {
                int parent_ex = c->ex / 2;
                int parent_ey = c->ey / 2;
                int parent_ez = c->ez / 2;
                uint64_t parent_id = get_morton_id(c->block_id, c->level - 1, parent_ex, parent_ey, parent_ez);
                const auto& group = parent_groups[parent_id];

                if (group.size() == 8) {
                    bool all_merge = true;
                    for (Cell3D* sib : group) {
                        if (sib->target_level >= sib->level) {
                            all_merge = false;
                            break;
                        }
                    }

                    if (all_merge) {
                        Cell3D* parent = nullptr;
                        merge_cells(group, parent);
                        next_cells.push_back(parent);
                        for (Cell3D* sib : group) {
                            processed[sib->cell_index] = true;
                            delete sib;
                        }
                        changes_applied = true;
                        overall_changed = true;
                    } else {
                        next_cells.push_back(c);
                        processed[c->cell_index] = true;
                    }
                } else {
                    next_cells.push_back(c);
                    processed[c->cell_index] = true;
                }
            } else {
                next_cells.push_back(c);
                processed[c->cell_index] = true;
            }
        }

        if (changes_applied) {
            cells = next_cells;
            std::sort(cells.begin(), cells.end(), [](const Cell3D* a, const Cell3D* b) {
                return a->morton_id < b->morton_id;
            });
            for (size_t i = 0; i < cells.size(); ++i) {
                cells[i]->cell_index = static_cast<int>(i);
            }
            setup_cell_connectivity();
        }
    }
}

void SolverDim<3>::flag_refinement_coarsening() {}

Cell3D* SolverDim<3>::find_leaf_cell(int block_id, double x, double y, double z) const {
    for (Cell3D* c : cells) {
        if (c->block_id == block_id) {
            double xmax = c->x_min + c->dx;
            double ymax = c->y_min + c->dy;
            double zmax = c->z_min + c->dz;
            const double eps = 1e-12;
            if (x >= c->x_min - eps && x <= xmax + eps &&
                y >= c->y_min - eps && y <= ymax + eps &&
                z >= c->z_min - eps && z <= zmax + eps) {
                return c;
            }
        }
    }
    return nullptr;
}

void SolverDim<3>::get_neigh_state_cell(const Cell3D& c, int node_idx, bool is_right_or_top,
                                        const double* face_state, double sig_face,
                                        double* neigh_state, double& sig_neigh, int dir) const
{
    sig_neigh = 0.0;
    for (int v = 0; v < 5; ++v) neigh_state[v] = 0.0;

    int face_idx = (dir == 0) ? (is_right_or_top ? 1 : 0) :
                   ((dir == 1) ? (is_right_or_top ? 3 : 2) :
                    (is_right_or_top ? 5 : 4));
    const NeighborInfo& ni = c.boundary_info[face_idx];

    if (ni.is_noslip_wall || ni.is_moving_wall) {
        double u_w = 0.0, v_w = 0.0, w_w = 0.0;
        if (ni.is_moving_wall) {
            if (dir == 0) v_w = ni.wall_velocity;
            else          u_w = ni.wall_velocity;
        }
        
        double rho_f = std::max(p.POS_LIMITER_EPS, face_state[0]);
        double u_f = face_state[1] / rho_f;
        double v_f = face_state[2] / rho_f;
        double w_f = face_state[3] / rho_f;
        
        double u_g = 2.0 * u_w - u_f;
        double v_g = 2.0 * v_w - v_f;
        double w_g = 2.0 * w_w - w_f;
        
        double p_f = (p.GAMMA - 1.0) * (face_state[4] - 0.5 * rho_f * (u_f*u_f + v_f*v_f + w_f*w_f));
        double T_f = std::max(p.POS_LIMITER_EPS, p_f / rho_f);
        double T_g = ni.is_isothermal ? ni.wall_temperature : T_f;
        
        double rho_g = rho_f;
        double p_g = rho_g * T_g;
        
        neigh_state[0] = rho_g;
        neigh_state[1] = rho_g * u_g;
        neigh_state[2] = rho_g * v_g;
        neigh_state[3] = rho_g * w_g;
        neigh_state[4] = p_g / (p.GAMMA - 1.0) + 0.5 * rho_g * (u_g*u_g + v_g*v_g + w_g*w_g);
        
        sig_neigh = sig_face;
    } else if (ni.is_wall) {
        for (int v = 0; v < 5; ++v) neigh_state[v] = face_state[v];
        neigh_state[1 + dir] = -face_state[1 + dir];
        sig_neigh = sig_face;
    } else if (ni.is_supersonic_inflow) {
        neigh_state[0] = ni.ref_rho;
        neigh_state[1] = ni.ref_rho * ni.ref_u;
        neigh_state[2] = ni.ref_rho * ni.ref_v;
        neigh_state[3] = ni.ref_rho * ni.ref_w;
        neigh_state[4] = ni.ref_p / (p.GAMMA - 1.0) + 0.5 * ni.ref_rho * (ni.ref_u*ni.ref_u + ni.ref_v*ni.ref_v + ni.ref_w*ni.ref_w);
        sig_neigh = 0.0;
    } else if (ni.is_characteristic) {
        double ref_state[5];
        ref_state[0] = ni.ref_rho;
        ref_state[1] = ni.ref_rho * ni.ref_u;
        ref_state[2] = ni.ref_rho * ni.ref_v;
        ref_state[3] = ni.ref_rho * ni.ref_w;
        ref_state[4] = ni.ref_p / (p.GAMMA - 1.0) + 0.5 * ni.ref_rho * (ni.ref_u*ni.ref_u + ni.ref_v*ni.ref_v + ni.ref_w*ni.ref_w);
        
        double n_x = 0.0, n_y = 0.0, n_z = 0.0;
        if (dir == 0) n_x = is_right_or_top ? 1.0 : -1.0;
        else if (dir == 1) n_y = is_right_or_top ? 1.0 : -1.0;
        else if (dir == 2) n_z = is_right_or_top ? 1.0 : -1.0;
        
        double rho_f = std::max(p.POS_LIMITER_EPS, face_state[0]);
        double u_f = face_state[1] / rho_f;
        double v_f = face_state[2] / rho_f;
        double w_f = face_state[3] / rho_f;
        double un_f = u_f * n_x + v_f * n_y + w_f * n_z;
        double p_f = (p.GAMMA - 1.0) * (face_state[4] - 0.5 * rho_f * (u_f*u_f + v_f*v_f + w_f*w_f));
        double c_f = std::sqrt(p.GAMMA * std::max(p.POS_LIMITER_EPS, p_f) / rho_f);
        
        double rho_r = ni.ref_rho;
        double u_r = ni.ref_u;
        double v_r = ni.ref_v;
        double w_r = ni.ref_w;
        double un_r = u_r * n_x + v_r * n_y + w_r * n_z;
        double c_r = std::sqrt(p.GAMMA * ni.ref_p / rho_r);
        
        double R_plus = un_f + 2.0 * c_f / (p.GAMMA - 1.0);
        double R_minus = un_r - 2.0 * c_r / (p.GAMMA - 1.0);
        
        double un, c, rho, u, v, w;
        if (un_f > 0.0) {
            un = 0.5 * (R_plus + R_minus);
            c = 0.25 * (p.GAMMA - 1.0) * (R_plus - R_minus);
            double entropy = p_f / std::pow(rho_f, p.GAMMA);
            rho = std::pow(c*c / (p.GAMMA * entropy), 1.0 / (p.GAMMA - 1.0));
            u = u_f + (un - un_f) * n_x;
            v = v_f + (un - un_f) * n_y;
            w = w_f + (un - un_f) * n_z;
        } else {
            un = 0.5 * (R_plus + R_minus);
            c = 0.25 * (p.GAMMA - 1.0) * (R_plus - R_minus);
            double entropy = ni.ref_p / std::pow(rho_r, p.GAMMA);
            rho = std::pow(c*c / (p.GAMMA * entropy), 1.0 / (p.GAMMA - 1.0));
            u = u_r + (un - un_r) * n_x;
            v = v_r + (un - un_r) * n_y;
            w = w_r + (un - un_r) * n_z;
        }
        
        neigh_state[0] = rho;
        neigh_state[1] = rho * u;
        neigh_state[2] = rho * v;
        neigh_state[3] = rho * w;
        neigh_state[4] = rho * c*c / (p.GAMMA * (p.GAMMA - 1.0)) + 0.5 * rho * (u*u + v*v + w*w);
        sig_neigh = 0.0;
    } else if (ni.is_total_pressure_comp || ni.is_total_pressure_incomp || ni.is_static_pressure) {
        double rho_f = std::max(p.POS_LIMITER_EPS, face_state[0]);
        double u_f = face_state[1] / rho_f;
        double v_f = face_state[2] / rho_f;
        double w_f = face_state[3] / rho_f;
        double p_g = ni.ref_p;
        neigh_state[0] = rho_f;
        neigh_state[1] = face_state[1];
        neigh_state[2] = face_state[2];
        neigh_state[3] = face_state[3];
        neigh_state[4] = p_g / (p.GAMMA - 1.0) + 0.5 * rho_f * (u_f*u_f + v_f*v_f + w_f*w_f);
        sig_neigh = sig_face;
    } else {
        for (int v = 0; v < 5; ++v) neigh_state[v] = face_state[v];
        sig_neigh = sig_face;
    }
}

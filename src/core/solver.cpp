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
#include <stdexcept>
#include "../ib/sbm_geometry.hpp"

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
static void parse_bc_string(const std::string& bc, NeighborInfo& ni) {
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

Solver::Solver(const Parameters& params)
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

    // Compute initial IB mask field
    if (p.ENABLE_IB) {
        update_ib_mask_field(current_time);
        if (p.IB_METHOD == "SBM") {
            ImmersedBoundary::initialize_sbm_geometry(*this);
        }
    }
}

/// Assemble the full right-hand side: IGR + inviscid sweeps + viscous fluxes.
void Solver::compute_rhs() {
    #pragma omp parallel
    {
        for (auto& b : blocks) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < b.RHS.data.size(); ++i) {
                b.RHS.data[i] = 0.0;
            }
        }
        compute_entropic_pressure();
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
            for (auto& b : blocks) {
                if (b.solid_mask.empty()) continue;
                #pragma omp for schedule(static)
                for (int ey = 0; ey < b.ny; ++ey) {
                    for (int ex = 0; ex < b.nx; ++ex) {
                        if (b.solid_mask[ey * b.nx + ex]) {
                            for (int v = 0; v < N_VARS; ++v) {
                                for (int iy = 0; iy < p.N_PTS; ++iy) {
                                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                                        b.RHS(v, ey, ex, iy, ix) = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// @file solver.cpp
/// @brief Solver constructor and compute_rhs dispatcher.

#include "solver.hpp"

static void parse_bc_string(const std::string& bc, NeighborInfo& ni) {
    if (bc == "WALL") ni.is_wall = true;
    else if (bc == "INFLOW") ni.is_inflow = true;
    else if (bc == "TRANSMISSIVE") ni.is_transmissive = true;
    else if (bc.find(':') != std::string::npos) {
        size_t sep = bc.find(':');
        ni.id = std::stoi(bc.substr(0, sep));
        ni.face = bc[sep + 1];
    } else {
        ni.is_transmissive = true;
    }
}

Solver::Solver(const Parameters& params)
    : p(params), basis(p.P_DEG)
{
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
}

/// Assemble the full right-hand side: IGR + X-sweep + Y-sweep.
void Solver::compute_rhs() {
    for (auto& b : blocks) {
        std::fill(b.RHS.data.begin(), b.RHS.data.end(), 0.0);
    }
    compute_entropic_pressure();
    sweep_x();
    sweep_y();
}

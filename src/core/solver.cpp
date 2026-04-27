/// @file solver.cpp
/// @brief Solver constructor and compute_rhs dispatcher.

#include "solver.hpp"

Solver::Solver(const Parameters& params)
    : p(params), U(p), RHS(p), basis(p)
{
    dx = (p.X_MAX - p.X_MIN) / p.N_ELEM_X;
    dy = (p.Y_MAX - p.Y_MIN) / p.N_ELEM_Y;

    int total_nodes = p.N_ELEM_Y * p.N_ELEM_X * p.N_PTS * p.N_PTS;
    sigma_field .resize(total_nodes, 0.0);
    S_buf       .resize(total_nodes, 0.0);
    sigma_xy_buf.resize(total_nodes, 0.0);
    sigma_yx_buf.resize(total_nodes, 0.0);
    sigma_RHS   .resize(total_nodes, 0.0);
    qx_buf      .resize(total_nodes, 0.0);
    qy_buf      .resize(total_nodes, 0.0);
}

/// Assemble the full right-hand side: IGR + X-sweep + Y-sweep.
void Solver::compute_rhs() {
    std::fill(RHS.data.begin(), RHS.data.end(), 0.0);
    compute_entropic_pressure();
    sweep_x();
    sweep_y();
}

/// @file boundary.cpp
/// @brief Neighbour-state extraction with boundary-condition handling.
///
/// For each element face the solver needs the state on the other side of the
/// interface.  At domain boundaries the ghost state is constructed from the
/// BC type (WALL → reflect normal velocity, TRANSMISSIVE → copy-out,
/// PERIODIC → wrap-around).

#include "../core/solver.hpp"

// =========================================================================
// X-direction (left / right) neighbour state
// =========================================================================

// =========================================================================
// X-direction (left / right) neighbour state
// =========================================================================

void Solver::get_neigh_state_x(const Block& b, int ey, int ex, int iy, bool is_right,
                                const double* face_state, double sig_face,
                                double* neigh_state, double& sig_neigh) const
{
    sig_neigh = 0.0;
    for (int v = 0; v < 4; ++v) neigh_state[v] = 0.0;

    const NeighborInfo& ni = is_right ? b.ni_r : b.ni_l;

    if (!is_right) {
        // ---- Left interface ----
        if (ex == 0) {
            if (ni.id != -1) {
                const Block& nb = blocks[ni.id];
                int nex = (ni.face == 'L') ? 0 : nb.nx - 1;
                const double* weights = (ni.face == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += nb.U(v, ey, nex, iy, k) * weights[k];
                    sig_neigh += nb.sigma_field[nb.get_flat_idx(ey, nex, iy, k, p.N_PTS)] * weights[k];
                }
            } else if (ni.is_periodic) {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += b.U(v, ey, b.nx - 1, iy, k) * basis.l_R[k];
                    sig_neigh += b.sigma_field[b.get_flat_idx(ey, b.nx - 1, iy, k, p.N_PTS)] * basis.l_R[k];
                }
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[1] = -face_state[1];
                sig_neigh = sig_face;
            } else if (ni.is_inflow) {
                neigh_state[0] = p.RHO_INF;
                neigh_state[1] = p.RHO_INF * p.U_INF;
                neigh_state[2] = p.RHO_INF * p.V_INF;
                neigh_state[3] = p.P_INF / (p.GAMMA - 1.0) + 0.5 * p.RHO_INF * (p.U_INF*p.U_INF + p.V_INF*p.V_INF);
                sig_neigh = 0.0;
            } else { // TRANSMISSIVE default
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += b.U(v, ey, ex - 1, iy, k) * basis.l_R[k];
                sig_neigh += b.sigma_field[b.get_flat_idx(ey, ex - 1, iy, k, p.N_PTS)] * basis.l_R[k];
            }
        }
    } else {
        // ---- Right interface ----
        if (ex == b.nx - 1) {
            if (ni.id != -1) {
                const Block& nb = blocks[ni.id];
                int nex = (ni.face == 'L') ? 0 : nb.nx - 1;
                const double* weights = (ni.face == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += nb.U(v, ey, nex, iy, k) * weights[k];
                    sig_neigh += nb.sigma_field[nb.get_flat_idx(ey, nex, iy, k, p.N_PTS)] * weights[k];
                }
            } else if (ni.is_periodic) {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += b.U(v, ey, 0, iy, k) * basis.l_L[k];
                    sig_neigh += b.sigma_field[b.get_flat_idx(ey, 0, iy, k, p.N_PTS)] * basis.l_L[k];
                }
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[1] = -face_state[1];
                sig_neigh = sig_face;
            } else if (ni.is_inflow) {
                neigh_state[0] = p.RHO_INF;
                neigh_state[1] = p.RHO_INF * p.U_INF;
                neigh_state[2] = p.RHO_INF * p.V_INF;
                neigh_state[3] = p.P_INF / (p.GAMMA - 1.0) + 0.5 * p.RHO_INF * (p.U_INF*p.U_INF + p.V_INF*p.V_INF);
                sig_neigh = 0.0;
            } else {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += b.U(v, ey, ex + 1, iy, k) * basis.l_L[k];
                sig_neigh += b.sigma_field[b.get_flat_idx(ey, ex + 1, iy, k, p.N_PTS)] * basis.l_L[k];
            }
        }
    }
}

// =========================================================================
// Y-direction (bottom / top) neighbour state
// =========================================================================

void Solver::get_neigh_state_y(const Block& b, int ey, int ex, int ix, bool is_top,
                                const double* face_state, double sig_face,
                                double* neigh_state, double& sig_neigh) const
{
    sig_neigh = 0.0;
    for (int v = 0; v < 4; ++v) neigh_state[v] = 0.0;

    const NeighborInfo& ni = is_top ? b.ni_t : b.ni_b;

    if (!is_top) {
        // ---- Bottom interface ----
        if (ey == 0) {
            if (ni.id != -1) {
                const Block& nb = blocks[ni.id];
                int ney = (ni.face == 'B') ? 0 : nb.ny - 1;
                const double* weights = (ni.face == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += nb.U(v, ney, ex, k, ix) * weights[k];
                    sig_neigh += nb.sigma_field[nb.get_flat_idx(ney, ex, k, ix, p.N_PTS)] * weights[k];
                }
            } else if (ni.is_periodic) {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += b.U(v, b.ny - 1, ex, k, ix) * basis.l_R[k];
                    sig_neigh += b.sigma_field[b.get_flat_idx(b.ny - 1, ex, k, ix, p.N_PTS)] * basis.l_R[k];
                }
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[2] = -face_state[2];
                sig_neigh = sig_face;
            } else if (ni.is_inflow) {
                neigh_state[0] = p.RHO_INF;
                neigh_state[1] = p.RHO_INF * p.U_INF;
                neigh_state[2] = p.RHO_INF * p.V_INF;
                neigh_state[3] = p.P_INF / (p.GAMMA - 1.0) + 0.5 * p.RHO_INF * (p.U_INF*p.U_INF + p.V_INF*p.V_INF);
                sig_neigh = 0.0;
            } else {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += b.U(v, ey - 1, ex, k, ix) * basis.l_R[k];
                sig_neigh += b.sigma_field[b.get_flat_idx(ey - 1, ex, k, ix, p.N_PTS)] * basis.l_R[k];
            }
        }
    } else {
        // ---- Top interface ----
        if (ey == b.ny - 1) {
            if (ni.id != -1) {
                const Block& nb = blocks[ni.id];
                int ney = (ni.face == 'B') ? 0 : nb.ny - 1;
                const double* weights = (ni.face == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += nb.U(v, ney, ex, k, ix) * weights[k];
                    sig_neigh += nb.sigma_field[nb.get_flat_idx(ney, ex, k, ix, p.N_PTS)] * weights[k];
                }
            } else if (ni.is_periodic) {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += b.U(v, 0, ex, k, ix) * basis.l_L[k];
                    sig_neigh += b.sigma_field[b.get_flat_idx(0, ex, k, ix, p.N_PTS)] * basis.l_L[k];
                }
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[2] = -face_state[2];
                sig_neigh = sig_face;
            } else if (ni.is_inflow) {
                neigh_state[0] = p.RHO_INF;
                neigh_state[1] = p.RHO_INF * p.U_INF;
                neigh_state[2] = p.RHO_INF * p.V_INF;
                neigh_state[3] = p.P_INF / (p.GAMMA - 1.0) + 0.5 * p.RHO_INF * (p.U_INF*p.U_INF + p.V_INF*p.V_INF);
                sig_neigh = 0.0;
            } else {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += b.U(v, ey + 1, ex, k, ix) * basis.l_L[k];
                sig_neigh += b.sigma_field[b.get_flat_idx(ey + 1, ex, k, ix, p.N_PTS)] * basis.l_L[k];
            }
        }
    }
}

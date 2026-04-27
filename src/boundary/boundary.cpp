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

void Solver::get_neigh_state_x(int ey, int ex, int iy, bool is_right,
                                const double* face_state, double sig_face,
                                double* neigh_state, double& sig_neigh) const
{
    sig_neigh = 0.0;
    for (int v = 0; v < 4; ++v) neigh_state[v] = 0.0;

    if (!is_right) {
        // ---- Left interface ----
        if (ex == 0) {
            if (p.BC_L == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += U(v, ey, p.N_ELEM_X - 1, iy, k) * basis.l_R[k];
                    sig_neigh += sigma_field[get_flat_idx(ey, p.N_ELEM_X - 1, iy, k)] * basis.l_R[k];
                }
            } else if (p.BC_L == "WALL") {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[1] = -face_state[1];   // reflect u-momentum
                sig_neigh = sig_face;
            } else { // TRANSMISSIVE
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += U(v, ey, ex - 1, iy, k) * basis.l_R[k];
                sig_neigh += sigma_field[get_flat_idx(ey, ex - 1, iy, k)] * basis.l_R[k];
            }
        }
    } else {
        // ---- Right interface ----
        if (ex == p.N_ELEM_X - 1) {
            if (p.BC_R == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += U(v, ey, 0, iy, k) * basis.l_L[k];
                    sig_neigh += sigma_field[get_flat_idx(ey, 0, iy, k)] * basis.l_L[k];
                }
            } else if (p.BC_R == "WALL") {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[1] = -face_state[1];
                sig_neigh = sig_face;
            } else {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += U(v, ey, ex + 1, iy, k) * basis.l_L[k];
                sig_neigh += sigma_field[get_flat_idx(ey, ex + 1, iy, k)] * basis.l_L[k];
            }
        }
    }
}

// =========================================================================
// Y-direction (bottom / top) neighbour state
// =========================================================================

void Solver::get_neigh_state_y(int ey, int ex, int ix, bool is_top,
                                const double* face_state, double sig_face,
                                double* neigh_state, double& sig_neigh) const
{
    sig_neigh = 0.0;
    for (int v = 0; v < 4; ++v) neigh_state[v] = 0.0;

    if (!is_top) {
        // ---- Bottom interface ----
        if (ey == 0) {
            if (p.BC_B == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += U(v, p.N_ELEM_Y - 1, ex, k, ix) * basis.l_R[k];
                    sig_neigh += sigma_field[get_flat_idx(p.N_ELEM_Y - 1, ex, k, ix)] * basis.l_R[k];
                }
            } else if (p.BC_B == "WALL") {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[2] = -face_state[2];   // reflect v-momentum
                sig_neigh = sig_face;
            } else {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += U(v, ey - 1, ex, k, ix) * basis.l_R[k];
                sig_neigh += sigma_field[get_flat_idx(ey - 1, ex, k, ix)] * basis.l_R[k];
            }
        }
    } else {
        // ---- Top interface ----
        if (ey == p.N_ELEM_Y - 1) {
            if (p.BC_T == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        neigh_state[v] += U(v, 0, ex, k, ix) * basis.l_L[k];
                    sig_neigh += sigma_field[get_flat_idx(0, ex, k, ix)] * basis.l_L[k];
                }
            } else if (p.BC_T == "WALL") {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[2] = -face_state[2];
                sig_neigh = sig_face;
            } else {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
            }
        } else {
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v)
                    neigh_state[v] += U(v, ey + 1, ex, k, ix) * basis.l_L[k];
                sig_neigh += sigma_field[get_flat_idx(ey + 1, ex, k, ix)] * basis.l_L[k];
            }
        }
    }
}

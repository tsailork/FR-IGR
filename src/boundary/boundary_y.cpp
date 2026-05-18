/**
 * @file boundary_y.cpp
 * @brief Implementation of Y-direction boundary and neighbor extraction logic.
 */

#include "../core/solver.hpp"
#include "boundary.hpp"

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

            } else if (ni.is_noslip_wall || ni.is_moving_wall) {
                // Viscous wall: tangential direction for B/T walls is x
                double u_w = 0.0, v_w = 0.0;
                if (ni.is_moving_wall) u_w = ni.wall_velocity;
                build_viscous_wall_ghost(face_state, neigh_state, u_w, v_w, p.GAMMA,
                                         ni.is_isothermal, ni.wall_temperature);
                sig_neigh = sig_face;
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[2] = -face_state[2];
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
                build_characteristic_ghost(face_state, ref_state, 0.0, -1.0, p.GAMMA, neigh_state);
                sig_neigh = 0.0;
            } else if (ni.is_supersonic_outflow) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
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

            } else if (ni.is_noslip_wall || ni.is_moving_wall) {
                double u_w = 0.0, v_w = 0.0;
                if (ni.is_moving_wall) u_w = ni.wall_velocity;
                build_viscous_wall_ghost(face_state, neigh_state, u_w, v_w, p.GAMMA,
                                         ni.is_isothermal, ni.wall_temperature);
                sig_neigh = sig_face;
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[2] = -face_state[2];
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
                build_characteristic_ghost(face_state, ref_state, 0.0, 1.0, p.GAMMA, neigh_state);
                sig_neigh = 0.0;
            } else if (ni.is_supersonic_outflow) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                sig_neigh = sig_face;
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

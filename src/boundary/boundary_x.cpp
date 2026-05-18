/**
 * @file boundary_x.cpp
 * @brief Implementation of X-direction boundary and neighbor extraction logic.
 */

#include "../core/solver.hpp"
#include "boundary.hpp"

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

            } else if (ni.is_noslip_wall || ni.is_moving_wall) {
                // Viscous wall: tangential direction for L/R walls is y
                double u_w = 0.0, v_w = 0.0;
                if (ni.is_moving_wall) v_w = ni.wall_velocity;
                build_viscous_wall_ghost(face_state, neigh_state, u_w, v_w, p.GAMMA,
                                         ni.is_isothermal, ni.wall_temperature);
                sig_neigh = sig_face;
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[1] = -face_state[1];
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
                build_characteristic_ghost(face_state, ref_state, -1.0, 0.0, p.GAMMA, neigh_state);
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

            } else if (ni.is_noslip_wall || ni.is_moving_wall) {
                double u_w = 0.0, v_w = 0.0;
                if (ni.is_moving_wall) v_w = ni.wall_velocity;
                build_viscous_wall_ghost(face_state, neigh_state, u_w, v_w, p.GAMMA,
                                         ni.is_isothermal, ni.wall_temperature);
                sig_neigh = sig_face;
            } else if (ni.is_wall) {
                for (int v = 0; v < 4; ++v) neigh_state[v] = face_state[v];
                neigh_state[1] = -face_state[1];
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
                build_characteristic_ghost(face_state, ref_state, 1.0, 0.0, p.GAMMA, neigh_state);
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
                    neigh_state[v] += b.U(v, ey, ex + 1, iy, k) * basis.l_L[k];
                sig_neigh += b.sigma_field[b.get_flat_idx(ey, ex + 1, iy, k, p.N_PTS)] * basis.l_L[k];
            }
        }
    }
}

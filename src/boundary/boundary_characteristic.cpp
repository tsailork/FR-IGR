/**
 * @file boundary_characteristic.cpp
 * @brief Implementation of characteristic (Riemann invariant) boundary condition ghost state generator.
 */

#include "boundary.hpp"
#include <algorithm>
#include <cmath>

void build_characteristic_ghost(const double* face_state, const double* ref_state,
                                double n_x, double n_y, double gamma,
                                double* neigh_state)
{
    // Interior primitives
    double rho_i = std::max(1e-14, face_state[0]);
    double u_i   = face_state[1] / rho_i;
    double v_i   = face_state[2] / rho_i;
    double p_i   = (gamma - 1.0) * (face_state[3] - 0.5 * rho_i * (u_i*u_i + v_i*v_i));
    if (p_i < 1e-14) p_i = 1e-14;
    double c_i   = std::sqrt(gamma * p_i / rho_i);
    double un_i  = u_i * n_x + v_i * n_y;
    double ut_i  = u_i * (-n_y) + v_i * n_x; // tangential velocity
    double s_i   = p_i / std::pow(rho_i, gamma);

    // Reference (exterior) primitives
    double rho_e = std::max(1e-14, ref_state[0]);
    double u_e   = ref_state[1] / rho_e;
    double v_e   = ref_state[2] / rho_e;
    double p_e   = (gamma - 1.0) * (ref_state[3] - 0.5 * rho_e * (u_e*u_e + v_e*v_e));
    if (p_e < 1e-14) p_e = 1e-14;
    double c_e   = std::sqrt(gamma * p_e / rho_e);
    double un_e  = u_e * n_x + v_e * n_y;
    double ut_e  = u_e * (-n_y) + v_e * n_x;
    double s_e   = p_e / std::pow(rho_e, gamma);

    // Riemann invariants (upwinded based on local normal wave propagation directions)
    double R_plus  = (un_i + c_i > 0.0) ? (un_i + 2.0 * c_i / (gamma - 1.0)) : (un_e + 2.0 * c_e / (gamma - 1.0));
    double R_minus = (un_i - c_i > 0.0) ? (un_i - 2.0 * c_i / (gamma - 1.0)) : (un_e - 2.0 * c_e / (gamma - 1.0));

    // Boundary normal state and sound speed
    double un_b = 0.5 * (R_plus + R_minus);
    double c_b  = 0.25 * (gamma - 1.0) * (R_plus - R_minus);
    
    // Entropy and tangential velocity come from upstream/upwind direction
    double s_b, ut_b;
    if (un_b > 0.0) {
        // Outflow: transport physical properties from interior
        s_b  = s_i;
        ut_b = ut_i;
    } else {
        // Inflow: transport reference state properties from exterior
        s_b  = s_e;
        ut_b = ut_e;
    }

    // Reconstruct conservative ghost state variables
    double rho_b = std::pow(c_b * c_b / (gamma * s_b), 1.0 / (gamma - 1.0));
    double p_b   = rho_b * c_b * c_b / gamma;
    
    double u_b = un_b * n_x - ut_b * n_y;
    double v_b = un_b * n_y + ut_b * n_x;

    neigh_state[0] = rho_b;
    neigh_state[1] = rho_b * u_b;
    neigh_state[2] = rho_b * v_b;
    neigh_state[3] = p_b / (gamma - 1.0) + 0.5 * rho_b * (u_b*u_b + v_b*v_b);
}

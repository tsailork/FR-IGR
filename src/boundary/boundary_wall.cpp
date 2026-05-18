/**
 * @file boundary_wall.cpp
 * @brief Implementation of viscous wall boundary condition ghost state generator.
 */

#include "boundary.hpp"
#include <algorithm>
#include <cmath>

void build_viscous_wall_ghost(const double* face_state, double* neigh_state,
                              double u_wall, double v_wall, double gamma,
                              bool isothermal, double T_wall)
{
    // Retrieve interior primitives safely
    double rho = std::max(1e-14, face_state[0]);
    double u = face_state[1] / rho;
    double v = face_state[2] / rho;
    double p = (gamma - 1.0) * (face_state[3] - 0.5 * rho * (u*u + v*v));
    if (p < 1e-14) p = 1e-14;
    double T_int = p / rho;

    // Ghost velocity: mirror about wall velocity values
    double u_g = 2.0 * u_wall - u;
    double v_g = 2.0 * v_wall - v;
    double rho_g = rho;

    double T_g;
    if (isothermal) {
        T_g = 2.0 * T_wall - T_int;
        if (T_g < 1e-14) T_g = 1e-14;
    } else {
        // Adiabatic wall: copy internal temperature across interface
        T_g = T_int;
    }

    double p_g = rho_g * T_g;

    neigh_state[0] = rho_g;
    neigh_state[1] = rho_g * u_g;
    neigh_state[2] = rho_g * v_g;
    neigh_state[3] = p_g / (gamma - 1.0) + 0.5 * rho_g * (u_g*u_g + v_g*v_g);
}

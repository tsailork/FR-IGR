/**
 * @file boundary_backpressure.cpp
 * @brief Implementations of specialized boundary conditions for backpressure and plenum supplies.
 */

#include "boundary.hpp"
#include <cmath>
#include <algorithm>

void build_total_pressure_comp_ghost(const double* face_state, double Pt_target,
                                     double gamma, double* neigh_state)
{
    // Extract state
    double rho = face_state[0];
    double rhou = face_state[1];
    double rhov = face_state[2];
    double E = face_state[3];

    // Compute velocity & pressure
    double u = rhou / rho;
    double v = rhov / rho;
    double V2 = u*u + v*v;
    double P = (gamma - 1.0) * (E - 0.5 * rho * V2);

    // Guard negative pressure/density
    if (P <= 1e-10) P = 1e-10;
    if (rho <= 1e-10) rho = 1e-10;

    // Compressible Mach number and Pt
    double c2 = gamma * P / rho;
    double M2 = V2 / c2;
    double term = 1.0 + 0.5 * (gamma - 1.0) * M2;
    double Pt_int = P * std::pow(term, gamma / (gamma - 1.0));

    // Scaling factor alpha
    double alpha = 1.0;
    if (Pt_int > 1e-10) {
        alpha = Pt_target / Pt_int;
    }
    if (alpha < 1e-6) alpha = 1e-6; // Clamp to avoid non-physical negative/near-zero values

    // Scale entire state
    neigh_state[0] = alpha * rho;
    neigh_state[1] = alpha * rhou;
    neigh_state[2] = alpha * rhov;
    neigh_state[3] = alpha * E;
}

void build_total_pressure_incomp_ghost(const double* face_state, double Pt_target,
                                       double gamma, double* neigh_state)
{
    // Extract state
    double rho = face_state[0];
    double rhou = face_state[1];
    double rhov = face_state[2];
    double E = face_state[3];

    // Compute velocity & pressure
    double u = rhou / rho;
    double v = rhov / rho;
    double V2 = u*u + v*v;
    double P = (gamma - 1.0) * (E - 0.5 * rho * V2);

    // Guard negative pressure/density
    if (P <= 1e-10) P = 1e-10;
    if (rho <= 1e-10) rho = 1e-10;

    // Incompressible Bernoulli Pt
    double Pt_int = P + 0.5 * rho * V2;

    // Scaling factor alpha
    double alpha = 1.0;
    if (Pt_int > 1e-10) {
        alpha = Pt_target / Pt_int;
    }
    if (alpha < 1e-6) alpha = 1e-6; // Clamp to avoid non-physical negative/near-zero values

    // Scale entire state
    neigh_state[0] = alpha * rho;
    neigh_state[1] = alpha * rhou;
    neigh_state[2] = alpha * rhov;
    neigh_state[3] = alpha * E;
}

void build_static_pressure_ghost(const double* face_state, double P_target,
                                 double gamma, double* neigh_state)
{
    // Extract state
    double rho = face_state[0];
    double rhou = face_state[1];
    double rhov = face_state[2];

    // Compute velocity
    double u = rhou / rho;
    double v = rhov / rho;
    double V2 = u*u + v*v;

    // Guard negative density
    if (rho <= 1e-10) rho = 1e-10;

    // Ghost state
    neigh_state[0] = rho;
    neigh_state[1] = rhou;
    neigh_state[2] = rhov;
    neigh_state[3] = P_target / (gamma - 1.0) + 0.5 * rho * V2;
}

/// @file euler_flux.cpp
/// @brief Pointwise Euler flux evaluation and Rusanov Riemann solver.
///
/// These functions are called at every solution and face point during
/// the X- and Y-sweeps.  They are small, scalar routines that benefit
/// from inlining by the compiler at -O2 / -O3.

#include "../core/solver.hpp"

// =========================================================================
// Pointwise physical flux (with entropic pressure σ)
// =========================================================================

/// Compute the Euler flux at a single solution point.
///
/// @details
/// Data Structures & Indexing:
///   - U(v, ey, ex, iy, ix): The global state array. `v` is the variable (rho, rho*u, rho*v, E). 
///     `ey`, `ex` are global element indices. `iy`, `ix` are local solution point indices.
///   - F, G: Output arrays of size 4. Flat 1D arrays for the local physical fluxes.
/// Assumptions:
///   - `U` contains physically valid states (handled by limiters). Density and pressure are 
///     clamped to a small positive floor (1e-10) to prevent division by zero or NaN wavespeeds.
///
/// @param F  Output: X-direction flux vector (4 components), or nullptr.
/// @param G  Output: Y-direction flux vector (4 components), or nullptr.
/// @param sigma  Local entropic pressure value.
void Solver::get_flux_pointwise(const Block& b, int ey, int ex, int iy, int ix,
                                 double* F, double* G, double sigma) const
{
    double rho   = std::max(1e-10, b.U(0, ey, ex, iy, ix));
    double u     = b.U(1, ey, ex, iy, ix) / rho;
    double v     = b.U(2, ey, ex, iy, ix) / rho;
    double E     = b.U(3, ey, ex, iy, ix);
    double press = std::max(1e-10, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));

    if (F) {
        F[0] = rho * u;
        F[1] = rho * u * u + press + sigma;
        F[2] = rho * u * v;
        F[3] = (E + press + sigma) * u;
    }
    if (G) {
        G[0] = rho * v;
        G[1] = rho * v * u;
        G[2] = rho * v * v + press + sigma;
        G[3] = (E + press + sigma) * v;
    }
}

// =========================================================================
// Rusanov / Local Lax-Friedrichs Riemann solver
// =========================================================================

/// Compute the common interface flux via the Rusanov approximation.
///
/// @details
/// Data Structures & Indexing:
///   - UL, UR: Flat arrays of size 4 containing the extrapolated conservative state at the left 
///     and right sides of an element interface.
///   - F_comm: Flat array of size 4 to store the resulting common numerical flux.
/// Assumptions:
///   - UL and UR have already been computed by interpolating the interior solution points to the face.
///   - Density and pressure are clamped to a small positive floor (1e-10) to prevent NaNs.
///
/// @param UL, UR  Left and right conserved states (4 components each).
/// @param F_comm  Output: common flux at the interface.
/// @param dir     0 = X-direction, 1 = Y-direction.
/// @param sigl, sigr  Left / right entropic pressure values.
void Solver::solve_riemann(const double* UL, const double* UR, double* F_comm,
                            int dir, double sigl, double sigr) const
{
    double rhoL = std::max(1e-10, UL[0]);
    double uL = UL[1] / rhoL, vL = UL[2] / rhoL;
    double pL = std::max(1e-10, (p.GAMMA - 1) * (UL[3] - 0.5 * rhoL * (uL*uL + vL*vL)));

    double rhoR = std::max(1e-10, UR[0]);
    double uR = UR[1] / rhoR, vR = UR[2] / rhoR;
    double pR = std::max(1e-10, (p.GAMMA - 1) * (UR[3] - 0.5 * rhoR * (uR*uR + vR*vR)));

    double vnL = (dir == 0) ? uL : vL;
    double vnR = (dir == 0) ? uR : vR;

    double cL = std::sqrt(p.GAMMA * pL / rhoL);
    double cR = std::sqrt(p.GAMMA * pR / rhoR);
    double max_wave = std::max(std::abs(vnL) + cL, std::abs(vnR) + cR);

    double FL[4], FR[4];
    if (dir == 0) {
        FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL + sigl;
        FL[2] = rhoL*uL*vL;  FL[3] = (UL[3] + pL + sigl) * uL;
        FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR + sigr;
        FR[2] = rhoR*uR*vR;  FR[3] = (UR[3] + pR + sigr) * uR;
    } else {
        FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;
        FL[2] = rhoL*vL*vL + pL + sigl;  FL[3] = (UL[3] + pL + sigl) * vL;
        FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;
        FR[2] = rhoR*vR*vR + pR + sigr;  FR[3] = (UR[3] + pR + sigr) * vR;
    }

    for (int v = 0; v < 4; ++v)
        F_comm[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * max_wave * (UR[v] - UL[v]);
}

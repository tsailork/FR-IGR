/**
 * @file euler_flux.cpp
 * @brief Pointwise Euler flux evaluation and Rusanov Riemann solver.
 *
 * These functions are called at every solution and face point during
 * the X- and Y-sweeps. They are small, scalar routines that benefit
 * from inlining by the compiler at -O2 / -O3.
 *
 * @see sweep_x.cpp
 * @see sweep_y.cpp
 */

#include "../core/solver.hpp"

// =========================================================================
// Pointwise physical flux (with entropic pressure σ)
// =========================================================================

/**
 * @brief Compute the Euler flux at a single solution point.
 *
 * @details
 * Evaluates the pointwise physical flux \f$ F(U) \f$ and \f$ G(U) \f$ for the 2D Euler equations.
 * Incorporates the Isotropic Gradient Regularization (IGR) entropic pressure \f$ \sigma \f$ 
 * into the momentum and energy fluxes to handle shocks.
 *
 * Data Structures & Indexing:
 *   - U(v, ey, ex, iy, ix): The global state array. `v` is the variable (rho, rho*u, rho*v, E). 
 *     `ey`, `ex` are global element indices. `iy`, `ix` are local solution point indices.
 *   - F, G: Output arrays of size 4. Flat 1D arrays for the local physical fluxes.
 * Assumptions:
 *   - `U` contains physically valid states (handled by limiters like the Zhang-Shu bounds-preserving limiter). 
 *     Density and pressure are clamped to a small positive floor (1e-10) to prevent division by zero or NaN wavespeeds.
 *
 * @param b      Reference to the Block containing the state.
 * @param ey     Element Y index.
 * @param ex     Element X index.
 * @param iy     Solution point Y index.
 * @param ix     Solution point X index.
 * @param F      Output: X-direction flux vector (4 components), or nullptr.
 * @param G      Output: Y-direction flux vector (4 components), or nullptr.
 * @param sigma  Local entropic pressure value (\f$ \sigma \f$).
 *
 * @note Relies on prior application of Zhang-Shu bounds-preserving limiters to ensure \f$ \rho > 0, p > 0 \f$.
 * @see Limiters::apply_positivity_limiter
 */
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

/**
 * @brief Compute the common interface flux via the Rusanov (Local Lax-Friedrichs) approximation.
 *
 * @details
 * Evaluates the Rusanov Riemann solver at element interfaces:
 * \f[ F_{comm} = \frac{1}{2}(F(U_L) + F(U_R)) - \frac{1}{2} \max(|\lambda_L|, |\lambda_R|) (U_R - U_L) \f]
 * where \f$ \lambda = |v_n| + c \f$ is the maximum wave speed.
 * 
 * Data Structures & Indexing:
 *   - UL, UR: Flat arrays of size 4 containing the extrapolated conservative state at the left 
 *     and right sides of an element interface.
 *   - F_comm: Flat array of size 4 to store the resulting common numerical flux.
 * Assumptions:
 *   - UL and UR have already been computed by interpolating the interior solution points to the face.
 *   - Density and pressure are clamped to a small positive floor (1e-10) to prevent NaNs.
 *
 * @param UL      Left conserved state vector (4 components).
 * @param UR      Right conserved state vector (4 components).
 * @param F_comm  Output: Common numerical flux at the interface.
 * @param dir     Direction flag (0 = X-direction, 1 = Y-direction).
 * @param sigl    Left entropic pressure value (\f$ \sigma_L \f$).
 * @param sigr    Right entropic pressure value (\f$ \sigma_R \f$).
 *
 * @see Solver::sweep_x
 * @see Solver::sweep_y
 */
void Solver::solve_riemann(const double* UL, const double* UR, double* F_comm,
                           int dir, double SL, double SR, double thetaL, double thetaR) const
{
    double rhoL = std::max(p.POS_LIMITER_EPS, UL[0]);
    double uL = UL[1] / rhoL, vL = UL[2] / rhoL;
    double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UL[3] - 0.5 * rhoL * (uL*uL + vL*vL)));

    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    double uR = UR[1] / rhoR, vR = UR[2] / rhoR;
    double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UR[3] - 0.5 * rhoR * (uR*uR + vR*vR)));

    double vnL = (dir == 0) ? uL : vL;
    double vnR = (dir == 0) ? uR : vR;

    double cL, cR;
    if (p.ENABLE_PPR) {
        double pL_phan = SL / rhoL;
        double pR_phan = SR / rhoR;
        double theta_cfl_L = (p.PPR_ADAPTIVE_THETA) ? thetaL : p.PPR_THETA;
        double theta_cfl_R = (p.PPR_ADAPTIVE_THETA) ? thetaR : p.PPR_THETA;

        if (pL - pL_phan < 0.0) {
            double theta_safe = (pL - p.POS_LIMITER_EPS) / (pL_phan - pL);
            theta_cfl_L = std::min(theta_cfl_L, std::max(0.0, theta_safe));
        }
        if (pR - pR_phan < 0.0) {
            double theta_safe = (pR - p.POS_LIMITER_EPS) / (pR_phan - pR);
            theta_cfl_R = std::min(theta_cfl_R, std::max(0.0, theta_safe));
        }

        double pL_reg = pL + theta_cfl_L * (pL - pL_phan);
        double pR_reg = pR + theta_cfl_R * (pR - pR_phan);

        double pL_safe = std::max(pL, pL_reg);
        double pR_safe = std::max(pR, pR_reg);

        pL = std::max(p.POS_LIMITER_EPS, pL_reg);
        pR = std::max(p.POS_LIMITER_EPS, pR_reg);

        cL = std::sqrt(p.GAMMA * pL_safe / rhoL);
        cR = std::sqrt(p.GAMMA * pR_safe / rhoR);
    } else {
        cL = std::sqrt(p.GAMMA * pL / rhoL);
        cR = std::sqrt(p.GAMMA * pR / rhoR);
    }

    double max_wave = std::max(std::abs(vnL) + cL, std::abs(vnR) + cR);

    double FL[4], FR[4];
    if (dir == 0) {
        FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL;
        FL[2] = rhoL*uL*vL;  FL[3] = (UL[3] + pL) * uL;
        FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR;
        FR[2] = rhoR*uR*vR;  FR[3] = (UR[3] + pR) * uR;
    } else {
        FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;
        FL[2] = rhoL*vL*vL + pL;  FL[3] = (UL[3] + pL) * vL;
        FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;
        FR[2] = rhoR*vR*vR + pR;  FR[3] = (UR[3] + pR) * vR;
    }

    for (int v = 0; v < 4; ++v)
        F_comm[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * max_wave * (UR[v] - UL[v]);
}

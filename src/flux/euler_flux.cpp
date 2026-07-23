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

void Solver::compute_interface_flux(const double* UL, const double* UR,
                                    double sigL, double sigR,
                                    double SL, double SR,
                                    double thetaL, double thetaR,
                                    int dir,
                                    double* Flux_comm, double& Flux_S_comm) const
{
    // 1. Solve Riemann fluxes
    solve_riemann(UL, UR, Flux_comm, dir, SL, SR, thetaL, thetaR);

    // 2. Add entropic pressure face contributions to momentum and energy fluxes
    double rhoL = std::max(p.POS_LIMITER_EPS, UL[0]);
    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    double vnL = (dir == 0) ? (UL[1] / rhoL) : (UL[2] / rhoL);
    double vnR = (dir == 0) ? (UR[1] / rhoR) : (UR[2] / rhoR);

    Flux_comm[1 + dir] += 0.5 * (sigL + sigR);
    Flux_comm[3] += 0.5 * (sigL * vnL + sigR * vnR);

    // 3. Compute PPR advection flux if enabled
    if (p.ENABLE_PPR) {
        double uL = UL[1 + dir] / rhoL;
        double uR = UR[1 + dir] / rhoR;

        // Compute local sound speeds to estimate wave speed for phantom pressure advection
        double keL = 0.0, keR = 0.0;
        for (int d = 0; d < 2; ++d) {
            double vL = UL[1 + d] / rhoL;
            double vR = UR[1 + d] / rhoR;
            keL += vL * vL;
            keR += vR * vR;
        }
        double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UL[3] - 0.5 * rhoL * keL));
        double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UR[3] - 0.5 * rhoR * keR));

        double pL_reg = pL + thetaL * (pL - SL / rhoL);
        double pR_reg = pR + thetaR * (pR - SR / rhoR);

        double cL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) / rhoL);
        double cR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) / rhoR);

        double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(uL) + cL, std::abs(uR) + cR);
        Flux_S_comm = 0.5 * p.PPR_ADV_MULT * (SL * uL + SR * uR) - 0.5 * lam * (SR - SL);
    } else {
        Flux_S_comm = 0.0;
    }
}

// =========================================================================
// SolverDim<3> 3D Euler Flux and Riemann Solver Implementations
// =========================================================================

void SolverDim<3>::get_flux_pointwise(const Block3D& b, int ez, int ey, int ex, int iz, int iy, int ix,
                                     double* F, double* G, double* H, double sigma) const
{
    double rho   = std::max(1e-10, b.U(0, ez, ey, ex, iz, iy, ix));
    double u     = b.U(1, ez, ey, ex, iz, iy, ix) / rho;
    double v     = b.U(2, ez, ey, ex, iz, iy, ix) / rho;
    double w     = b.U(3, ez, ey, ex, iz, iy, ix) / rho;
    double E     = b.U(4, ez, ey, ex, iz, iy, ix);
    double press = std::max(1e-10, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v + w*w)));

    if (F) {
        F[0] = rho * u;
        F[1] = rho * u * u + press + sigma;
        F[2] = rho * u * v;
        F[3] = rho * u * w;
        F[4] = (E + press + sigma) * u;
    }
    if (G) {
        G[0] = rho * v;
        G[1] = rho * v * u;
        G[2] = rho * v * v + press + sigma;
        G[3] = rho * v * w;
        G[4] = (E + press + sigma) * v;
    }
    if (H) {
        H[0] = rho * w;
        H[1] = rho * w * u;
        H[2] = rho * w * v;
        H[3] = rho * w * w + press + sigma;
        H[4] = (E + press + sigma) * w;
    }
}

void SolverDim<3>::solve_riemann(const double* UL, const double* UR, double* F_comm,
                                 int dir, double SL, double SR, double thetaL, double thetaR) const
{
    double rhoL = std::max(p.POS_LIMITER_EPS, UL[0]);
    double uL = UL[1] / rhoL, vL = UL[2] / rhoL, wL = UL[3] / rhoL;
    double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UL[4] - 0.5 * rhoL * (uL*uL + vL*vL + wL*wL)));

    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    double uR = UR[1] / rhoR, vR = UR[2] / rhoR, wR = UR[3] / rhoR;
    double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1) * (UR[4] - 0.5 * rhoR * (uR*uR + vR*vR + wR*wR)));

    double vnL = (dir == 0) ? uL : ((dir == 1) ? vL : wL);
    double vnR = (dir == 0) ? uR : ((dir == 1) ? vR : wR);

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

    double FL[5], FR[5];
    if (dir == 0) {
        FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL;  FL[2] = rhoL*uL*vL;  FL[3] = rhoL*uL*wL;  FL[4] = (UL[4] + pL) * uL;
        FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR;  FR[2] = rhoR*uR*vR;  FR[3] = rhoR*uR*wR;  FR[4] = (UR[4] + pR) * uR;
    } else if (dir == 1) {
        FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;  FL[2] = rhoL*vL*vL + pL;  FL[3] = rhoL*vL*wL;  FL[4] = (UL[4] + pL) * vL;
        FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;  FR[2] = rhoR*vR*vR + pR;  FR[3] = rhoR*vR*wR;  FR[4] = (UR[4] + pR) * vR;
    } else {
        FL[0] = rhoL*wL;  FL[1] = rhoL*wL*uL;  FL[2] = rhoL*wL*vL;  FL[3] = rhoL*wL*wL + pL;  FL[4] = (UL[4] + pL) * wL;
        FR[0] = rhoR*wR;  FR[1] = rhoR*wR*uR;  FR[2] = rhoR*wR*vR;  FR[3] = rhoR*wR*wR + pR;  FR[4] = (UR[4] + pR) * wR;
    }

    for (int v = 0; v < 5; ++v)
        F_comm[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * max_wave * (UR[v] - UL[v]);
}

void SolverDim<3>::compute_interface_flux(const double* UL, const double* UR,
                                         double sigL, double sigR,
                                         double SL, double SR,
                                         double thetaL, double thetaR,
                                         int dir,
                                         double* Flux_comm, double& Flux_S_comm) const
{
    solve_riemann(UL, UR, Flux_comm, dir, SL, SR, thetaL, thetaR);

    double rhoL = std::max(p.POS_LIMITER_EPS, UL[0]);
    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    double vnL = (dir == 0) ? (UL[1] / rhoL) : ((dir == 1) ? (UL[2] / rhoL) : (UL[3] / rhoL));
    double vnR = (dir == 0) ? (UR[1] / rhoR) : ((dir == 1) ? (UR[2] / rhoR) : (UR[3] / rhoR));

    Flux_comm[1 + dir] += 0.5 * (sigL + sigR);
    Flux_comm[4] += 0.5 * (sigL * vnL + sigR * vnR);

    if (p.ENABLE_PPR) {
        double uL = (dir == 0) ? (UL[1] / rhoL) : ((dir == 1) ? (UL[2] / rhoL) : (UL[3] / rhoL));
        double uR = (dir == 0) ? (UR[1] / rhoR) : ((dir == 1) ? (UR[2] / rhoR) : (UR[3] / rhoR));

        double keL = 0.0, keR = 0.0;
        for (int d = 0; d < 3; ++d) {
            double velL = UL[1 + d] / rhoL;
            double velR = UR[1 + d] / rhoR;
            keL += velL * velL;
            keR += velR * velR;
        }
        double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UL[4] - 0.5 * rhoL * keL));
        double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UR[4] - 0.5 * rhoR * keR));

        double pL_reg = pL + thetaL * (pL - SL / rhoL);
        double pR_reg = pR + thetaR * (pR - SR / rhoR);

        double cL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) / rhoL);
        double cR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) / rhoR);

        double lam = std::abs(p.PPR_ADV_MULT) * std::max(std::abs(uL) + cL, std::abs(uR) + cR);
        Flux_S_comm = 0.5 * p.PPR_ADV_MULT * (SL * uL + SR * uR) - 0.5 * lam * (SR - SL);
    } else {
        Flux_S_comm = 0.0;
    }
}

void SolverDim<3>::get_flux_pointwise_cell(const Cell3D& c, int iz, int iy, int ix,
                                           double* F, double* G, double* H, double sigma) const
{
    double rho   = std::max(p.POS_LIMITER_EPS, c.get_U(0, iz, iy, ix, p.N_PTS));
    double u     = c.get_U(1, iz, iy, ix, p.N_PTS) / rho;
    double v     = c.get_U(2, iz, iy, ix, p.N_PTS) / rho;
    double w     = c.get_U(3, iz, iy, ix, p.N_PTS) / rho;
    double E     = c.get_U(4, iz, iy, ix, p.N_PTS);
    double press = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v + w*w)));
    if (p.ENABLE_PPR) {
        int idx = iz * p.N_PTS * p.N_PTS + iy * p.N_PTS + ix;
        double P_phan = c.S_field[idx] / rho;
        double theta_cfl = (p.PPR_ADAPTIVE_THETA) ? c.theta_avg : p.PPR_THETA;
        if (press - P_phan < 0.0) {
            double theta_safe = (press - p.POS_LIMITER_EPS) / (P_phan - press);
            theta_cfl = std::min(theta_cfl, std::max(0.0, theta_safe));
        }
        double P_reg  = press + theta_cfl * (press - P_phan);
        press = std::max(p.POS_LIMITER_EPS, P_reg);
    }

    if (F) {
        F[0] = rho * u;
        F[1] = rho * u * u + press + sigma;
        F[2] = rho * u * v;
        F[3] = rho * u * w;
        F[4] = (E + press + sigma) * u;
    }
    if (G) {
        G[0] = rho * v;
        G[1] = rho * v * u;
        G[2] = rho * v * v + press + sigma;
        G[3] = rho * v * w;
        G[4] = (E + press + sigma) * v;
    }
    if (H) {
        H[0] = rho * w;
        H[1] = rho * w * u;
        H[2] = rho * w * v;
        H[3] = rho * w * w + press + sigma;
        H[4] = (E + press + sigma) * w;
    }
}


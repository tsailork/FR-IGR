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
    double inv_rhoL = 1.0 / rhoL;
    double uL = UL[1] * inv_rhoL, vL = UL[2] * inv_rhoL;
    double keL = 0.5 * rhoL * (uL*uL + vL*vL);
    double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UL[3] - keL));

    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    double inv_rhoR = 1.0 / rhoR;
    double uR = UR[1] * inv_rhoR, vR = UR[2] * inv_rhoR;
    double keR = 0.5 * rhoR * (uR*uR + vR*vR);
    double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UR[3] - keR));

    double vnL = (dir == 0) ? uL : vL;
    double vnR = (dir == 0) ? uR : vR;

    double pL_reg = pL, pR_reg = pR;
    if (p.ENABLE_PPR) {
        double pL_phan = SL * inv_rhoL;
        double pR_phan = SR * inv_rhoR;
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

        pL_reg = std::max(p.POS_LIMITER_EPS, pL + theta_cfl_L * (pL - pL_phan));
        pR_reg = std::max(p.POS_LIMITER_EPS, pR + theta_cfl_R * (pR - pR_phan));
    }

    if (p.RIEMANN_SOLVER == "RUSANOV") {
        double cL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) * inv_rhoL);
        double cR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) * inv_rhoR);
        double max_wave = std::max(std::abs(vnL) + cL, std::abs(vnR) + cR);

        double FL[4], FR[4];
        if (dir == 0) {
            FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL_reg;  FL[2] = rhoL*uL*vL;  FL[3] = (UL[3] + pL_reg) * uL;
            FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR_reg;  FR[2] = rhoR*uR*vR;  FR[3] = (UR[3] + pR_reg) * uR;
        } else {
            FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;  FL[2] = rhoL*vL*vL + pL_reg;  FL[3] = (UL[3] + pL_reg) * vL;
            FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;  FR[2] = rhoR*vR*vR + pR_reg;  FR[3] = (UR[3] + pR_reg) * vR;
        }

        for (int v = 0; v < 4; ++v)
            F_comm[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * max_wave * (UR[v] - UL[v]);
    } else {
        // HLLC Riemann Solver (Default)
        double aL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) * inv_rhoL);
        double aR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) * inv_rhoR);

        double sqrt_rhoL = std::sqrt(rhoL);
        double sqrt_rhoR = std::sqrt(rhoR);
        double inv_sqrt_sum = 1.0 / (sqrt_rhoL + sqrt_rhoR);

        double vn_roe = (sqrt_rhoL * vnL + sqrt_rhoR * vnR) * inv_sqrt_sum;
        double HL = (UL[3] + pL_reg) * inv_rhoL;
        double HR = (UR[3] + pR_reg) * inv_rhoR;
        double H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_sqrt_sum;
        double u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_sqrt_sum;
        double v_roe = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_sqrt_sum;
        double ke_roe = 0.5 * (u_roe*u_roe + v_roe*v_roe);
        double a_roe = std::sqrt(std::max(1e-12, (p.GAMMA - 1.0) * (H_roe - ke_roe)));

        double SL_wave = std::min(vnL - aL, vn_roe - a_roe);
        double SR_wave = std::max(vnR + aR, vn_roe + a_roe);

        double num_star = pR_reg - pL_reg + rhoL * vnL * (SL_wave - vnL) - rhoR * vnR * (SR_wave - vnR);
        double den_star = rhoL * (SL_wave - vnL) - rhoR * (SR_wave - vnR);
        double S_star = num_star / den_star;

        double FL[4], FR[4];
        if (dir == 0) {
            FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL_reg;  FL[2] = rhoL*uL*vL;  FL[3] = (UL[3] + pL_reg) * uL;
            FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR_reg;  FR[2] = rhoR*uR*vR;  FR[3] = (UR[3] + pR_reg) * uR;
        } else {
            FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;  FL[2] = rhoL*vL*vL + pL_reg;  FL[3] = (UL[3] + pL_reg) * vL;
            FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;  FR[2] = rhoR*vR*vR + pR_reg;  FR[3] = (UR[3] + pR_reg) * vR;
        }

        if (SL_wave >= 0.0) {
            for (int v = 0; v < 4; ++v) F_comm[v] = FL[v];
        } else if (SL_wave < 0.0 && S_star >= 0.0) {
            double facL = rhoL * (SL_wave - vnL) / (SL_wave - S_star);
            double UL_star[4];
            UL_star[0] = facL;
            if (dir == 0) {
                UL_star[1] = facL * S_star;
                UL_star[2] = facL * vL;
            } else {
                UL_star[1] = facL * uL;
                UL_star[2] = facL * S_star;
            }
            UL_star[3] = facL * (UL[3]*inv_rhoL + (S_star - vnL) * (S_star + pL_reg / (rhoL * (SL_wave - vnL))));
            for (int v = 0; v < 4; ++v)
                F_comm[v] = FL[v] + SL_wave * (UL_star[v] - UL[v]);
        } else if (S_star < 0.0 && SR_wave >= 0.0) {
            double facR = rhoR * (SR_wave - vnR) / (SR_wave - S_star);
            double UR_star[4];
            UR_star[0] = facR;
            if (dir == 0) {
                UR_star[1] = facR * S_star;
                UR_star[2] = facR * vR;
            } else {
                UR_star[1] = facR * uL;
                UR_star[2] = facR * S_star;
            }
            UR_star[3] = facR * (UR[3]*inv_rhoR + (S_star - vnR) * (S_star + pR_reg / (rhoR * (SR_wave - vnR))));
            for (int v = 0; v < 4; ++v)
                F_comm[v] = FR[v] + SR_wave * (UR_star[v] - UR[v]);
        } else {
            for (int v = 0; v < 4; ++v) F_comm[v] = FR[v];
        }
    }
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
        if (p.RIEMANN_SOLVER == "HLLC") {
            double inv_rhoL = 1.0 / rhoL;
            double inv_rhoR = 1.0 / rhoR;
            double uL = UL[1] * inv_rhoL, vL = UL[2] * inv_rhoL;
            double uR = UR[1] * inv_rhoR, vR = UR[2] * inv_rhoR;
            double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UL[3] - 0.5 * rhoL * (uL*uL + vL*vL)));
            double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UR[3] - 0.5 * rhoR * (uR*uR + vR*vR)));

            double pL_reg = pL + thetaL * (pL - SL * inv_rhoL);
            double pR_reg = pR + thetaR * (pR - SR * inv_rhoR);

            double aL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) * inv_rhoL);
            double aR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) * inv_rhoR);

            double sqrt_rhoL = std::sqrt(rhoL);
            double sqrt_rhoR = std::sqrt(rhoR);
            double inv_sqrt_sum = 1.0 / (sqrt_rhoL + sqrt_rhoR);

            double vn_roe = (sqrt_rhoL * vnL + sqrt_rhoR * vnR) * inv_sqrt_sum;
            double HL = (UL[3] + pL_reg) * inv_rhoL;
            double HR = (UR[3] + pR_reg) * inv_rhoR;
            double H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_sqrt_sum;
            double u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_sqrt_sum;
            double v_roe = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_sqrt_sum;
            double ke_roe = 0.5 * (u_roe*u_roe + v_roe*v_roe);
            double a_roe = std::sqrt(std::max(1e-12, (p.GAMMA - 1.0) * (H_roe - ke_roe)));

            double SL_wave = std::min(vnL - aL, vn_roe - a_roe);
            double SR_wave = std::max(vnR + aR, vn_roe + a_roe);

            double num_star = pR_reg - pL_reg + rhoL * vnL * (SL_wave - vnL) - rhoR * vnR * (SR_wave - vnR);
            double den_star = rhoL * (SL_wave - vnL) - rhoR * (SR_wave - vnR);
            double S_star = num_star / den_star;

            double FL_S = SL * vnL;
            double FR_S = SR * vnR;

            if (SL_wave >= 0.0) {
                Flux_S_comm = FL_S;
            } else if (SL_wave < 0.0 && S_star >= 0.0) {
                double SL_star_val = SL * (SL_wave - vnL) / (SL_wave - S_star);
                Flux_S_comm = FL_S + SL_wave * (SL_star_val - SL);
            } else if (S_star < 0.0 && SR_wave >= 0.0) {
                double SR_star_val = SR * (SR_wave - vnR) / (SR_wave - S_star);
                Flux_S_comm = FR_S + SR_wave * (SR_star_val - SR);
            } else {
                Flux_S_comm = FR_S;
            }
            Flux_S_comm *= p.PPR_ADV_MULT;
        } else {
            double uL = UL[1 + dir] / rhoL;
            double uR = UR[1 + dir] / rhoR;

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
        }
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
    double inv_rhoL = 1.0 / rhoL;
    double uL = UL[1] * inv_rhoL, vL = UL[2] * inv_rhoL, wL = UL[3] * inv_rhoL;
    double keL = 0.5 * rhoL * (uL*uL + vL*vL + wL*wL);
    double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UL[4] - keL));

    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    double inv_rhoR = 1.0 / rhoR;
    double uR = UR[1] * inv_rhoR, vR = UR[2] * inv_rhoR, wR = UR[3] * inv_rhoR;
    double keR = 0.5 * rhoR * (uR*uR + vR*vR + wR*wR);
    double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UR[4] - keR));

    double vnL = (dir == 0) ? uL : ((dir == 1) ? vL : wL);
    double vnR = (dir == 0) ? uR : ((dir == 1) ? vR : wR);

    double pL_reg = pL, pR_reg = pR;
    if (p.ENABLE_PPR) {
        double pL_phan = SL * inv_rhoL;
        double pR_phan = SR * inv_rhoR;
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

        pL_reg = std::max(p.POS_LIMITER_EPS, pL + theta_cfl_L * (pL - pL_phan));
        pR_reg = std::max(p.POS_LIMITER_EPS, pR + theta_cfl_R * (pR - pR_phan));
    }

    if (p.RIEMANN_SOLVER == "RUSANOV") {
        double cL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) * inv_rhoL);
        double cR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) * inv_rhoR);
        double max_wave = std::max(std::abs(vnL) + cL, std::abs(vnR) + cR);

        double FL[5], FR[5];
        if (dir == 0) {
            FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL_reg;  FL[2] = rhoL*uL*vL;  FL[3] = rhoL*uL*wL;  FL[4] = (UL[4] + pL_reg) * uL;
            FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR_reg;  FR[2] = rhoR*uR*vR;  FR[3] = rhoR*uR*wR;  FR[4] = (UR[4] + pR_reg) * uR;
        } else if (dir == 1) {
            FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;  FL[2] = rhoL*vL*vL + pL_reg;  FL[3] = rhoL*vL*wL;  FL[4] = (UL[4] + pL_reg) * vL;
            FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;  FR[2] = rhoR*vR*vR + pR_reg;  FR[3] = rhoR*vR*wR;  FR[4] = (UR[4] + pR_reg) * vR;
        } else {
            FL[0] = rhoL*wL;  FL[1] = rhoL*wL*uL;  FL[2] = rhoL*wL*vL;  FL[3] = rhoL*wL*wL + pL_reg;  FL[4] = (UL[4] + pL_reg) * wL;
            FR[0] = rhoR*wR;  FR[1] = rhoR*wR*uR;  FR[2] = rhoR*wR*vR;  FR[3] = rhoR*wR*wR + pR_reg;  FR[4] = (UR[4] + pR_reg) * wR;
        }

        for (int v = 0; v < 5; ++v)
            F_comm[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * max_wave * (UR[v] - UL[v]);
    } else {
        // 3D HLLC Riemann Solver
        double aL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) * inv_rhoL);
        double aR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) * inv_rhoR);

        double sqrt_rhoL = std::sqrt(rhoL);
        double sqrt_rhoR = std::sqrt(rhoR);
        double inv_sqrt_sum = 1.0 / (sqrt_rhoL + sqrt_rhoR);

        double vn_roe = (sqrt_rhoL * vnL + sqrt_rhoR * vnR) * inv_sqrt_sum;
        double HL = (UL[4] + pL_reg) * inv_rhoL;
        double HR = (UR[4] + pR_reg) * inv_rhoR;
        double H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_sqrt_sum;
        double u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_sqrt_sum;
        double v_roe = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_sqrt_sum;
        double w_roe = (sqrt_rhoL * wL + sqrt_rhoR * wR) * inv_sqrt_sum;
        double ke_roe = 0.5 * (u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
        double a_roe = std::sqrt(std::max(1e-12, (p.GAMMA - 1.0) * (H_roe - ke_roe)));

        double SL_wave = std::min(vnL - aL, vn_roe - a_roe);
        double SR_wave = std::max(vnR + aR, vn_roe + a_roe);

        double num_star = pR_reg - pL_reg + rhoL * vnL * (SL_wave - vnL) - rhoR * vnR * (SR_wave - vnR);
        double den_star = rhoL * (SL_wave - vnL) - rhoR * (SR_wave - vnR);
        double S_star = num_star / den_star;

        double FL[5], FR[5];
        if (dir == 0) {
            FL[0] = rhoL*uL;  FL[1] = rhoL*uL*uL + pL_reg;  FL[2] = rhoL*uL*vL;  FL[3] = rhoL*uL*wL;  FL[4] = (UL[4] + pL_reg) * uL;
            FR[0] = rhoR*uR;  FR[1] = rhoR*uR*uR + pR_reg;  FR[2] = rhoR*uR*vR;  FR[3] = rhoR*uR*wR;  FR[4] = (UR[4] + pR_reg) * uR;
        } else if (dir == 1) {
            FL[0] = rhoL*vL;  FL[1] = rhoL*vL*uL;  FL[2] = rhoL*vL*vL + pL_reg;  FL[3] = rhoL*vL*wL;  FL[4] = (UL[4] + pL_reg) * vL;
            FR[0] = rhoR*vR;  FR[1] = rhoR*vR*uR;  FR[2] = rhoR*vR*vR + pR_reg;  FR[3] = rhoR*vR*wR;  FR[4] = (UR[4] + pR_reg) * vR;
        } else {
            FL[0] = rhoL*wL;  FL[1] = rhoL*wL*uL;  FL[2] = rhoL*wL*vL;  FL[3] = rhoL*wL*wL + pL_reg;  FL[4] = (UL[4] + pL_reg) * wL;
            FR[0] = rhoR*wR;  FR[1] = rhoR*wR*uR;  FR[2] = rhoR*wR*vR;  FR[3] = rhoR*wR*wR + pR_reg;  FR[4] = (UR[4] + pR_reg) * wR;
        }

        if (SL_wave >= 0.0) {
            for (int v = 0; v < 5; ++v) F_comm[v] = FL[v];
        } else if (SL_wave < 0.0 && S_star >= 0.0) {
            double facL = rhoL * (SL_wave - vnL) / (SL_wave - S_star);
            double UL_star[5];
            UL_star[0] = facL;
            UL_star[1] = facL * (uL + ((dir == 0) ? (S_star - vnL) : 0.0));
            UL_star[2] = facL * (vL + ((dir == 1) ? (S_star - vnL) : 0.0));
            UL_star[3] = facL * (wL + ((dir == 2) ? (S_star - vnL) : 0.0));
            UL_star[4] = facL * (UL[4]*inv_rhoL + (S_star - vnL) * (S_star + pL_reg / (rhoL * (SL_wave - vnL))));
            for (int v = 0; v < 5; ++v)
                F_comm[v] = FL[v] + SL_wave * (UL_star[v] - UL[v]);
        } else if (S_star < 0.0 && SR_wave >= 0.0) {
            double facR = rhoR * (SR_wave - vnR) / (SR_wave - S_star);
            double UR_star[5];
            UR_star[0] = facR;
            UR_star[1] = facR * (uR + ((dir == 0) ? (S_star - vnR) : 0.0));
            UR_star[2] = facR * (vR + ((dir == 1) ? (S_star - vnR) : 0.0));
            UR_star[3] = facR * (wR + ((dir == 2) ? (S_star - vnR) : 0.0));
            UR_star[4] = facR * (UR[4]*inv_rhoR + (S_star - vnR) * (S_star + pR_reg / (rhoR * (SR_wave - vnR))));
            for (int v = 0; v < 5; ++v)
                F_comm[v] = FR[v] + SR_wave * (UR_star[v] - UR[v]);
        } else {
            for (int v = 0; v < 5; ++v) F_comm[v] = FR[v];
        }
    }
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
        if (p.RIEMANN_SOLVER == "HLLC") {
            double inv_rhoL = 1.0 / rhoL;
            double inv_rhoR = 1.0 / rhoR;
            double uL = UL[1] * inv_rhoL, vL = UL[2] * inv_rhoL, wL = UL[3] * inv_rhoL;
            double uR = UR[1] * inv_rhoR, vR = UR[2] * inv_rhoR, wR = UR[3] * inv_rhoR;
            double pL = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UL[4] - 0.5 * rhoL * (uL*uL + vL*vL + wL*wL)));
            double pR = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (UR[4] - 0.5 * rhoR * (uR*uR + vR*vR + wR*wR)));

            double pL_reg = pL + thetaL * (pL - SL * inv_rhoL);
            double pR_reg = pR + thetaR * (pR - SR * inv_rhoR);

            double aL = std::sqrt(p.GAMMA * std::max(pL, pL_reg) * inv_rhoL);
            double aR = std::sqrt(p.GAMMA * std::max(pR, pR_reg) * inv_rhoR);

            double sqrt_rhoL = std::sqrt(rhoL);
            double sqrt_rhoR = std::sqrt(rhoR);
            double inv_sqrt_sum = 1.0 / (sqrt_rhoL + sqrt_rhoR);

            double vn_roe = (sqrt_rhoL * vnL + sqrt_rhoR * vnR) * inv_sqrt_sum;
            double HL = (UL[4] + pL_reg) * inv_rhoL;
            double HR = (UR[4] + pR_reg) * inv_rhoR;
            double H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_sqrt_sum;
            double u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_sqrt_sum;
            double v_roe = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_sqrt_sum;
            double w_roe = (sqrt_rhoL * wL + sqrt_rhoR * wR) * inv_sqrt_sum;
            double ke_roe = 0.5 * (u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
            double a_roe = std::sqrt(std::max(1e-12, (p.GAMMA - 1.0) * (H_roe - ke_roe)));

            double SL_wave = std::min(vnL - aL, vn_roe - a_roe);
            double SR_wave = std::max(vnR + aR, vn_roe + a_roe);

            double num_star = pR_reg - pL_reg + rhoL * vnL * (SL_wave - vnL) - rhoR * vnR * (SR_wave - vnR);
            double den_star = rhoL * (SL_wave - vnL) - rhoR * (SR_wave - vnR);
            double S_star = num_star / den_star;

            double FL_S = SL * vnL;
            double FR_S = SR * vnR;

            if (SL_wave >= 0.0) {
                Flux_S_comm = FL_S;
            } else if (SL_wave < 0.0 && S_star >= 0.0) {
                double SL_star_val = SL * (SL_wave - vnL) / (SL_wave - S_star);
                Flux_S_comm = FL_S + SL_wave * (SL_star_val - SL);
            } else if (S_star < 0.0 && SR_wave >= 0.0) {
                double SR_star_val = SR * (SR_wave - vnR) / (SR_wave - S_star);
                Flux_S_comm = FR_S + SR_wave * (SR_star_val - SR);
            } else {
                Flux_S_comm = FR_S;
            }
            Flux_S_comm *= p.PPR_ADV_MULT;
        } else {
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
        }
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


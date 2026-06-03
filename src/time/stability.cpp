/**
 * @file stability.cpp
 * @brief Stability error detection and CFL dynamic stability bounds computation.
 *
 * This file contains critical routines to guarantee the stability of the high-order Flux Reconstruction solver.
 * It enforces strict positivity on density and pressure, halting execution if non-physical values arise.
 * Additionally, it dictates the dynamic time-stepping limits based on local wave speeds.
 *
 * @see Solver::check_stability
 * @see Solver::compute_dt
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <iomanip>
#include <cstdlib>

/**
 * @brief Asserts the physical validity of the global thermodynamic state.
 *
 * Scans all degrees of freedom to ensure density (\f$\rho > 0\f$) and pressure (\f$p > 0\f$) remain positive,
 * and that no `NaN` values have been introduced. If a stability violation is found, it terminates the 
 * execution and dumps the failing state vector to `stderr`.
 *
 * @note This is typically invoked after every stage in the SSP-RK3 explicit time-stepping cycle.
 */
void Solver::check_stability() const {
    for (auto& b : blocks) {
        for (int ey = 0; ey < b.ny; ++ey)
          for (int ex = 0; ex < b.nx; ++ex)
            for (int iy = 0; iy < p.N_PTS; ++iy)
              for (int ix = 0; ix < p.N_PTS; ++ix) {
                double rho  = b.U(0, ey, ex, iy, ix);
                double rhou = b.U(1, ey, ex, iy, ix);
                double rhov = b.U(2, ey, ex, iy, ix);
                double E    = b.U(3, ey, ex, iy, ix);
                double press = (p.GAMMA - 1.0) * (E - 0.5*(rhou*rhou + rhov*rhov)/rho);
                if (std::isnan(rho) || std::isnan(press) || rho <= 0.0 || press <= 0.0) {
                    std::cerr << std::scientific << std::setprecision(15)
                              << "\n[STABILITY ERROR] block=" << b.id << " elem=(" << ex << "," << ey
                              << ") node=(" << ix << "," << iy << ")"
                              << "\n  rho  = " << rho
                              << "\n  rhou = " << rhou
                              << "\n  rhov = " << rhov
                              << "\n  E    = " << E
                              << "\n  p    = " << press << "\n";
                    exit(1);
                }
              }
    }
}

/**
 * @brief Determines the maximal stable time-step \f$\Delta t\f$ based on the CFL condition.
 *
 * Iterates over the entire computational domain using OpenMP reductions to identify the 
 * maximum absolute wave speed: \f$ \lambda_{max} = |u| + c \f$. The resulting time-step incorporates
 * the Courant-Friedrichs-Lewy (CFL) limit for convective transport, and is further constrained
 * by diffusive Fourier limits if Isotropic Gradient Regularization (IGR) or viscous fluxes are active.
 *
 * \f[ \Delta t_{conv} = \frac{\text{CFL} \cdot h}{(2P+1) \cdot \lambda_{max}} \f]
 *
 * @return The globally stable explicit time-step \f$\Delta t\f$.
 * @see Parameters::CFL
 */
double Solver::compute_dt() const {
    double min_dt = 1e30;

    for (auto& b : blocks) {
        double max_lambda = 1e-10;
        #pragma omp parallel for collapse(2) reduction(max:max_lambda) schedule(static)
        for (int ey = 0; ey < b.ny; ++ey)
          for (int ex = 0; ex < b.nx; ++ex)
            for (int iy = 0; iy < p.N_PTS; ++iy)
              for (int ix = 0; ix < p.N_PTS; ++ix) {
                double rho = std::max(1e-12, b.U(0, ey, ex, iy, ix));
                double u   = b.U(1, ey, ex, iy, ix) / rho;
                double v   = b.U(2, ey, ex, iy, ix) / rho;
                double press = (p.GAMMA - 1.0) * (b.U(3, ey, ex, iy, ix) - 0.5*rho*(u*u + v*v));
                if (press < 1e-12) press = 1e-12;
                double c = std::sqrt(p.GAMMA * press / rho);
                max_lambda = std::max({max_lambda, std::abs(u) + c, std::abs(v) + c});
              }

        double h = std::min(b.dx, b.dy);
        double dt_conv = 0.5 * p.CFL * h / (max_lambda * (p.P_DEG + 1) * (p.P_DEG + 1));
        double dt_block = dt_conv;

        if (p.ENABLE_IGR && p.IGR_TYPE == "PARABOLIC") {
            double alpha_safe = std::max(1e-10, p.ALPHA_SCALE);
            double dt_diff  = 0.5 * p.IGR_TAU_R / (alpha_safe * (2*p.P_DEG+1) * (2*p.P_DEG+1));
            double dt_relax = 0.5 * p.IGR_TAU_R;
            dt_block = std::min({dt_conv, dt_diff, dt_relax});
        }

        // Viscous CFL: dt_visc ~ h² / (ν · (2P+1)²)
        if (p.ENABLE_NS) {
            double nu = 1.0 / p.RE;  // kinematic viscosity (non-dim, ρ_ref = 1)
            double h2 = std::min(b.dx * b.dx, b.dy * b.dy);
            double denom = (2*p.P_DEG+1) * (2*p.P_DEG+1);
            double dt_visc = 0.25 * p.CFL * h2 / (nu * denom);
            dt_block = std::min(dt_block, dt_visc);
        }

        min_dt = std::min(min_dt, dt_block);
    }
    return min_dt;
}

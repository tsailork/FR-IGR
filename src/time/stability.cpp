/// @file stability.cpp
/// @brief Stability error detection and CFL-based time-step computation.
///
/// check_stability() terminates the program on non-physical states (ρ ≤ 0,
/// p ≤ 0, NaN).  compute_dt() returns the largest stable time-step based on
/// convective and diffusive limits.
///
/// OpenMP: compute_dt uses a max-reduction for the global wave speed.

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <iomanip>
#include <cstdlib>

void Solver::check_stability() const {
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey)
      for (int ex = 0; ex < p.N_ELEM_X; ++ex)
        for (int iy = 0; iy < p.N_PTS; ++iy)
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double rho  = U(0, ey, ex, iy, ix);
            double rhou = U(1, ey, ex, iy, ix);
            double rhov = U(2, ey, ex, iy, ix);
            double E    = U(3, ey, ex, iy, ix);
            double press = (p.GAMMA - 1.0) * (E - 0.5*(rhou*rhou + rhov*rhov)/rho);
            if (std::isnan(rho) || std::isnan(press) || rho <= 0.0 || press <= 0.0) {
                std::cerr << std::scientific << std::setprecision(15)
                          << "\n[STABILITY ERROR] elem=(" << ex << "," << ey
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

double Solver::compute_dt() const {
    double max_lambda = 0.0;

    #pragma omp parallel for collapse(2) reduction(max:max_lambda) schedule(static)
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey)
      for (int ex = 0; ex < p.N_ELEM_X; ++ex)
        for (int iy = 0; iy < p.N_PTS; ++iy)
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double rho = clamp_positivity(U(0, ey, ex, iy, ix));
            double u   = U(1, ey, ex, iy, ix) / rho;
            double v   = U(2, ey, ex, iy, ix) / rho;
            double press = (p.GAMMA - 1.0) * (U(3, ey, ex, iy, ix) - 0.5*rho*(u*u + v*v));
            if (press < 1e-12) press = 1e-12;
            double c = std::sqrt(p.GAMMA * press / rho);
            max_lambda = std::max({max_lambda, std::abs(u) + c, std::abs(v) + c});
          }

    double h = std::min(dx, dy);
    double dt_conv = 0.5 * p.CFL * h / (max_lambda * (p.P_DEG + 1) * (p.P_DEG + 1));

    if (p.ENABLE_IGR && p.IGR_TYPE == "PARABOLIC") {
        double alpha_safe = std::max(1e-10, p.ALPHA_SCALE);
        double dt_diff  = 0.5 * p.IGR_TAU_R / (alpha_safe * (2*p.P_DEG+1) * (2*p.P_DEG+1));
        double dt_relax = 0.5 * p.IGR_TAU_R;
        return std::min({dt_conv, dt_diff, dt_relax});
    }
    return dt_conv;
}

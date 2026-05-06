/// @file entropic_pressure.cpp
/// @brief Top-level dispatch for the entropic pressure computation.
///
/// Routes to either the elliptic ADI solver or the parabolic BR2 evolution.
/// After solving, σ is clamped at each point to ≤ local physical pressure
/// to prevent the dissipative IGR term from draining more internal energy
/// than is available — a key robustness constraint for long-running
/// blast-wave simulations with wall reflections.

#include "../core/solver.hpp"

void Solver::compute_entropic_pressure() {
  if (!p.ENABLE_IGR)
    return;

  compute_sensor_source();

  if (p.IGR_TYPE == "PARABOLIC") {
    compute_igr_parabolic_rhs();
    // For parabolic mode, sigma_field is updated by the RK3 integrator
    // in rk3.cpp. The σ-clamp is applied there after each stage update
    // to avoid corrupting the SSP convex combinations.
  } else {
    // Symmetrised ADI: average XY and YX passes
    solve_adi_pass(S_buf, sigma_xy_buf, true);
    solve_adi_pass(S_buf, sigma_yx_buf, false);

    for (size_t i = 0; i < sigma_field.size(); ++i)
      sigma_field[i] = 0.5 * (sigma_xy_buf[i] + sigma_yx_buf[i]);

    // -----------------------------------------------------------------
    // σ-clamp (elliptic only): bound entropic pressure by local
    // physical pressure.  Clamping σ ≤ p ensures the total effective
    // pressure (p + σ) ≤ 2p, bounding the energy drain rate.
    // -----------------------------------------------------------------
#pragma omp parallel for schedule(static)
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey)
      for (int ex = 0; ex < p.N_ELEM_X; ++ex)
        for (int iy = 0; iy < p.N_PTS; ++iy)
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            int idx = get_flat_idx(ey, ex, iy, ix);
            double rho = std::max(1e-14, U(0, ey, ex, iy, ix));
            double rhou = U(1, ey, ex, iy, ix);
            double rhov = U(2, ey, ex, iy, ix);
            double E = U(3, ey, ex, iy, ix);
            double press = (p.GAMMA - 1.0) * (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
            if (press < 1e-14)
              press = 1e-14;
            sigma_field[idx] = std::min(sigma_field[idx], press);
          }
  }
}

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
  } else {
    for (auto& b : blocks) {
      // Symmetrised ADI: average XY and YX passes
      solve_adi_pass(b, b.S_buf, b.sigma_xy_buf, true);
      solve_adi_pass(b, b.S_buf, b.sigma_yx_buf, false);

      for (size_t i = 0; i < b.sigma_field.size(); ++i)
        b.sigma_field[i] = 0.5 * (b.sigma_xy_buf[i] + b.sigma_yx_buf[i]);

      #pragma omp for schedule(static)
      for (int ey = 0; ey < b.ny; ++ey)
        for (int ex = 0; ex < b.nx; ++ex)
          for (int iy = 0; iy < p.N_PTS; ++iy)
            for (int ix = 0; ix < p.N_PTS; ++ix) {
              int idx = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
              double rho = std::max(1e-14, b.U(0, ey, ex, iy, ix));
              double rhou = b.U(1, ey, ex, iy, ix);
              double rhov = b.U(2, ey, ex, iy, ix);
              double E = b.U(3, ey, ex, iy, ix);
              double press = (p.GAMMA - 1.0) * (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
              if (press < 1e-14)
                press = 1e-14;
              b.sigma_field[idx] = std::min(b.sigma_field[idx], press);
            }
    }
  }
}

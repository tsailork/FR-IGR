/// @file rk3.cpp
/// @brief Strong Stability Preserving Runge-Kutta 3rd Order (SSP-RK3).
///
/// Three stages:
///   u^(1) = u^n + dt · L(u^n)
///   u^(2) = ¾ u^n + ¼ (u^(1) + dt · L(u^(1)))
///   u^(n+1) = ⅓ u^n + ⅔ (u^(2) + dt · L(u^(2)))
///
/// After each stage: positivity limiter, entropy limiter, stability check.
///
/// Parabolic σ field is advanced in lockstep with the flow.  When
/// IGR_SUB_ITERS > 1, the σ relaxation is sub-iterated: after each
/// RK3 stage updates U, the sigma field is evolved through multiple
/// forward-Euler sub-steps (holding U fixed) to accelerate convergence
/// toward the elliptic steady state.
///
/// OpenMP: the BLAS-like vector updates are parallelised.

#include "../core/solver.hpp"
#include "../limiters/entropy.hpp"
#include "../limiters/positivity.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/// Perform one full SSP-RK3 time step.
///
/// @details
/// Data Structures & Indexing:
///   - `U(v, ey, ex, iy, ix)`: The global conserved state array. Mutated
///   in-place during each RK stage.
///   - `RHS(v, ey, ex, iy, ix)`: The global explicit right-hand side array
///   computed by `compute_rhs()`.
///   - `sigma_field`: The global scalar entropic pressure field, advanced
///   alongside `U` if using Parabolic IGR.
///   - `sigma_RHS`: The explicit RHS for the `sigma_field`.
///   - `U_old`, `sig_old`: Thread-local (to this function) copies of the
///   initial state at $t^n$ needed for RK combinations.
/// Assumptions:
///   - `U` and `sigma_field` contain valid states at the start of the timestep.
///   - OpenMP is used for the BLAS-like vector updates: `#pragma omp parallel
///   for`. These are thread-safe
///     because each thread operates on independent indices of the flat 1D
///     arrays `U.data` and `sigma_field`.
void Solver::step_rk3(double dt) {
  std::vector<State> U_old;
  std::vector<std::vector<double>> sig_old;
  for (auto &b : blocks) {
    U_old.push_back(b.U);
    sig_old.push_back(b.sigma_field);
  }

  const bool is_parabolic = (p.IGR_TYPE == "PARABOLIC");
  const int n_sub = std::max(1, p.IGR_SUB_ITERS);
  const double dt_sub = dt / n_sub;

  current_limiter_stats.num_limited = 0;
  current_limiter_stats.sum_theta = 0.0;
  auto add_stats = [&](const Limiters::LimiterStats &s) {
    current_limiter_stats.num_limited += s.num_limited;
    current_limiter_stats.sum_theta += s.sum_theta;
  };

  static constexpr bool USE_PRESSURE_GRAD_CAP = false;
  static constexpr double PGRAD_CAP_C = 2.0;

  auto clamp_sigma_all = [&]() {
    for (auto &b : blocks) {
#pragma omp parallel for schedule(static)
      for (int ey = 0; ey < b.ny; ++ey)
        for (int ex = 0; ex < b.nx; ++ex)
          for (int iy = 0; iy < p.N_PTS; ++iy)
            for (int ix = 0; ix < p.N_PTS; ++ix) {
              int idx = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
              if (b.sigma_field[idx] < 0.0) {
                b.sigma_field[idx] = 0.0;
                continue;
              }
              double rho = std::max(1e-14, b.U(0, ey, ex, iy, ix));
              double rhou = b.U(1, ey, ex, iy, ix);
              double rhov = b.U(2, ey, ex, iy, ix);
              double E = b.U(3, ey, ex, iy, ix);
              double press = (p.GAMMA - 1.0) *
                             (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
              if (press < 1e-14)
                press = 1e-14;

              b.sigma_field[idx] = std::min(b.sigma_field[idx], press);

              if (USE_PRESSURE_GRAD_CAP) {
                double dpdx = 0, dpdy = 0;
                for (int k = 0; k < p.N_PTS; ++k) {
                  double rk = std::max(1e-14, b.U(0, ey, ex, iy, k));
                  double Ek = b.U(3, ey, ex, iy, k);
                  double pk = (p.GAMMA - 1.0) *
                              (Ek - 0.5 *
                                        (std::pow(b.U(1, ey, ex, iy, k), 2) +
                                         std::pow(b.U(2, ey, ex, iy, k), 2)) /
                                        rk);
                  dpdx += basis.D[ix][k] * pk;
                }
                for (int k = 0; k < p.N_PTS; ++k) {
                  double rk = std::max(1e-14, b.U(0, ey, ex, k, ix));
                  double Ek = b.U(3, ey, ex, k, ix);
                  double pk = (p.GAMMA - 1.0) *
                              (Ek - 0.5 *
                                        (std::pow(b.U(1, ey, ex, k, ix), 2) +
                                         std::pow(b.U(2, ey, ex, k, ix), 2)) /
                                        rk);
                  dpdy += basis.D[iy][k] * pk;
                }
                dpdx *= (2.0 / b.dx);
                dpdy *= (2.0 / b.dy);
                double grad_p_mag = std::sqrt(dpdx * dpdx + dpdy * dpdy);
                double h = std::min(b.dx, b.dy);
                double sig_cap = PGRAD_CAP_C * h * grad_p_mag / (p.GAMMA - 1.0);
                b.sigma_field[idx] = std::min(b.sigma_field[idx], sig_cap);
              }
            }
    }
  };

  auto sub_iterate_sigma_all =
      [&](const std::vector<std::vector<double>> &sig_stage_old, double alpha,
          double beta) {
        if (n_sub == 1) {
          for (size_t bid = 0; bid < blocks.size(); ++bid) {
            auto &b = blocks[bid];
            const size_t N_sig = b.sigma_field.size();
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N_sig; ++i)
              b.sigma_field[i] =
                  alpha * sig_stage_old[bid][i] +
                  beta * (b.sigma_field[i] + dt * b.sigma_RHS[i]);
          }
          clamp_sigma_all();
        } else {
          for (size_t bid = 0; bid < blocks.size(); ++bid) {
            auto &b = blocks[bid];
            const size_t N_sig = b.sigma_field.size();
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N_sig; ++i)
              b.sigma_field[i] =
                  alpha * sig_stage_old[bid][i] +
                  beta * (b.sigma_field[i] + dt_sub * b.sigma_RHS[i]);
          }
          clamp_sigma_all();

          for (int sub = 1; sub < n_sub; ++sub) {
            compute_igr_parabolic_rhs();
            for (auto &b : blocks) {
              const size_t N_sig = b.sigma_field.size();
#pragma omp parallel for schedule(static)
              for (size_t i = 0; i < N_sig; ++i)
                b.sigma_field[i] += dt_sub * b.sigma_RHS[i];
            }
            clamp_sigma_all();
          }
        }
      };

  // Stage 1
  compute_rhs();
  for (size_t bid = 0; bid < blocks.size(); ++bid) {
    auto &b = blocks[bid];
    const size_t N_U = b.U.data.size();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_U; ++i)
      b.U.data[i] = U_old[bid].data[i] + dt * b.RHS.data[i];
  }
  if (is_parabolic)
    sub_iterate_sigma_all(sig_old, 0.0, 1.0);

  if (p.ENABLE_POS_LIMITER) {
    for (auto &b : blocks)
      add_stats(Limiters::apply_positivity_limiter(b.U, basis, p));
  }
  if (p.ENABLE_ENTROPY_LIMITER)
    add_stats(Limiters::apply_entropy_limiter(*this));
  check_stability();

  // Stage 2
  compute_rhs();
  for (size_t bid = 0; bid < blocks.size(); ++bid) {
    auto &b = blocks[bid];
    const size_t N_U = b.U.data.size();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_U; ++i)
      b.U.data[i] =
          0.75 * U_old[bid].data[i] + 0.25 * (b.U.data[i] + dt * b.RHS.data[i]);
  }
  if (is_parabolic)
    sub_iterate_sigma_all(sig_old, 0.75, 0.25);

  if (p.ENABLE_POS_LIMITER) {
    for (auto &b : blocks)
      add_stats(Limiters::apply_positivity_limiter(b.U, basis, p));
  }
  if (p.ENABLE_ENTROPY_LIMITER)
    add_stats(Limiters::apply_entropy_limiter(*this));
  check_stability();

  // Stage 3
  compute_rhs();
  for (size_t bid = 0; bid < blocks.size(); ++bid) {
    auto &b = blocks[bid];
    const size_t N_U = b.U.data.size();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_U; ++i)
      b.U.data[i] = (1.0 / 3.0) * U_old[bid].data[i] +
                    (2.0 / 3.0) * (b.U.data[i] + dt * b.RHS.data[i]);
  }
  if (is_parabolic)
    sub_iterate_sigma_all(sig_old, 1.0 / 3.0, 2.0 / 3.0);

  if (p.ENABLE_POS_LIMITER) {
    for (auto &b : blocks)
      add_stats(Limiters::apply_positivity_limiter(b.U, basis, p));
  }
  if (p.ENABLE_ENTROPY_LIMITER)
    add_stats(Limiters::apply_entropy_limiter(*this));
  check_stability();
}

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
///   - `U(v, ey, ex, iy, ix)`: The global conserved state array. Mutated in-place during each RK stage.
///   - `RHS(v, ey, ex, iy, ix)`: The global explicit right-hand side array computed by `compute_rhs()`.
///   - `sigma_field`: The global scalar entropic pressure field, advanced alongside `U` if using Parabolic IGR.
///   - `sigma_RHS`: The explicit RHS for the `sigma_field`.
///   - `U_old`, `sig_old`: Thread-local (to this function) copies of the initial state at $t^n$ needed for RK combinations.
/// Assumptions:
///   - `U` and `sigma_field` contain valid states at the start of the timestep.
///   - OpenMP is used for the BLAS-like vector updates: `#pragma omp parallel for`. These are thread-safe 
///     because each thread operates on independent indices of the flat 1D arrays `U.data` and `sigma_field`.
void Solver::step_rk3(double dt) {
  State U_old = U;
  std::vector<double> sig_old = sigma_field;
  const size_t N_U = U.data.size();
  const size_t N_sig = sigma_field.size();
  const bool is_parabolic = (p.IGR_TYPE == "PARABOLIC");
  const int n_sub = std::max(1, p.IGR_SUB_ITERS);
  const double dt_sub = dt / n_sub;

  current_limiter_stats.num_limited = 0;
  current_limiter_stats.sum_theta = 0.0;
  auto add_stats = [&](const Limiters::LimiterStats& s) {
      current_limiter_stats.num_limited += s.num_limited;
      current_limiter_stats.sum_theta += s.sum_theta;
  };

  // Approach D toggle: cap σ by local pressure gradient magnitude.
  // At contacts, p is continuous → |∇p| ≈ 0 → σ forced to ~0.
  // At shocks, |∇p| is large → cap is loose and non-interfering.
  static constexpr bool USE_PRESSURE_GRAD_CAP = true; // true;
  static constexpr double PGRAD_CAP_C = 2.0;          // Tunable coefficient

  // Helper: clamp σ to [0, local physical pressure] after each stage,
  // with optional pressure-gradient cap.
  auto clamp_sigma = [&]() {
#pragma omp parallel for schedule(static)
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey)
      for (int ex = 0; ex < p.N_ELEM_X; ++ex)
        for (int iy = 0; iy < p.N_PTS; ++iy)
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            int idx = get_flat_idx(ey, ex, iy, ix);
            if (sigma_field[idx] < 0.0) {
              sigma_field[idx] = 0.0;
              continue;
            }
            double rho = std::max(1e-14, U(0, ey, ex, iy, ix));
            double rhou = U(1, ey, ex, iy, ix);
            double rhov = U(2, ey, ex, iy, ix);
            double E = U(3, ey, ex, iy, ix);
            double press =
                (p.GAMMA - 1.0) * (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
            if (press < 1e-14)
              press = 1e-14;

            // Standard clamp: σ ≤ p
            sigma_field[idx] = std::min(sigma_field[idx], press);

            // Approach D: σ ≤ C · h · |∇p| / (γ-1)
            if (USE_PRESSURE_GRAD_CAP) {
              double dpdx = 0, dpdy = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                double rk = std::max(1e-14, U(0, ey, ex, iy, k));
                double ruk = U(1, ey, ex, iy, k);
                double rvk = U(2, ey, ex, iy, k);
                double Ek = U(3, ey, ex, iy, k);
                double pk =
                    (p.GAMMA - 1.0) * (Ek - 0.5 * (ruk * ruk + rvk * rvk) / rk);
                dpdx += basis.D[ix][k] * pk;
              }
              for (int k = 0; k < p.N_PTS; ++k) {
                double rk = std::max(1e-14, U(0, ey, ex, k, ix));
                double ruk = U(1, ey, ex, k, ix);
                double rvk = U(2, ey, ex, k, ix);
                double Ek = U(3, ey, ex, k, ix);
                double pk =
                    (p.GAMMA - 1.0) * (Ek - 0.5 * (ruk * ruk + rvk * rvk) / rk);
                dpdy += basis.D[iy][k] * pk;
              }
              dpdx *= (2.0 / dx);
              dpdy *= (2.0 / dy);
              double grad_p_mag = std::sqrt(dpdx * dpdx + dpdy * dpdy);
              double h = std::min(dx, dy);
              double sig_cap = PGRAD_CAP_C * h * grad_p_mag / (p.GAMMA - 1.0);
              sigma_field[idx] = std::min(sigma_field[idx], sig_cap);
            }
          }
  };

  // -----------------------------------------------------------------
  // Helper: sub-iterate the parabolic σ relaxation.
  //
  // Holds U fixed and runs n_sub forward-Euler steps of size dt_sub
  // on the sigma equation:
  //   σ^(k+1) = σ^(k) + dt_sub · RHS_σ(σ^(k), U)
  //
  // For n_sub = 1 this is equivalent to the original single-step
  // update.  For n_sub > 1, the sensor source S is recomputed once
  // (it depends on U which is frozen), while the parabolic diffusion
  // RHS is recomputed each sub-step (it depends on σ which changes).
  // -----------------------------------------------------------------
  auto sub_iterate_sigma = [&](const std::vector<double> &sig_stage_old,
                               double alpha, double beta) {
    // First sub-step uses the sigma_RHS already computed by compute_rhs().
    // Apply the RK3 convex combination for the first sub-step:
    //   σ = α·sig_old + β·(σ_current + dt_sub · σ_RHS)
    // where sig_old is from the start of the RK3 step and σ_current
    // is the running sigma.
    if (n_sub == 1) {
// Standard RK3 update (no sub-iteration)
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N_sig; ++i)
        sigma_field[i] = alpha * sig_stage_old[i] +
                         beta * (sigma_field[i] + dt * sigma_RHS[i]);
      clamp_sigma();
    } else {
// First sub-step: apply the RK3 convex combination with dt_sub
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N_sig; ++i)
        sigma_field[i] = alpha * sig_stage_old[i] +
                         beta * (sigma_field[i] + dt_sub * sigma_RHS[i]);
      clamp_sigma();

      // Remaining sub-steps: pure forward-Euler on σ, U frozen.
      // Sensor S is frozen (depends on U), only the diffusion operator
      // (which depends on σ) is recomputed.
      for (int sub = 1; sub < n_sub; ++sub) {
        compute_igr_parabolic_rhs(); // recompute σ_RHS from current σ
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N_sig; ++i)
          sigma_field[i] += dt_sub * sigma_RHS[i];
        clamp_sigma();
      }
    }
  };

  // =====================================================================
  // Stage 1:  u^(1) = u^n + dt · L(u^n)
  // =====================================================================
  compute_rhs();

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N_U; ++i)
    U.data[i] = U_old.data[i] + dt * RHS.data[i];

  if (is_parabolic)
    sub_iterate_sigma(sig_old, 0.0, 1.0); // σ = sig_old + dt·RHS → α=0, β=1

  if (p.ENABLE_POS_LIMITER)
    add_stats(Limiters::apply_positivity_limiter(U, basis, p));
  if (p.ENABLE_ENTROPY_LIMITER)
    add_stats(Limiters::apply_entropy_limiter(*this));
  check_stability();

  // =====================================================================
  // Stage 2:  u^(2) = ¾ u^n + ¼ (u^(1) + dt · L(u^(1)))
  // =====================================================================
  compute_rhs();

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N_U; ++i)
    U.data[i] = 0.75 * U_old.data[i] + 0.25 * (U.data[i] + dt * RHS.data[i]);

  if (is_parabolic)
    sub_iterate_sigma(sig_old, 0.75, 0.25);

  if (p.ENABLE_POS_LIMITER)
    add_stats(Limiters::apply_positivity_limiter(U, basis, p));
  if (p.ENABLE_ENTROPY_LIMITER)
    add_stats(Limiters::apply_entropy_limiter(*this));
  check_stability();

  // =====================================================================
  // Stage 3:  u^(n+1) = ⅓ u^n + ⅔ (u^(2) + dt · L(u^(2)))
  // =====================================================================
  compute_rhs();

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N_U; ++i)
    U.data[i] = (1.0 / 3.0) * U_old.data[i] +
                (2.0 / 3.0) * (U.data[i] + dt * RHS.data[i]);

  if (is_parabolic)
    sub_iterate_sigma(sig_old, 1.0 / 3.0, 2.0 / 3.0);

  if (p.ENABLE_POS_LIMITER)
    add_stats(Limiters::apply_positivity_limiter(U, basis, p));
  if (p.ENABLE_ENTROPY_LIMITER)
    add_stats(Limiters::apply_entropy_limiter(*this));
  check_stability();

}

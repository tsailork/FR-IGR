/// @file rk3.cpp
/// @brief Strong Stability Preserving Runge-Kutta 3rd Order (SSP-RK3).
///
/// Three stages:
///   u^(1) = u^n + dt · L(u^n)
///   u^(2) = ¾ u^n + ¼ (u^(1) + dt · L(u^(1)))
///   u^(n+1) = ⅓ u^n + ⅔ (u^(2) + dt · L(u^(2)))
///
/// After each stage: positivity limiter, entropy limiter, stability check.
/// Parabolic σ field is advanced in lockstep, clamped to ≥ 0.
///
/// OpenMP: the BLAS-like vector updates are parallelised.

#include "../core/solver.hpp"
#include "../limiters/positivity.hpp"
#include "../limiters/entropy.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::step_rk3(double dt) {
    State U_old = U;
    std::vector<double> sig_old = sigma_field;
    const size_t N_U   = U.data.size();
    const size_t N_sig = sigma_field.size();
    const bool is_parabolic = (p.IGR_TYPE == "PARABOLIC");

    // =====================================================================
    // Stage 1:  u^(1) = u^n + dt · L(u^n)
    // =====================================================================
    compute_rhs();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_U; ++i)
        U.data[i] = U_old.data[i] + dt * RHS.data[i];

    if (is_parabolic) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N_sig; ++i) {
            sigma_field[i] = sig_old[i] + dt * sigma_RHS[i];
            if (sigma_field[i] < 0) sigma_field[i] = 0.0;
        }
    }

    if (p.ENABLE_POS_LIMITER)     Limiters::apply_positivity_limiter(U, basis, p);
    if (p.ENABLE_ENTROPY_LIMITER) Limiters::apply_entropy_limiter(*this);
    check_stability();

    // =====================================================================
    // Stage 2:  u^(2) = ¾ u^n + ¼ (u^(1) + dt · L(u^(1)))
    // =====================================================================
    compute_rhs();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_U; ++i)
        U.data[i] = 0.75 * U_old.data[i] + 0.25 * (U.data[i] + dt * RHS.data[i]);

    if (is_parabolic) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N_sig; ++i) {
            sigma_field[i] = 0.75 * sig_old[i] + 0.25 * (sigma_field[i] + dt * sigma_RHS[i]);
            if (sigma_field[i] < 0) sigma_field[i] = 0.0;
        }
    }

    if (p.ENABLE_POS_LIMITER)     Limiters::apply_positivity_limiter(U, basis, p);
    if (p.ENABLE_ENTROPY_LIMITER) Limiters::apply_entropy_limiter(*this);
    check_stability();

    // =====================================================================
    // Stage 3:  u^(n+1) = ⅓ u^n + ⅔ (u^(2) + dt · L(u^(2)))
    // =====================================================================
    compute_rhs();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_U; ++i)
        U.data[i] = (1.0/3.0) * U_old.data[i] + (2.0/3.0) * (U.data[i] + dt * RHS.data[i]);

    if (is_parabolic) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N_sig; ++i) {
            sigma_field[i] = (1.0/3.0) * sig_old[i] + (2.0/3.0) * (sigma_field[i] + dt * sigma_RHS[i]);
            if (sigma_field[i] < 0) sigma_field[i] = 0.0;
        }
    }

    if (p.ENABLE_POS_LIMITER)     Limiters::apply_positivity_limiter(U, basis, p);
    if (p.ENABLE_ENTROPY_LIMITER) Limiters::apply_entropy_limiter(*this);
    check_stability();
}

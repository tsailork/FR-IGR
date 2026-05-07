/// @file adi_solver.cpp
/// @brief Thomas algorithm + symmetrised ADI pass for the elliptic IGR solve.
///
/// Each 1-D line through the grid yields an independent tridiagonal system
/// that can be solved in O(N).  The ADI method splits the 2-D Helmholtz
/// equation into two 1-D passes (X then Y, and Y then X), then averages
/// the results for directional symmetry.
///
/// OpenMP: parallelised over the outer loop (each 1-D line is independent).

#include "adi_solver.hpp"
#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// =========================================================================
// Thomas algorithm for tridiagonal systems
// =========================================================================

void solve_tridiagonal(const std::vector<double>& a,
                       const std::vector<double>& b,
                       const std::vector<double>& c,
                       const std::vector<double>& d,
                       std::vector<double>& x)
{
    int n = static_cast<int>(d.size());
    if (n == 0) return;
    if (n == 1) { x[0] = d[0] / b[0]; return; }

    std::vector<double> cp(n), dp(n);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double den = b[i] - a[i] * cp[i - 1];
        if (std::abs(den) < 1e-15) den = 1e-15;
        cp[i] = c[i] / den;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = dp[i] - cp[i] * x[i + 1];
}

// =========================================================================
// ADI pass (one direction-pair: X→Y or Y→X)
// =========================================================================

void Solver::solve_adi_pass(Block& b, const std::vector<double>& S,
                            std::vector<double>& Out, bool x_first)
{
    int total_nodes = b.ny * b.nx * p.N_PTS * p.N_PTS;
    std::vector<double> Sigma_Star(total_nodes, 0.0);

    auto solve_direction = [&](bool horizontal,
                               const std::vector<double>& source,
                               std::vector<double>& output,
                               bool is_first_pass)
    {
        int n_outer = horizontal ? b.ny * p.N_PTS : b.nx * p.N_PTS;
        int n_elems = horizontal ? b.nx : b.ny;
        double h_elem = horizontal ? b.dx : b.dy;
        double eps = p.ALPHA_SCALE * (h_elem * h_elem);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_outer; ++i) {
            int n_1d = n_elems * p.N_PTS;
            std::vector<double> A(n_1d, 0.0), B(n_1d), C(n_1d, 0.0), RHS_1d(n_1d);

            auto get_idx = [&](int k) {
                if (horizontal) {
                    int ey = i / p.N_PTS, iy = i % p.N_PTS;
                    return b.get_flat_idx(ey, k / p.N_PTS, iy, k % p.N_PTS, p.N_PTS);
                } else {
                    int ex = i / p.N_PTS, ix = i % p.N_PTS;
                    return b.get_flat_idx(k / p.N_PTS, ex, k % p.N_PTS, ix, p.N_PTS);
                }
            };

            for (int k = 0; k < n_1d; ++k) {
                int e_idx = k / p.N_PTS, p_idx = k % p.N_PTS;
                double z_curr = basis.z[p_idx];

                double x_prev = (p_idx == 0)
                    ? ((e_idx - 1) + 0.5 * (1 + basis.z[p.N_PTS - 1])) * h_elem
                    : (e_idx + 0.5 * (1 + basis.z[p_idx - 1])) * h_elem;
                double x_curr = (e_idx + 0.5 * (1 + z_curr)) * h_elem;
                double x_next = (p_idx == p.N_PTS - 1)
                    ? ((e_idx + 1) + 0.5 * (1 + basis.z[0])) * h_elem
                    : (e_idx + 0.5 * (1 + basis.z[p_idx + 1])) * h_elem;

                double h_L = x_curr - x_prev, h_R = x_next - x_curr;
                double h_avg = 0.5 * (h_L + h_R);

                int flat = get_idx(k);
                double rho_curr = std::max(1e-12, b.U.data[flat * 4 + 0]);
                double rho_prev = (k > 0) ? std::max(1e-12, b.U.data[get_idx(k - 1) * 4 + 0]) : rho_curr;
                double rho_next = (k < n_1d - 1) ? std::max(1e-12, b.U.data[get_idx(k + 1) * 4 + 0]) : rho_curr;

                double K_L = eps / (0.5 * (rho_prev + rho_curr));
                double K_R = eps / (0.5 * (rho_next + rho_curr));

                A[k] = -K_L / (h_L * h_avg);
                C[k] = -K_R / (h_R * h_avg);
                B[k] = 1.0 / rho_curr + (K_L / h_L + K_R / h_R) / h_avg;

                RHS_1d[k] = is_first_pass ? source[flat] : source[flat] / rho_curr;
            }

            // Neumann boundaries
            {
                double rho0 = std::max(1e-12, b.U.data[get_idx(0) * 4 + 0]);
                double rho1 = std::max(1e-12, b.U.data[get_idx(1) * 4 + 0]);
                double h_R_s = (basis.z[1] - basis.z[0]) * 0.5 * h_elem;
                double K_R = eps / (0.5 * (rho0 + rho1));
                double t = K_R / (0.5 * h_R_s * h_R_s);
                B[0] = 1.0 / rho0 + t;  C[0] = -t;  A[0] = 0.0;

                int last = n_1d - 1;
                double rhoN = std::max(1e-12, b.U.data[get_idx(last) * 4 + 0]);
                double rhoNm1 = std::max(1e-12, b.U.data[get_idx(last - 1) * 4 + 0]);
                double h_L_s = (basis.z[p.N_PTS - 1] - basis.z[p.N_PTS - 2]) * 0.5 * h_elem;
                double K_L = eps / (0.5 * (rhoN + rhoNm1));
                double tL = K_L / (0.5 * h_L_s * h_L_s);
                B[last] = 1.0 / rhoN + tL;  A[last] = -tL;  C[last] = 0.0;
            }

            std::vector<double> x_sol(n_1d);
            solve_tridiagonal(A, B, C, RHS_1d, x_sol);
            for (int k = 0; k < n_1d; ++k)
                output[get_idx(k)] = std::max(0.0, x_sol[k]);
        }
    };

    if (x_first) {
        solve_direction(true,  S,          Sigma_Star, true);
        solve_direction(false, Sigma_Star, Out,        false);
    } else {
        solve_direction(false, S,          Sigma_Star, true);
        solve_direction(true,  Sigma_Star, Out,        false);
    }
}

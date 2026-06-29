/**
 * @file adi_solver.cpp
 * @brief Thomas algorithm and symmetrized ADI passes for the elliptic IGR solver.
 */

#include "adi_solver.hpp"
#include "../core/solver.hpp"
#include <iostream>

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

void Solver::solve_adi_pass(Block& b, const std::vector<double>& S,
                            std::vector<double>& Out, bool x_first)
{
    std::cerr << "Error: Symmetrized ADI solver is deprecated and disabled.\n";
    std::exit(EXIT_FAILURE);
}

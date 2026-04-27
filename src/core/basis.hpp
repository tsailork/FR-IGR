/// @file basis.hpp
/// @brief 1-D Lagrange basis on Gauss-Legendre points for Flux Reconstruction.
///
/// Provides:
///   - Solution-point locations  z[]  and quadrature weights  w[].
///   - Lagrange basis values at faces:  l_L[]  (x = −1)  and  l_R[]  (x = +1).
///   - Derivative matrix  D[i][j] = l'_j(z_i).
///   - Radau correction polynomial derivatives  dgl[], dgr[]  (FR-DG variant).

#pragma once
#include <cmath>
#include <iostream>
#include <vector>
#include "parameters.hpp"

// ============================================================================
// Legendre polynomial helpers (small, keep inline for performance)
// ============================================================================

/// Evaluate the Legendre polynomial P_n(x) via the three-term recurrence.
inline double legendre(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double Lm2 = 1.0, Lm1 = x, L = 0.0;
    for (int k = 2; k <= n; ++k) {
        L = ((2.0 * k - 1.0) * x * Lm1 - (k - 1.0) * Lm2) / k;
        Lm2 = Lm1;
        Lm1 = L;
    }
    return L;
}

/// Evaluate P'_n(x).  Uses:  (x²−1) P'_n = n (x P_n − P_{n−1}).
inline double legendre_prime(int n, double x) {
    if (n == 0) return 0.0;
    if (std::abs(x) >= 1.0 - 1e-12) {
        double sign = (x < 0 && ((n + 1) % 2 != 0)) ? -1.0 : 1.0;
        return sign * 0.5 * n * (n + 1);
    }
    return (n * (x * legendre(n, x) - legendre(n - 1, x))) / (x * x - 1.0);
}

// ============================================================================
// Basis struct — constructor is implemented in basis.cpp
// ============================================================================

struct Basis {
    std::vector<double> z;     ///< Gauss-Legendre solution points on [−1, 1].
    std::vector<double> w;     ///< Quadrature weights.
    std::vector<double> l_L;   ///< Lagrange basis evaluated at x = −1.
    std::vector<double> l_R;   ///< Lagrange basis evaluated at x = +1.
    std::vector<std::vector<double>> D;  ///< Derivative matrix.
    std::vector<double> dgl;   ///< Left Radau correction derivative.
    std::vector<double> dgr;   ///< Right Radau correction derivative.

    /// Build the basis for the given polynomial degree.
    explicit Basis(const Parameters& p);
};

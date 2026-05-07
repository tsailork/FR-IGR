/// @file basis.cpp
/// @brief Construction of the 1-D Lagrange / FR-DG basis.
///
/// Hardcoded Gauss-Legendre quadrature points for P = 0..3.
/// The derivative matrix, boundary evaluations, and Radau correction
/// derivatives are computed analytically from the quadrature nodes.

#include "basis.hpp"

// ---------------------------------------------------------------------------
// Private helper: barycentric Lagrange interpolation
// ---------------------------------------------------------------------------
static double lagrange_poly(int j, double x,
                            const std::vector<double>& z,
                            const std::vector<double>& bary_w) {
    // Exact match at a node?
    for (size_t k = 0; k < z.size(); ++k)
        if (std::abs(x - z[k]) < 1e-12) return (static_cast<int>(k) == j) ? 1.0 : 0.0;
    double term_j   = bary_w[j] / (x - z[j]);
    double sum_all  = 0.0;
    for (size_t k = 0; k < z.size(); ++k) sum_all += bary_w[k] / (x - z[k]);
    return term_j / sum_all;
}

// ---------------------------------------------------------------------------
Basis::Basis(int P_DEG) {
    // 1. Gauss-Legendre nodes and weights (hardcoded for P = 0..3)
    if (P_DEG == 0) {
        z = {0.0};
        w = {2.0};
    } else if (P_DEG == 1) {
        z = {-0.5773502691896257, 0.5773502691896257};
        w = {1.0, 1.0};
    } else if (P_DEG == 2) {
        z = {-0.774596669241483, 0.0, 0.774596669241483};
        w = {5.0/9.0, 8.0/9.0, 5.0/9.0};
    } else if (P_DEG == 3) {
        z = {-0.861136311594053, -0.339981043584856,
              0.339981043584856,  0.861136311594053};
        w = {(18.0 - std::sqrt(30.0)) / 36.0,
             (18.0 + std::sqrt(30.0)) / 36.0,
             (18.0 + std::sqrt(30.0)) / 36.0,
             (18.0 - std::sqrt(30.0)) / 36.0};
    } else {
        std::cerr << "Error: P_DEG " << P_DEG
                  << " not hardcoded in basis.cpp" << std::endl;
        exit(1);
    }

    const int N = static_cast<int>(z.size());
    l_L.resize(N);  l_R.resize(N);
    dgl.resize(N);  dgr.resize(N);
    D.resize(N, std::vector<double>(N));

    // 2. Barycentric weights
    std::vector<double> bary(N);
    for (int j = 0; j < N; ++j) {
        double prod = 1.0;
        for (int k = 0; k < N; ++k)
            if (j != k) prod *= (z[j] - z[k]);
        bary[j] = 1.0 / prod;
    }

    // 3. Basis values at faces, derivative matrix, Radau corrections
    for (int i = 0; i < N; ++i) {
        l_L[i] = lagrange_poly(i, -1.0, z, bary);
        l_R[i] = lagrange_poly(i,  1.0, z, bary);

        for (int j = 0; j < N; ++j) {
            if (i != j)
                D[i][j] = (bary[j] / bary[i]) / (z[i] - z[j]);
            else {
                double s = 0.0;
                for (int k = 0; k < N; ++k)
                    if (k != i) s += 1.0 / (z[i] - z[k]);
                D[i][j] = s;
            }
        }

        double dL_P    = legendre_prime(P_DEG,     z[i]);
        double dL_Pp1  = legendre_prime(P_DEG + 1, z[i]);
        double sign    = (P_DEG % 2 == 0) ? 1.0 : -1.0;
        dgl[i] = (sign * 0.5) * (dL_P - dL_Pp1);
        dgr[i] = 0.5  * (dL_P + dL_Pp1);
    }
}

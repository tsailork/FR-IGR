#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include "parameters.hpp"

// Helper: Legendre Polynomial Recursion L_n(x)
inline double legendre(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    
    double L_prev2 = 1.0;
    double L_prev1 = x;
    double L_curr = 0.0;

    for (int k = 2; k <= n; ++k) {
        L_curr = ((2.0 * k - 1.0) * x * L_prev1 - (k - 1.0) * L_prev2) / k;
        L_prev2 = L_prev1;
        L_prev1 = L_curr;
    }
    return L_curr;
}

// Helper: Legendre Derivative L'_n(x) using recursion
// Relation: (x^2 - 1) L'_n(x) = n * (x * L_n(x) - L_{n-1}(x))
inline double legendre_prime(int n, double x) {
    if (n == 0) return 0.0;
    if (std::abs(x) >= 1.0 - 1e-12) {
        // Handle edges explicitly to avoid division by zero
        // L'_n(1) = n(n+1)/2. L'_n(-1) = (-1)^(n+1) * n(n+1)/2
        double sign = (x < 0 && ((n+1)%2 != 0)) ? -1.0 : 1.0;
        return sign * 0.5 * n * (n+1);
    }
    double Ln = legendre(n, x);
    double Ln_minus_1 = legendre(n - 1, x);
    return (n * (x * Ln - Ln_minus_1)) / (x * x - 1.0);
}

struct Basis {
    std::vector<double> z;       // Solution Points
    std::vector<double> w;       // Integration Weights (if needed)
    std::vector<double> l_L;     // Lagrange basis at x = -1
    std::vector<double> l_R;     // Lagrange basis at x = +1
    
    // Derivative Matrix D_ij = l'_j(z_i)
    std::vector<std::vector<double>> D;

    // Correction polynomial derivatives at solution points
    // g'_LB (Left correction) and g'_RB (Right correction)
    std::vector<double> dgl; 
    std::vector<double> dgr; 

    Basis(const Parameters& p) {
        // 1. Define Gauss-Legendre Points (Reference Element [-1, 1])
        if (p.P_DEG == 0) {
            z = {0.0};
        } else if (p.P_DEG == 1) {
            // 2 Points: +/- 1/sqrt(3)
            z = {-0.5773502691896257, 0.5773502691896257};
        } else if (p.P_DEG == 2) { 
            // 3 Points
            z = {-0.774596669241483, 0.0, 0.774596669241483};
        } else if (p.P_DEG == 3) { 
            // 4 Points
            z = {-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053};
        } else {
            // Fallback: This snippet focuses on hardcoded GL points for performance
            std::cerr << "Error: P_DEG " << p.P_DEG << " not hardcoded in basis.hpp" << std::endl;
            exit(1);
        }

        int N = (int)z.size(); // N_PTS
        l_L.resize(N);
        l_R.resize(N);
        dgl.resize(N);
        dgr.resize(N);
        D.resize(N, std::vector<double>(N));

        // 2. Compute Barycentric Weights for stable Lagrange
        std::vector<double> weights(N);
        for (int j = 0; j < N; ++j) {
            double prod = 1.0;
            for (int k = 0; k < N; ++k) {
                if (j != k) prod *= (z[j] - z[k]);
            }
            weights[j] = 1.0 / prod;
        }

        // 3. Compute Lagrange Basis Values and Derivative Matrix
        for (int i = 0; i < N; ++i) {
            // Lagrange at boundaries
            l_L[i] = lagrange_poly(i, -1.0, weights);
            l_R[i] = lagrange_poly(i, 1.0, weights);

            // Derivative Matrix D_ij
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    D[i][j] = (weights[j] / weights[i]) / (z[i] - z[j]);
                } else {
                    double sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        if (k != i) sum += 1.0 / (z[i] - z[k]);
                    }
                    D[i][j] = sum;
                }
            }

            // 4. Compute Radau Correction Derivatives (FR-DG)
            // g_LB = (-1)^P / 2 * (L_P - L_{P+1})
            // g_RB = 1/2 * (L_P + L_{P+1})
            
            // Derivatives:
            double dL_P = legendre_prime(p.P_DEG, z[i]);
            double dL_Pplus1 = legendre_prime(p.P_DEG + 1, z[i]);

            double sign = (p.P_DEG % 2 == 0) ? 1.0 : -1.0;
            
            dgl[i] = (sign * 0.5) * (dL_P - dL_Pplus1);
            dgr[i] = 0.5 * (dL_P + dL_Pplus1);
        }
    }

    // Lagrange Basis Polynomial L_j(x)
    double lagrange_poly(int j, double x, const std::vector<double>& w) {
        // Handle case where x matches a node exactly
        for(size_t k=0; k<z.size(); ++k) {
            if (std::abs(x - z[k]) < 1e-12) return ((int)k == j) ? 1.0 : 0.0;
        }
        
        // Barycentric Formula
        double term_j = w[j] / (x - z[j]);
        double sum_terms = 0.0;
        
        for(size_t k=0; k<z.size(); ++k) {
            sum_terms += w[k] / (x - z[k]);
        }
        
        return term_j / sum_terms;
    }
};

/**
 * @file basis.hpp
 * @brief 1-D Lagrange basis on Gauss-Legendre points for Flux Reconstruction.
 *
 * Provides the foundational mathematical structures for the high-order Flux Reconstruction
 * (FR) method, specifically the FR-DG (Discontinuous Galerkin) variant.
 * 
 * Includes:
 *   - Solution-point locations \f$ z_i \f$ and quadrature weights \f$ w_i \f$.
 *   - Lagrange basis values at cell interfaces: \f$ l_L \f$ at \f$ x = -1 \f$ and \f$ l_R \f$ at \f$ x = +1 \f$.
 *   - Derivative matrix \f$ D_{ij} = l'_j(z_i) \f$.
 *   - Radau correction polynomial derivatives \f$ g'_L, g'_R \f$ evaluated at solution points.
 * 
 * @see Solver
 * @see Parameters
 */

#pragma once
#include <cmath>
#include <iostream>
#include <vector>
#include "parameters.hpp"

// ============================================================================
// Legendre polynomial helpers (small, keep inline for performance)
// ============================================================================

/**
 * @brief Evaluate the Legendre polynomial \f$ P_n(x) \f$ via the three-term recurrence.
 *
 * The recurrence relation used is:
 * \f[ (k) P_k(x) = (2k - 1) x P_{k-1}(x) - (k - 1) P_{k-2}(x) \f]
 *
 * @param n Degree of the Legendre polynomial.
 * @param x Evaluation point \f$ x \in [-1, 1] \f$.
 * @return Value of \f$ P_n(x) \f$.
 * @note Kept inline for performance during basis generation.
 */
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

/**
 * @brief Evaluate the derivative of the Legendre polynomial \f$ P'_n(x) \f$.
 *
 * Utilizes the analytical identity:
 * \f[ (x^2 - 1) P'_n(x) = n (x P_n(x) - P_{n-1}(x)) \f]
 * At the boundaries \f$ x = \pm 1 \f$, L'Hopital's rule or known identities are applied.
 *
 * @param n Degree of the Legendre polynomial.
 * @param x Evaluation point \f$ x \in [-1, 1] \f$.
 * @return Value of \f$ P'_n(x) \f$.
 * @see legendre
 */
inline double legendre_prime(int n, double x) {
    if (n == 0) return 0.0;
    if (std::abs(x) >= 1.0 - 1e-12) {
        double sign = (x < 0 && ((n + 1) % 2 != 0)) ? -1.0 : 1.0;
        return sign * 0.5 * n * (n + 1);
    }
    return (n * (x * legendre(n, x) - legendre(n - 1, x))) / (x * x - 1.0);
}

// ============================================================================
// Flat Matrix Helper (Contiguous 64-byte aligned 2D layout)
// ============================================================================

/**
 * @struct FlatMatrix
 * @brief Contiguous 2D matrix structure for SIMD vectorization.
 */
struct FlatMatrix {
    std::vector<double> data;
    size_t rows{0};
    size_t cols{0};

    struct RowProxy {
        const double* row_ptr;
        size_t cols{0};
        [[nodiscard]] inline double operator[](size_t c) const noexcept { return row_ptr[c]; }
        [[nodiscard]] inline size_t size() const noexcept { return cols; }
    };

    struct MutRowProxy {
        double* row_ptr;
        size_t cols{0};
        [[nodiscard]] inline double& operator[](size_t c) noexcept { return row_ptr[c]; }
        [[nodiscard]] inline size_t size() const noexcept { return cols; }
    };

    FlatMatrix() = default;
    FlatMatrix(size_t r, size_t c, double val = 0.0) : data(r * c, val), rows(r), cols(c) {}

    void resize(size_t r, size_t c, double val = 0.0) {
        rows = r;
        cols = c;
        data.assign(r * c, val);
    }

    [[nodiscard]] inline size_t size() const noexcept { return rows; }

    [[nodiscard]] inline double operator()(size_t r, size_t c) const noexcept {
        return data[r * cols + c];
    }

    [[nodiscard]] inline double& operator()(size_t r, size_t c) noexcept {
        return data[r * cols + c];
    }

    [[nodiscard]] inline RowProxy operator[](size_t r) const noexcept {
        return RowProxy{data.data() + r * cols, cols};
    }

    [[nodiscard]] inline MutRowProxy operator[](size_t r) noexcept {
        return MutRowProxy{data.data() + r * cols, cols};
    }

    [[nodiscard]] inline const double* data_ptr() const noexcept { return data.data(); }
    [[nodiscard]] inline double* data_ptr() noexcept { return data.data(); }
};

// ============================================================================
// Basis struct — constructor is implemented in basis.cpp
// ============================================================================

/**
 * @struct Basis
 * @brief Encapsulates the 1D Lagrange polynomial basis and FR-DG correction operators.
 *
 * Central to the tensor-product formulation of the 2D solver. It provides 
 * all operators required to compute gradients and flux divergences.
 * 
 * @see Solver::sweep_x
 * @see Solver::sweep_y
 */
struct Basis {
    int P{0};                  ///< Polynomial degree P.
    std::vector<double> z;     ///< Gauss-Legendre solution points on \f$ [-1, 1] \f$.
    std::vector<double> w;     ///< Quadrature weights for numerical integration.
    std::vector<double> l_L;   ///< Lagrange basis evaluated at the left interface (\f$ x = -1 \f$).
    std::vector<double> l_R;   ///< Lagrange basis evaluated at the right interface (\f$ x = +1 \f$).
    FlatMatrix D;              ///< Nodal derivative matrix \f$ D_{ij} \f$.
    std::vector<double> dgl;   ///< Left Radau correction derivative \f$ g'_L \f$.
    std::vector<double> dgr;   ///< Right Radau correction derivative \f$ g'_R \f$.

    // Quadtree prolongation and restriction matrices
    std::vector<double> bary_w;                 ///< Barycentric weights for Lagrange interpolation.
    FlatMatrix P1;       ///< Prolongation matrix for child cell 1 (coarse face to bottom/left fine face).
    FlatMatrix P2;       ///< Prolongation matrix for child cell 2 (coarse face to top/right fine face).
    FlatMatrix R1;       ///< Conservative L2 restriction matrix for child cell 1.
    FlatMatrix R2;       ///< Conservative L2 restriction matrix for child cell 2.

    /**
     * @brief Nodal Lagrange polynomial evaluation using barycentric weights.
     */
    [[nodiscard]] double interpolate_lagrange(int j, double x) const;

    /**
     * @brief Construct the basis operators for a given polynomial degree.
     * 
     * Computes nodes, weights, barycentric weights, and analytical derivatives.
     * @param P_DEG Polynomial degree \f$ P \f$ defining \f$ P+1 \f$ solution points.
     */
    explicit Basis(int P_DEG);
};

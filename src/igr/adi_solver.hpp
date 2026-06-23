/**
 * @file adi_solver.hpp
 * @brief Declarations for the Thomas algorithm tridiagonal linear solver.
 */

#pragma once
#include <vector>

/**
 * @brief Solve a tridiagonal system of linear equations using the Thomas algorithm (TDMA).
 *
 * Solves a tridiagonal system of the form:
 * \f[ a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i \f]
 *
 * @param[in] a Lower diagonal coefficients (size N, index 0 is ignored).
 * @param[in] b Main diagonal coefficients (size N).
 * @param[in] c Upper diagonal coefficients (size N, index N-1 is ignored).
 * @param[in] d Right-hand side values (size N).
 * @param[out] x Output solution vector (size N).
 */
void solve_tridiagonal(const std::vector<double>& a,
                       const std::vector<double>& b,
                       const std::vector<double>& c,
                       const std::vector<double>& d,
                       std::vector<double>& x);

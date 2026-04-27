/// @file adi_solver.hpp
/// @brief Thomas algorithm and ADI pass declarations.

#pragma once
#include <vector>

/// Solve a tridiagonal system  A x_{i-1} + B x_i + C x_{i+1} = D_i.
void solve_tridiagonal(const std::vector<double>& a,
                       const std::vector<double>& b,
                       const std::vector<double>& c,
                       const std::vector<double>& d,
                       std::vector<double>& x);

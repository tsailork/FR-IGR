/**
 * @file limiter_smooth.hpp
 * @brief C1-Smooth Limiting Pipeline for Implicit JFNK DG/FR Solvers.
 *
 * Implements smooth softplus, softmin, C1-smooth LMEP modal filter,
 * and Zhang-Shu positivity limiter evaluated on interior nodes AND interface points.
 */

#pragma once

#include "../core/basis.hpp"
#include "../core/cell.hpp"
#include "../core/parameters.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace Limiters {

/**
 * @brief C1-smooth softplus function: softplus_eta(x) = (1/eta) * ln(1 + e^(-eta * |x|)) + max(x, 0)
 */
inline double softplus(double x, double eta = 20.0) {
    return (std::log1p(std::exp(-std::abs(eta * x))) / eta) + (x > 0.0 ? x : 0.0);
}

/**
 * @brief C1-smooth softmin operator across a set of values with quadrature/stencil weights.
 */
inline double softmin(const std::vector<double>& vals, const std::vector<double>& weights, double beta = 15.0) {
    if (vals.empty()) return 0.0;
    double min_val = vals[0];
    for (double v : vals) min_val = std::min(min_val, v);

    double sum_w = 0.0;
    for (double w : weights) sum_w += w;
    if (sum_w <= 0.0) sum_w = 1.0;

    double weighted_exp_sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        double w_norm = weights[i] / sum_w;
        weighted_exp_sum += w_norm * std::exp(-beta * (vals[i] - min_val));
    }
    return min_val - (1.0 / beta) * std::log(std::max(1e-15, weighted_exp_sum));
}

/**
 * @brief C1-Smooth LMEP Modal Filter for 2D leaf cells.
 */
void apply_smooth_lmep_filter_2d(CellDim<2>& cell, const Basis& basis, double gamma = 1.4, double ks = 30.0, double eta = 20.0, double beta = 15.0, int s_power = 2);

/**
 * @brief Zhang-Shu Positivity-Preserving Limiter for 2D leaf cells (checking interior nodes AND face points).
 */
void apply_zhang_shu_limiter_2d(CellDim<2>& cell, const Basis& basis, double gamma = 1.4, double eps = 1e-12);

/**
 * @brief Combined full limiter pipeline for 2D cell.
 */
void apply_full_limiter_2d(CellDim<2>& cell, const Basis& basis, double gamma = 1.4, double eps = 1e-12);

} // namespace Limiters

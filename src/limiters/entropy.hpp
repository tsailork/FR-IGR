/**
 * @file entropy.hpp
 * @brief Specific thermodynamic entropy minimum preservation limiter.
 *
 * Prevents non-physical expansion shocks (which violate the second law of thermodynamics)
 * by scaling the polynomial degrees of freedom towards the cell average if the local 
 * entropy drops below an admissible threshold (the minimum over the cell and its neighbors).
 */

#pragma once

/**
 * @class Solver
 * @brief Forward declaration of the Solver class.
 */
class Solver;

#include "limiter_common.hpp"

namespace Limiters {

/**
 * @brief Apply the entropy minimum preservation limiter to the active solver grid.
 *
 * Scans each block element, identifies the local specific entropy floor, and scales 
 * polynomial coefficients toward the cell average where necessary using bisection.
 *
 * @param[in,out] solver The active solver instance whose state fields will be limited.
 * @return LimiterStats containing the count of modified elements and average scaling parameter.
 */
LimiterStats apply_entropy_limiter(Solver& solver);

} // namespace Limiters

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
template<int Dim> class SolverDim;
using Solver = SolverDim<2>;

#include "limiter_common.hpp"

namespace Limiters {

/**
 * @brief Apply the entropy minimum preservation limiter to the active solver grid.
 *
 * Scans each block element, identifies the local specific entropy floor s_floor, and applies
 * the selected LIMITER_STRATEGY (ZHANG_SHU, BBCH, MODAL, HERMITE) to enforce specific entropy
 * s = p / rho^gamma >= s_floor across solution nodes and face checking points.
 *
 * @param[in,out] solver The active solver instance whose state fields will be limited.
 * @return LimiterStats containing the count of modified elements and average scaling parameter.
 */
LimiterStats apply_entropy_limiter(Solver &solver);
LimiterStats apply_entropy_limiter(SolverDim<3> &solver);

} // namespace Limiters

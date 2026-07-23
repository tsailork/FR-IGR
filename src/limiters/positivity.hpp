/**
 * @file positivity.hpp
 * @brief Zhang-Shu bounds-preserving positivity limiter on decoupled Cells.
 */

#pragma once
#include <vector>
#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"
#include "limiter_common.hpp"

namespace Limiters {

/**
 * @brief Enforce positive density and pressure fields at all physical degrees of freedom.
 *
 * Checks density and pressure values at both internal solution points and face-extrapolated 
 * interface points. If values drop below the defined floor, scales the element's local 
 * polynomial coefficients around the cell average to recover positivity while preserving mass.
 *
 * @param[in,out] cells Flat vector of computational cells.
 * @param[in] basis The high-order polynomial basis definition.
 * @param[in] p Global simulation parameters defining positivity floors.
 * @return LimiterStats containing the count of modified elements and average scaling parameter.
 */
LimiterStats apply_positivity_limiter(std::vector<Cell*>& cells, const Basis& basis, const Parameters& p);
LimiterStats apply_positivity_limiter(std::vector<Cell3D*>& cells, const Basis& basis, const Parameters& p);

} // namespace Limiters

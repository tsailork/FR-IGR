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
 * interface points (including PPR phantom and regularized pressure when ENABLE_PPR is true).
 * Supports 4 modular limiter strategies selected via LIMITER_STRATEGY:
 *  - ZHANG_SHU : Zhang & Shu (2010/2011) uniform scaling to cell average.
 *  - BBCH      : Dumont & Loubère (2013), Dzanic & Witherden (2022) Bernstein-Bézier Convex Hull limiter.
 *  - MODAL     : Persson & Peraire (2006), Dzanic & Witherden (2022) Scale-aware Legendre modal damping limiter.
 *  - HERMITE   : Ciallella & Torlo (2023), Vilar (2019) Subgrid Hermite patch limiter.
 *
 * @param[in,out] cells Flat vector of computational cells.
 * @param[in] basis The high-order polynomial basis definition.
 * @param[in] p Global simulation parameters defining positivity floors.
 * @return LimiterStats containing the count of modified elements and average scaling parameter.
 */
LimiterStats apply_positivity_limiter(std::vector<Cell*>& cells, const Basis& basis, const Parameters& p);
LimiterStats apply_positivity_limiter(std::vector<Cell3D*>& cells, const Basis& basis, const Parameters& p);

} // namespace Limiters

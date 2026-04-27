/// @file positivity.hpp
/// @brief Zhang-Shu positivity-preserving limiter declaration.

#pragma once
#include "../core/state.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"

namespace Limiters {
/// Enforce ρ ≥ ε and p ≥ ε at every solution point by scaling toward
/// the cell average.  Each element is independent.
void apply_positivity_limiter(State& U, const Basis& basis, const Parameters& p);
}

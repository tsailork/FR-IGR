/// @file initial_conditions.hpp
/// @brief Initial condition setup declarations.

#pragma once
#include "../core/parameters.hpp"
#include "../core/state.hpp"
#include "../core/basis.hpp"

namespace IC {
/// Apply the chosen initial condition to the state U.
void apply(State& U, const Parameters& p, const Basis& basis, double dx, double dy);

/// Sigmoid function for smoothing transitions.
double sigmoid(double x, double x0, double delta);
}

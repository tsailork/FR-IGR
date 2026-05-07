/// @file initial_conditions.hpp
/// @brief Initial condition setup declarations.

#pragma once
#include "../core/parameters.hpp"
#include "../core/state.hpp"
#include "../core/basis.hpp"

class Solver;

namespace IC {
/// Apply the chosen initial condition to the state U across all blocks.
void apply(Solver& solver);

/// Sigmoid function for smoothing transitions.
double sigmoid(double x, double x0, double delta);
}

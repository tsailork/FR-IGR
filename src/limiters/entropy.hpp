/// @file entropy.hpp
/// @brief Entropy minimum-preservation limiter declaration.

#pragma once

// Forward declaration — the implementation needs the full Solver definition.
class Solver;

#include "limiter_common.hpp"

namespace Limiters {
/// Prevent spurious entropy undershoots by scaling toward the cell average.
/// The entropy floor is the minimum over the cell and its face neighbours.
LimiterStats apply_entropy_limiter(Solver& solver);
}

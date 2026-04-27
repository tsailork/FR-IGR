/// @file restart.hpp
/// @brief Load conserved variables from a previous VTS output file.

#pragma once
#include "../core/parameters.hpp"
#include "../core/state.hpp"
#include <string>

namespace Restart {
/// Load a VTS snapshot into the solver state.
/// Returns true on success, false on failure.
bool load_restart(const std::string& filename, State& U, const Parameters& p);
}

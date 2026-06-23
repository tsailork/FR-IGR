/**
 * @file restart.hpp
 * @brief Serialization utilities for checkpoint-based simulation restarts.
 *
 * Provides interfaces to load previously saved multi-block solution configurations from VTK 
 * StructuredGrid files, restoring exact double-precision fields for density, momentum, and energy.
 */

#pragma once
#include "../core/parameters.hpp"
#include "../core/state.hpp"
#include <string>

/**
 * @struct Block
 * @brief Forward declaration of the computational grid block struct.
 */
struct Block;

/**
 * @namespace Restart
 * @brief Contains routines for loading saved checkpoint states to resume simulations.
 */
namespace Restart {

/**
 * @brief Load a pre-existing multi-block VTS restart dataset into the active solver blocks.
 *
 * Parses the specified XML-based checkpoint file, matches block configurations, 
 * and populates the conserved state fields. Also restores corresponding parameter records.
 *
 * @param[in] filename Path to the restart snapshot file (e.g. "pv_outputs/solution_restart_0.vtm").
 * @param[in,out] blocks Vector of computational blocks to populate.
 * @param[in] p Global simulation parameters reference database.
 * @return True if restart completes successfully, false if an error is encountered.
 */
bool load_restart(const std::string& filename, std::vector<Block>& blocks, const Parameters& p);

} // namespace Restart

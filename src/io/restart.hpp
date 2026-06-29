/**
 * @file restart.hpp
 * @brief Serialization utilities for checkpoint-based simulation restarts on decoupled cells.
 */

#pragma once
#include <string>

class Solver;

namespace Restart {

/**
 * @brief Load a pre-existing single-file VTU restart dataset and map values to cells by Morton ID.
 *
 * @param[in] filename Path to the restart snapshot file.
 * @param[in,out] solver The active Solver context.
 * @return True if restart completes successfully, false if an error is encountered.
 */
bool load_restart(const std::string& filename, Solver& solver);

} // namespace Restart

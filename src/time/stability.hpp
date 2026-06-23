/**
 * @file stability.hpp
 * @brief Redirection hook and placeholder for the CFL condition and stability checking subsystems.
 *
 * @note The dynamic timestep calculations and state stability verifications are implemented 
 * as member functions of the main `Solver` class:
 * - `Solver::compute_dt()`
 * - `Solver::check_stability()`
 *
 * The corresponding declarations are located in `solver.hpp`, and implementations 
 * are located in `stability.cpp`. This file acts as a placeholder to preserve 
 * compilation unit boundaries.
 *
 * @see solver.hpp
 * @see stability.cpp
 */

#pragma once

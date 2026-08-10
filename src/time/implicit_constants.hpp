/**
 * @file implicit_constants.hpp
 * @brief Centralized named compile-time constants for the FR-IGR implicit solver subsystem.
 */

#pragma once

namespace fr::implicit {

// Finite difference perturbation scaling
constexpr double DEFAULT_EPS_FACTOR = 1.0e-7;

// Realizability and thermodynamic positivity floors
constexpr double REALIZABILITY_EPS = 1.0e-12;

// Matrix inversion singularity tolerance
constexpr double SINGULARITY_TOL = 1.0e-14;

// Line search stopping bounds
constexpr double MIN_LINE_SEARCH_ALPHA = 1.0e-6;

// Krylov subspace dimensions and GMRES defaults
constexpr int DEFAULT_GMRES_RESTART = 25;
constexpr int DEFAULT_GMRES_MAX_ITERS = 25;
constexpr double DEFAULT_GMRES_TOL = 1.0e-2;
constexpr double DEFAULT_NEWTON_TOL = 1.0e-4;

} // namespace fr::implicit

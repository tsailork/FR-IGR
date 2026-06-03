/**
 * @file initial_conditions.hpp
 * @brief Initial condition setup declarations for the Euler solver.
 *
 * Defines the namespace and routines responsible for populating the initial 
 * thermodynamic state (\f$\rho, \rho u, \rho v, E\f$) of the domain prior to SSP-RK3 integration.
 */

#pragma once
#include "../core/parameters.hpp"
#include "../core/state.hpp"
#include "../core/basis.hpp"

class Solver;

/**
 * @namespace IC
 * @brief Contains initial condition routines and smoothing utilities.
 */
namespace IC {

/**
 * @brief Applies the chosen initial condition to the state \f$U\f$ across all blocks.
 *
 * Evaluates the initialization configuration (e.g., Riemann 2D configuration, Blast wave, 
 * Isentropic Vortex) and assigns the exact physical values to all Gauss-Legendre quadrature points.
 *
 * @param[in,out] solver The main solver instance whose `blocks` will be initialized.
 * @see Parameters::IC_TYPE
 * @note Overwrites any existing state unless an immersed boundary (IB) mask prevents it.
 */
void apply(Solver& solver);

/**
 * @brief Evaluates a sigmoid function for initial condition smoothing.
 *
 * Smooths discontinuous initial states (like shock waves) over a characteristic width \f$\delta\f$ 
 * to prevent immediate Gibbs-like oscillations in high-order polynomial basis representations.
 *
 * @param[in] x The spatial coordinate to evaluate.
 * @param[in] x0 The location of the step/discontinuity.
 * @param[in] delta The width of the smoothing region.
 * @return The smoothed scalar multiplier bounded within \f$[0, 1]\f$.
 */
double sigmoid(double x, double x0, double delta);
}

/**
 * @file boundary.hpp
 * @brief Declarations for modular boundary condition utilities and ghost state generators.
 *
 * This file contains definitions for constructing ghost states at the domain boundaries.
 * Ghost states are utilized to impose various physical boundary conditions such as 
 * viscous no-slip walls, characteristic boundaries, and prescribed pressure boundaries.
 * These are fundamental for solving the 2D Euler equations with the Flux Reconstruction (FR) method.
 *
 * @see Solver::apply_boundaries
 */

#pragma once

/**
 * @brief Construct a viscous no-slip or moving wall ghost state.
 *
 * Implements viscous wall boundary conditions by mirroring velocity components
 * and applying temperature/pressure matching rules for adiabatic or isothermal conditions:
 * - Velocity: \f$ u_{ghost} = 2 \cdot u_{wall} - u_{interior} \f$
 * - Adiabatic: \f$ T_{ghost} = T_{interior} \f$
 * - Isothermal: \f$ T_{ghost} = 2 \cdot T_{wall} - T_{interior} \f$
 *
 * This is primarily utilized in conjunction with the Volume Penalty Method (VPM) or 
 * direct face flux calculations for boundary faces.
 *
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] u_wall Prescribed wall boundary velocity component in the x-direction (m/s)
 * @param[in] v_wall Prescribed wall boundary velocity component in the y-direction (m/s)
 * @param[in] gamma Specific heat ratio \f$ \gamma \f$
 * @param[in] isothermal True to impose an isothermal wall boundary, false for adiabatic
 * @param[in] T_wall Prescribed target wall temperature (used only if isothermal is true)
 *
 * @note The energy equation is updated to reflect the new ghost momentum and temperature/pressure.
 */
void build_viscous_wall_ghost(const double* face_state, double* neigh_state,
                              double u_wall, double v_wall, double gamma,
                              bool isothermal, double T_wall);

/**
 * @brief Construct a characteristic ghost state using 1D Riemann invariants.
 *
 * Calculates a ghost boundary state normal to the computational interface using
 * one-dimensional characteristic variables and upwinded wave propagation:
 * - Riemann invariant \f$ R_+ = u_n + \frac{2c}{\gamma - 1} \f$
 * - Riemann invariant \f$ R_- = u_n - \frac{2c}{\gamma - 1} \f$
 * Upwinds invariants, entropy, and tangential velocity according to local wave speeds.
 * This is essential for non-reflecting far-field boundary conditions.
 *
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] ref_state Exterior physical reference state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] n_x Outer unit normal coordinate direction component x
 * @param[in] n_y Outer unit normal coordinate direction component y
 * @param[in] gamma Specific heat ratio \f$ \gamma \f$
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 *
 * @see euler_flux
 */
void build_characteristic_ghost(const double* face_state, const double* ref_state,
                                double n_x, double n_y, double gamma,
                                double* neigh_state);

/**
 * @brief Construct a compressible total pressure ghost state.
 *
 * Imposes a specific total pressure at the boundary using isentropic relations
 * applicable to compressible flows.
 *
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] Pt_target Target total pressure to impose \f$ P_t \f$
 * @param[in] gamma Specific heat ratio \f$ \gamma \f$
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 */
void build_total_pressure_comp_ghost(const double* face_state, double Pt_target,
                                     double gamma, double* neigh_state);

/**
 * @brief Construct an incompressible/Bernoulli total pressure ghost state.
 *
 * Utilizes the Bernoulli equation to impose a total pressure condition,
 * typical for low-speed/incompressible boundary zones.
 *
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] Pt_target Target total pressure to impose \f$ P_t \f$
 * @param[in] gamma Specific heat ratio \f$ \gamma \f$
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 */
void build_total_pressure_incomp_ghost(const double* face_state, double Pt_target,
                                       double gamma, double* neigh_state);

/**
 * @brief Construct a static pressure ghost state.
 *
 * Extrapolates density and velocity from the interior face state, but replaces
 * the energy component to exactly match the target static pressure.
 *
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] P_target Target static pressure to impose \f$ P \f$
 * @param[in] gamma Specific heat ratio \f$ \gamma \f$
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 */
void build_static_pressure_ghost(const double* face_state, double P_target,
                                 double gamma, double* neigh_state);


/**
 * @file boundary.hpp
 * @brief Declarations for modular boundary condition utilities and ghost state generators.
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
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] u_wall Prescribed wall boundary velocity component in the x-direction
 * @param[in] v_wall Prescribed wall boundary velocity component in the y-direction
 * @param[in] gamma Specific heat ratio
 * @param[in] isothermal True to impose an isothermal wall boundary, false for adiabatic
 * @param[in] T_wall Prescribed target wall temperature (used only if isothermal is true)
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
 * Upwinds invariants, entropy, and tangential velocity according to local waves.
 *
 * @param[in] face_state Interior face-extrapolated conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] ref_state Exterior physical reference state \f$ [\rho, \rho u, \rho v, E] \f$
 * @param[in] n_x Outer unit normal coordinate direction component x
 * @param[in] n_y Outer unit normal coordinate direction component y
 * @param[in] gamma Specific heat ratio
 * @param[out] neigh_state Reconstructed ghost conservative state \f$ [\rho, \rho u, \rho v, E] \f$
 */
void build_characteristic_ghost(const double* face_state, const double* ref_state,
                                double n_x, double n_y, double gamma,
                                double* neigh_state);

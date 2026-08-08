/**
 * @file ib_wall_function.hpp
 * @brief Declarations for Compressible Off-Body Law-of-the-Wall Model for Immersed Boundaries.
 */

#pragma once

#include <cmath>

namespace fr::ib {

/**
 * @struct WallFunctionResult
 * @brief Outputs from the Van Driest compressible law-of-the-wall solver.
 */
struct WallFunctionResult {
    double u_tau = 0.0;       ///< Friction velocity (m/s)
    double tau_w = 0.0;       ///< Wall shear stress (Pa)
    double q_w = 0.0;         ///< Wall heat flux (W/m^2)
    double u_wall_eff = 0.0;  ///< Effective wall slip velocity for ghost node state coupling (m/s)
    double y_plus = 0.0;      ///< Calculated non-dimensional wall distance y+
};

/**
 * @class WallFunctionModel
 * @brief Off-body compressible turbulent boundary layer wall function model.
 */
class WallFunctionModel {
public:
    /**
     * @brief Construct wall function model with law-of-the-wall constants.
     * @param kappa Von Kármán constant (default: 0.41)
     * @param B Log-law intercept constant (default: 5.2)
     */
    explicit WallFunctionModel(double kappa = 0.41, double B = 5.2)
        : kappa_(kappa), B_(B) {}

    /**
     * @brief Solve Van Driest compressible law-of-the-wall at off-body sampling height delta_wf.
     * @param delta_wf Physical sampling distance from wall (m)
     * @param rho_wf Fluid density at sampling location (kg/m^3)
     * @param u_parallel Tangential fluid velocity magnitude relative to body (m/s)
     * @param T_wf Fluid temperature at sampling location (K)
     * @param T_wall Wall temperature (K)
     * @param mu_wf Dynamic viscosity at sampling location (Pa s)
     * @param gamma Ratio of specific heats
     * @param max_iter Maximum Newton-Raphson iterations
     * @param tol Newton-Raphson convergence tolerance
     * @return WallFunctionResult struct containing u_tau, tau_w, q_w, u_wall_eff, y_plus
     */
    WallFunctionResult solve_wall_function(double delta_wf, double rho_wf, double u_parallel,
                                           double T_wf, double T_wall, double mu_wf, double gamma,
                                           int max_iter = 50, double tol = 1e-6) const;

private:
    double kappa_;
    double B_;

    /**
     * @brief Newton-Raphson solver for friction velocity u_tau.
     */
    double solve_u_tau(double delta_wf, double rho_w, double u_parallel, double mu_w,
                       int max_iter, double tol) const;
};

} // namespace fr::ib

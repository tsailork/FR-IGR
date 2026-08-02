/**
 * @file ppr.hpp
 * @brief Hyperbolic Non-Equilibrium Phantom Pressure Relaxation (PPR) shock capturing module.
 */

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "../core/parameters.hpp"
#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "../core/solver.hpp"

namespace PPR {

/**
 * @brief Computes physical, phantom, and regularized pressures and regularized acoustic sound speed.
 */
inline void get_thermodynamics(double rho, double rhou, double rhov, double E, double S,
                              double theta, double gamma, double eps,
                              double& P_phys, double& P_phan, double& P_reg, double& a_reg)
{
    double u = rhou / rho;
    double v = rhov / rho;
    double e_kin = 0.5 * rho * (u * u + v * v);
    P_phys = std::max(eps, (gamma - 1.0) * (E - e_kin));
    P_phan = S / rho;
    P_reg  = std::max(eps, P_phys + theta * (P_phys - P_phan));

    double a2_reg = ((1.0 + std::max(0.0, theta)) / rho) * (P_phys + (gamma - 1.0) * P_reg);
    double a2_phys = gamma * P_phys / rho;
    a_reg = std::sqrt(std::max(a2_reg, a2_phys));
}

/**
 * @brief Evaluates regularized sound speed given state variables and element theta.
 */
inline double get_a_reg(double rho, double rhou, double rhov, double E, double S,
                        double theta, double gamma, double eps)
{
    double P_phys, P_phan, P_reg, a_reg;
    get_thermodynamics(rho, rhou, rhov, E, S, theta, gamma, eps, P_phys, P_phan, P_reg, a_reg);
    return a_reg;
}

inline void get_thermodynamics_3d(double rho, double rhou, double rhov, double rhow, double E, double S,
                                 double theta, double gamma, double eps,
                                 double& P_phys, double& P_phan, double& P_reg, double& a_reg)
{
    double u = rhou / rho;
    double v = rhov / rho;
    double w = rhow / rho;
    double e_kin = 0.5 * rho * (u * u + v * v + w * w);
    P_phys = std::max(eps, (gamma - 1.0) * (E - e_kin));
    P_phan = S / rho;
    P_reg  = std::max(eps, P_phys + theta * (P_phys - P_phan));

    double a2_reg = ((1.0 + std::max(0.0, theta)) / rho) * (P_phys + (gamma - 1.0) * P_reg);
    double a2_phys = gamma * P_phys / rho;
    a_reg = std::sqrt(std::max(a2_reg, a2_phys));
}

inline double get_a_reg_3d(double rho, double rhou, double rhov, double rhow, double E, double S,
                           double theta, double gamma, double eps)
{
    double P_phys, P_phan, P_reg, a_reg;
    get_thermodynamics_3d(rho, rhou, rhov, rhow, E, S, theta, gamma, eps, P_phys, P_phan, P_reg, a_reg);
    return a_reg;
}

/**
 * @brief Computes element-constant theta_e for 2D cells using nondimensional velocity divergence,
 *        Mach scaling, Thermodynamic Energy Guard, and face-neighbor max smoothing.
 */
void compute_element_theta_2d(const std::vector<CellDim<2>*>& cells,
                             const Basis& basis,
                             const Parameters& p);

void compute_element_theta_3d(const std::vector<CellDim<3>*>& cells,
                             const Basis& basis,
                             const Parameters& p);

/**
 * @brief Pointwise analytical operator-splitting relaxation step for phantom pressure S:
 *        S_new = rho * P_phys + (S - rho * P_phys) * exp(-dt_stage / tau)
 */
void relax_phantom_pressure_2d(CellDim<2>& cell, double dt_stage, const Basis& basis, const Parameters& p);
void relax_phantom_pressure_3d(CellDim<3>& cell, double dt_stage, const Basis& basis, const Parameters& p);

/**
 * @brief Nodal-stencil phantom pressure bound limiter.
 *        Clamps P_phan using neighborhood min/max physical pressure and analytical bounds.
 */
void apply_phantom_pressure_limiter_2d(const std::vector<CellDim<2>*>& cells, const Basis& basis, const Parameters& p);
void apply_phantom_pressure_limiter_3d(const std::vector<CellDim<3>*>& cells, const Basis& basis, const Parameters& p);

} // namespace PPR

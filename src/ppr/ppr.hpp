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
#include "../apsr/apsr.hpp"

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

/**
 * @brief Evaluates regularized pressure and APSR anisotropic stress tensor components at a solution point.
 */
inline void get_pointwise_regularization(const CellDim<2>& c, int iy, int ix,
                                         const Basis& basis, const Parameters& p,
                                         double press_phys, double& press_eff,
                                         double& tau_xx, double& tau_xy, double& tau_yy)
{
    tau_xx = 0.0; tau_xy = 0.0; tau_yy = 0.0;
    press_eff = press_phys;

    if (!p.ENABLE_PPR) return;

    double rho = std::max(p.POS_LIMITER_EPS, c.get_U(0, iy, ix, p.N_PTS));

    if (p.ENABLE_APSR) {
        double dP_dx = 0.0, dP_dy = 0.0;
        for (int k = 0; k < p.N_PTS; ++k) {
            double r_x = std::max(p.POS_LIMITER_EPS, c.get_U(0, iy, k, p.N_PTS));
            double u_x = c.get_U(1, iy, k, p.N_PTS) / r_x;
            double v_x = c.get_U(2, iy, k, p.N_PTS) / r_x;
            double E_x = c.get_U(3, iy, k, p.N_PTS);
            double p_x = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E_x - 0.5 * r_x * (u_x * u_x + v_x * v_x)));

            double r_y = std::max(p.POS_LIMITER_EPS, c.get_U(0, k, ix, p.N_PTS));
            double u_y = c.get_U(1, k, ix, p.N_PTS) / r_y;
            double v_y = c.get_U(2, k, ix, p.N_PTS) / r_y;
            double E_y = c.get_U(3, k, ix, p.N_PTS);
            double p_y = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E_y - 0.5 * r_y * (u_y * u_y + v_y * v_y)));

            dP_dx += basis.D[ix][k] * p_x;
            dP_dy += basis.D[iy][k] * p_y;
        }
        dP_dx *= (2.0 / c.dx);
        dP_dy *= (2.0 / c.dy);

        double grad_mag = std::sqrt(dP_dx * dP_dx + dP_dy * dP_dy);
        double n_x = 1.0, n_y = 0.0;
        if (grad_mag > 1e-10) {
            n_x = dP_dx / grad_mag;
            n_y = dP_dy / grad_mag;
        }

        double P_phan = c.S_field[iy * p.N_PTS + ix] / rho;
        double tau_mag = c.theta_avg * 0.5 * (press_phys - P_phan + std::sqrt((press_phys - P_phan)*(press_phys - P_phan) + p.POS_LIMITER_EPS * p.POS_LIMITER_EPS));
        double alpha = p.APSR_ALPHA;

        tau_xx = tau_mag * ((1.0 - alpha) * n_x * n_x + alpha);
        tau_xy = tau_mag * (1.0 - alpha) * n_x * n_y;
        tau_yy = tau_mag * ((1.0 - alpha) * n_y * n_y + alpha);
    } else {
        double P_phan = c.S_field[iy * p.N_PTS + ix] / rho;
        double P_reg  = press_phys + c.theta_avg * (press_phys - P_phan);
        press_eff = std::max(p.POS_LIMITER_EPS, P_reg);
    }
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

namespace PPR {

/**
 * @brief Pre-conditions left and right interface states for Riemann solver when PPR/APSR is active.
 */
inline void prepare_riemann_states(const double UL[4], const double UR[4],
                                   double SL, double SR,
                                   double thetaL, double thetaR, int dir,
                                   const Parameters& p,
                                   double pL, double pR,
                                   double& pL_reg, double& pR_reg,
                                   double& aL_reg, double& aR_reg)
{
    pL_reg = pL;
    pR_reg = pR;
    double rhoL = std::max(p.POS_LIMITER_EPS, UL[0]);
    double rhoR = std::max(p.POS_LIMITER_EPS, UR[0]);
    aL_reg = std::sqrt(p.GAMMA * pL / rhoL);
    aR_reg = std::sqrt(p.GAMMA * pR / rhoR);

    if (!p.ENABLE_PPR) return;

    if (p.ENABLE_APSR) {
        double P_physL = pL, P_physR = pR, P_phanL, P_phanR;
        APSR::get_thermodynamics_apsr_2d(UL, SL, p, P_physL, P_phanL, pL_reg, aL_reg);
        APSR::get_thermodynamics_apsr_2d(UR, SR, p, P_physR, P_phanR, pR_reg, aR_reg);
        if (dir == 1) {
            pL_reg = pL + p.APSR_ALPHA * (pL_reg - pL);
            pR_reg = pR + p.APSR_ALPHA * (pR_reg - pR);
            aL_reg = std::sqrt(p.GAMMA * pL_reg / rhoL);
            aR_reg = std::sqrt(p.GAMMA * pR_reg / rhoR);
        }
    } else {
        double theta_f = 0.5 * (thetaL + thetaR);
        double P_phanL, P_phanR;
        get_thermodynamics(rhoL, UL[1], UL[2], UL[3], SL, theta_f, p.GAMMA, p.POS_LIMITER_EPS, pL, P_phanL, pL_reg, aL_reg);
        get_thermodynamics(rhoR, UR[1], UR[2], UR[3], SR, theta_f, p.GAMMA, p.POS_LIMITER_EPS, pR, P_phanR, pR_reg, aR_reg);
    }
}

} // namespace PPR

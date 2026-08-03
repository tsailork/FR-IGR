/**
 * @file apsr.hpp
 * @brief Rectified Anisotropic Phantom Stress Relaxation (APSR-R) physics engine.
 */

#ifndef APSR_HPP
#define APSR_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"

namespace APSR {

/**
 * @brief Non-negative C^\infty smooth ReLU function: 0.5 * (x + sqrt(x^2 + eps^2)).
 * Strictly positive (> 0) for all real x, zero offset, guarantees non-negative dissipation.
 */
inline double smooth_relu(double x, double eps = 1e-12) {
    return 0.5 * (x + std::sqrt(x * x + eps * eps));
}

/**
 * @brief Ducros vorticity filter: (div u)^2 / ((div u)^2 + ||curl u||^2 + eps).
 * Returns 1.0 for pure compressional shock waves, and 0.0 for pure shear flows / vortex cores.
 */
inline double compute_ducros_filter(double div_u, double curl_u_sq, double eps = 1e-8) {
    double div_sq = div_u * div_u;
    return div_sq / (div_sq + curl_u_sq + eps);
}

/**
 * @brief Computes 2D element-wise APSR coupling parameter theta_max from 1D shock-width theory.
 */
void compute_element_theta_apsr_2d(const std::vector<CellDim<2>*>& cells,
                                   const Basis& basis,
                                   const Parameters& p);

/**
 * @brief Computes 3D element-wise APSR coupling parameter theta_max from 1D shock-width theory.
 */
void compute_element_theta_apsr_3d(const std::vector<CellDim<3>*>& cells,
                                   const Basis& basis,
                                   const Parameters& p);

/**
 * @brief Evaluates node-point anisotropic phantom stress tensor components in 2D (tau_xx, tau_xy, tau_yy).
 */
void compute_stress_tensor_2d(const std::vector<CellDim<2>*>& cells,
                              const Basis& basis,
                              const Parameters& p);

/**
 * @brief Evaluates node-point anisotropic phantom stress tensor components in 3D (tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz).
 */
void compute_stress_tensor_3d(const std::vector<CellDim<3>*>& cells,
                              const Basis& basis,
                              const Parameters& p);

/**
 * @brief 2D APSR thermodynamics and regularized Whitham sound speed evaluation.
 */
inline void get_thermodynamics_apsr_2d(const double U[4],
                                       double S_val,
                                       const Parameters& p,
                                       double& P_phys,
                                       double& P_phan,
                                       double& P_reg,
                                       double& a_reg,
                                       double n_x = 1.0,
                                       double n_y = 0.0) {
    double rho = std::max(p.POS_LIMITER_EPS, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;
    double E = U[3];

    double e_kin = 0.5 * rho * (u * u + v * v);
    P_phys = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E - e_kin));
    P_phan = S_val / rho;

    double theta = p.PPR_CONSTANT_THETA;
    double P_diff = P_phys - P_phan;

    if (p.PPR_USE_LIMITER) {
        double p_diff_min = -p.PPR_C_POS * P_phys;
        double p_diff_max =  p.PPR_C_MAX * P_phys;
        P_diff = std::clamp(P_diff, p_diff_min, p_diff_max);
        P_phan = P_phys - P_diff;
    }

    P_reg = P_phys + theta * P_diff;
    P_reg = std::max(p.POS_LIMITER_EPS, P_reg);

    // Whitham non-equilibrium sound speed
    double a2_reg = ((1.0 + std::max(0.0, theta)) / rho) * (P_phys + (p.GAMMA - 1.0) * P_reg);
    a_reg = std::sqrt(std::max(1e-12, a2_reg));
}

} // namespace APSR

#endif // APSR_HPP

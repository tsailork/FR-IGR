/**
 * @file ib_wall_function.cpp
 * @brief Implementation of Compressible Off-Body Law-of-the-Wall Model for Immersed Boundaries.
 */

#include "ib_wall_function.hpp"
#include <cmath>
#include <algorithm>

namespace fr::ib {

double WallFunctionModel::solve_u_tau(double delta_wf, double rho_w, double u_parallel, double mu_w,
                                     int max_iter, double tol) const {
    if (u_parallel < 1e-12 || delta_wf < 1e-12 || rho_w < 1e-12 || mu_w < 1e-14) {
        return 0.0;
    }

    // Initial guess based on blasius/empirical skin friction relation
    double u_tau = 0.05 * u_parallel;

    for (int iter = 0; iter < max_iter; ++iter) {
        double y_plus = std::max(1e-6, rho_w * u_tau * delta_wf / mu_w);
        
        // Log-law profile equation f(u_tau) = 0
        double log_term = std::log(y_plus);
        double f = u_tau * ((1.0 / kappa_) * log_term + B_) - u_parallel;
        double f_prime = (1.0 / kappa_) * log_term + B_ + (1.0 / kappa_);

        if (std::abs(f_prime) < 1e-14) break;

        double delta_u_tau = f / f_prime;
        u_tau -= delta_u_tau;
        u_tau = std::max(1e-8, u_tau);

        if (std::abs(delta_u_tau) < tol * u_tau) {
            break;
        }
    }

    return u_tau;
}

WallFunctionResult WallFunctionModel::solve_wall_function(double delta_wf, double rho_wf, double u_parallel,
                                                     double T_wf, double T_wall, double mu_wf, double gamma,
                                                     int max_iter, double tol) const {
    WallFunctionResult res;
    if (u_parallel < 1e-12 || delta_wf < 1e-12) {
        return res;
    }

    // Wall density via equation of state (assuming constant pressure across boundary layer)
    double rho_w = rho_wf * (T_wf / std::max(1e-6, T_wall));
    double mu_w = mu_wf * std::pow(std::max(1e-6, T_wall / T_wf), 0.7); // Sutherland power-law approximation

    // Solve for friction velocity u_tau
    res.u_tau = solve_u_tau(delta_wf, rho_w, u_parallel, mu_w, max_iter, tol);

    // Calculate y+ non-dimensional height
    res.y_plus = std::max(1e-6, rho_w * res.u_tau * delta_wf / std::max(1e-14, mu_w));

    // Calculate wall shear stress tau_w
    res.tau_w = rho_w * res.u_tau * res.u_tau;

    // Calculate wall heat flux q_w (Reynolds analogy)
    double Cp = 1004.5; // J/(kg K) for air
    double Pr = 0.72;
    res.q_w = (res.tau_w / std::max(1e-12, u_parallel)) * Cp * (T_wall - T_wf) / std::pow(Pr, 2.0 / 3.0);

    // Calculate effective slip velocity for ghost cell coupling
    double u_plus = (1.0 / kappa_) * std::log(res.y_plus) + B_;
    double u_log = res.u_tau * u_plus;
    res.u_wall_eff = std::max(0.0, u_parallel - u_log);

    return res;
}

} // namespace fr::ib

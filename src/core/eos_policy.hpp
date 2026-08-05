/**
 * @file eos_policy.hpp
 * @brief Zero-overhead compile-time Equation of State (EOS) physical policy classes.
 */

#pragma once
#include <cmath>
#include <algorithm>
#include "constants.hpp"

namespace fr::physics {

/**
 * @struct IdealGasEOS
 * @brief Compile-time policy for Ideal Gas thermodynamics (2D/3D).
 */
struct IdealGasEOS {
    /**
     * @brief Compute thermodynamic pressure in 2D.
     */
    [[nodiscard]] static inline double pressure(double rho, double u, double v, double E, double gamma) noexcept {
        double r = std::max(fr::constants::POS_LIMITER_EPS, rho);
        double e_kin = 0.5 * r * (u * u + v * v);
        return std::max(fr::constants::POS_LIMITER_EPS, (gamma - 1.0) * (E - e_kin));
    }

    /**
     * @brief Compute thermodynamic pressure in 3D.
     */
    [[nodiscard]] static inline double pressure_3d(double rho, double u, double v, double w, double E, double gamma) noexcept {
        double r = std::max(fr::constants::POS_LIMITER_EPS, rho);
        double e_kin = 0.5 * r * (u * u + v * v + w * w);
        return std::max(fr::constants::POS_LIMITER_EPS, (gamma - 1.0) * (E - e_kin));
    }

    /**
     * @brief Compute speed of sound \f$ c = \sqrt{\gamma p / \rho} \f$.
     */
    [[nodiscard]] static inline double speed_of_sound(double rho, double pressure, double gamma) noexcept {
        double r = std::max(fr::constants::POS_LIMITER_EPS, rho);
        double p = std::max(fr::constants::POS_LIMITER_EPS, pressure);
        return std::sqrt(gamma * p / r);
    }

    /**
     * @brief Compute total energy \f$ E = \frac{p}{\gamma - 1} + \frac{1}{2} \rho (u^2 + v^2) \f$.
     */
    [[nodiscard]] static inline double total_energy(double rho, double u, double v, double p, double gamma) noexcept {
        double r = std::max(fr::constants::POS_LIMITER_EPS, rho);
        double pr = std::max(fr::constants::POS_LIMITER_EPS, p);
        return pr / (gamma - 1.0) + 0.5 * r * (u * u + v * v);
    }
};

} // namespace fr::physics

/**
 * @file implicit_integrator.cpp
 * @brief Implementation of Encapsulated Manager Engine for the FR-IGR Implicit Subsystem.
 */

#include "implicit_integrator.hpp"
#include <iostream>
#include <algorithm>

namespace fr::implicit {

void ImplicitIntegrator2D::initialize(const Parameters& params) {
    step_counter = 0;
    current_implicit_cfl = params.IMPLICIT_CFL;
    if (params.TIME_INTEGRATOR == "ESDIRK34") {
        if (params.IMPLICIT_CFL_MODE == "RAMP" || params.IMPLICIT_CFL_MODE == "ADAPTIVE") {
            current_implicit_cfl = params.IMPLICIT_CFL_MIN;
        }
    }
}

bool ImplicitIntegrator2D::step(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    double dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective
) {
    // Update CFL multiplier for RAMP mode
    if (params.IMPLICIT_CFL_MODE == "RAMP") {
        if (step_counter < params.IMPLICIT_CFL_RAMP_STEPS && params.IMPLICIT_CFL_RAMP_STEPS > 0) {
            double frac = static_cast<double>(step_counter) / static_cast<double>(params.IMPLICIT_CFL_RAMP_STEPS);
            current_implicit_cfl = params.IMPLICIT_CFL_MIN + frac * (params.IMPLICIT_CFL_MAX - params.IMPLICIT_CFL_MIN);
        } else {
            current_implicit_cfl = params.IMPLICIT_CFL_MAX;
        }
    }

    double dt_attempt = dt;
    int max_retries = 3;
    bool step_success = false;

    for (int retry = 0; retry < max_retries; ++retry) {
        step_success = step_esdirk34_2d(
            cells, basis, params, dt_attempt,
            compute_R_effective, precond_2d, ilu_precond_2d,
            step_counter, last_stats
        );

        if (step_success) {
            last_stats.current_cfl = current_implicit_cfl;

            // Adaptive CFL scaling based on Newton iterations
            if (params.IMPLICIT_CFL_MODE == "ADAPTIVE") {
                if (last_stats.total_newton_iters <= params.IMPLICIT_CFL_TARGET_NEWTON_ITERS) {
                    current_implicit_cfl = std::min(params.IMPLICIT_CFL_MAX, current_implicit_cfl * params.IMPLICIT_CFL_GROWTH_FACTOR);
                } else {
                    current_implicit_cfl = std::max(params.IMPLICIT_CFL_MIN, current_implicit_cfl * params.IMPLICIT_CFL_REDUCTION_FACTOR);
                }
            }
            return true;
        }

        std::cout << "[WARNING] Implicit stage solve failed at step " << step_counter
                  << ". Retrying with reduced dt (" << dt_attempt * 0.5 << ")...\n";
        dt_attempt *= 0.5;
        if (params.IMPLICIT_CFL_MODE == "ADAPTIVE") {
            current_implicit_cfl = std::max(params.IMPLICIT_CFL_MIN, current_implicit_cfl * params.IMPLICIT_CFL_REDUCTION_FACTOR);
        }
    }

    return false;
}

} // namespace fr::implicit

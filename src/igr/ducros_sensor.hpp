/**
 * @file ducros_sensor.hpp
 * @brief Standalone sensor module for Ducros vorticity switch, non-dimensional velocity divergence,
 *        and piecewise linear schedule interpolation for PPR parameter theta.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"
#include "../core/solver.hpp"

namespace Sensors {

/**
 * @brief Computes the Ducros sensor value for separating shocks from vortical turbulence.
 * @param div_u Divergence of velocity field (div u = du/dx + dv/dy).
 * @param curl_u Curl of velocity field (curl u = dv/dx - du/dy).
 * @param eps Small safety constant to avoid 0/0.
 * @return Capped Ducros sensor in range [0, 1].
 */
inline double compute_ducros_sensor(double div_u, double curl_u, double eps = 1e-16) {
    double div2  = div_u * div_u;
    double curl2 = curl_u * curl_u;
    double val   = div2 / (div2 + curl2 + eps);
    return std::clamp(val, 0.0, 1.0);
}

/**
 * @brief Performs 1D piecewise linear interpolation over schedule vectors.
 * @param sensor_val Input sensor scalar (x-coordinate).
 * @param sens_sched Monotonically increasing sensor breakpoints.
 * @param theta_sched Corresponding target theta values.
 * @return Interpolated theta value.
 */
double interpolate_schedule(double sensor_val,
                             const std::vector<double>& sens_sched,
                             const std::vector<double>& theta_sched);

/**
 * @brief Evaluates the combined Ducros x non-dimensional divergence sensor at a solution point.
 * @param c Cell containing the solution state.
 * @param iy Local Y solution point index.
 * @param ix Local X solution point index.
 * @param basis FR basis containing derivative matrices.
 * @param p Global simulation parameters.
 * @param u_buf Pre-computed local u velocity matrix (size N_PTS x N_PTS).
 * @param v_buf Pre-computed local v velocity matrix (size N_PTS x N_PTS).
 * @return Combined sensor scalar S = s_ducros * div_nd.
 */
double compute_combined_ppr_sensor(const Cell& c, int iy, int ix,
                                   const Basis& basis, const Parameters& p,
                                   const double u_buf[MAX_PTS][MAX_PTS],
                                   const double v_buf[MAX_PTS][MAX_PTS]);

} // namespace Sensors

/**
 * @file ducros_sensor.cpp
 * @brief Implementation of the standalone Ducros sensor and theta schedule interpolation routines.
 */

#include "ducros_sensor.hpp"

namespace Sensors {

double interpolate_schedule(double sensor_val,
                             const std::vector<double>& sens_sched,
                             const std::vector<double>& theta_sched)
{
    if (sens_sched.empty() || theta_sched.empty()) return 0.0;
    if (sens_sched.size() != theta_sched.size()) return theta_sched[0];
    if (sens_sched.size() == 1) return theta_sched[0];

    // Left extrapolation / floor
    if (sensor_val <= sens_sched.front()) {
        return theta_sched.front();
    }
    // Right extrapolation / ceiling
    if (sensor_val >= sens_sched.back()) {
        return theta_sched.back();
    }

    // Binary search / segment lookup
    for (size_t k = 0; k < sens_sched.size() - 1; ++k) {
        if (sensor_val >= sens_sched[k] && sensor_val <= sens_sched[k+1]) {
            double ds = sens_sched[k+1] - sens_sched[k];
            if (std::abs(ds) < 1e-15) return theta_sched[k];
            double dt = theta_sched[k+1] - theta_sched[k];
            return theta_sched[k] + dt * (sensor_val - sens_sched[k]) / ds;
        }
    }

    return theta_sched.back();
}

double compute_combined_ppr_sensor(const Cell& c, int iy, int ix,
                                   const Basis& basis, const Parameters& p,
                                   const double u_buf[MAX_PTS][MAX_PTS],
                                   const double v_buf[MAX_PTS][MAX_PTS])
{
    // Compute velocity spatial derivatives via FR derivative matrix
    double du_dx = 0.0, dv_dx = 0.0;
    double du_dy = 0.0, dv_dy = 0.0;

    for (int k = 0; k < p.N_PTS; ++k) {
        du_dx += basis.D[ix][k] * u_buf[iy][k];
        dv_dx += basis.D[ix][k] * v_buf[iy][k];
        du_dy += basis.D[iy][k] * u_buf[k][ix];
        dv_dy += basis.D[iy][k] * v_buf[k][ix];
    }
    du_dx *= (2.0 / c.dx);
    dv_dx *= (2.0 / c.dx);
    du_dy *= (2.0 / c.dy);
    dv_dy *= (2.0 / c.dy);

    double div_u  = du_dx + dv_dy;
    double curl_u = dv_dx - du_dy;

    // Ducros sensor capped to [0, 1]
    double s_ducros = compute_ducros_sensor(div_u, curl_u);

    // Non-dimensional divergence (uncapped)
    double rho   = std::max(p.POS_LIMITER_EPS, c.get_U(0, iy, ix, p.N_PTS));
    double u_loc = u_buf[iy][ix];
    double v_loc = v_buf[iy][ix];
    double P_loc = std::max(p.POS_LIMITER_EPS,
        (p.GAMMA - 1.0) * (c.get_U(3, iy, ix, p.N_PTS) - 0.5 * rho * (u_loc*u_loc + v_loc*v_loc)));
    double a_loc  = std::sqrt(p.GAMMA * P_loc / rho);
    double h_loc  = std::min(c.dx, c.dy);
    double div_nd = -div_u * h_loc / (a_loc * (p.P_DEG + 1));

    // Combined sensor scalar: capped Ducros * uncapped non-dimensional divergence
    if (p.PPR_USE_DUCROS_SENSOR) {
        return s_ducros * div_nd;
    } else {
        return div_nd;
    }
}

} // namespace Sensors

/**
 * @file ib_motion.hpp
 * @brief Declarations for Dynamic Body Motion and Kinematic Trajectory Handler.
 */

#pragma once

#include <cmath>
#include <string>

class Parameters; // Forward declaration

namespace fr::ib {

/**
 * @struct MotionState
 * @brief Instantaneous kinematic motion state of an immersed body.
 */
struct MotionState {
    double pos[3] = {0.0, 0.0, 0.0};    ///< Center position offset (x_c, y_c, z_c)
    double vel[3] = {0.0, 0.0, 0.0};    ///< Translational velocity (u_b, v_b, w_b)
    double omega[3] = {0.0, 0.0, 0.0};  ///< Angular rotation velocity (omega_x, omega_y, omega_z)
};

/**
 * @class DynamicMotionHandler
 * @brief Manages dynamic motion and kinematic trajectories for immersed bodies on stationary grids.
 */
class DynamicMotionHandler {
public:
    DynamicMotionHandler() = default;

    /**
     * @brief Compute body motion state at simulation time t.
     * @param p Reference to Parameters database
     * @param time Current simulation time
     * @param node_pos Query point location (optional, for non-uniform rotational velocity)
     * @return MotionState struct
     */
    static MotionState evaluate_motion(const Parameters& p, double time, const double node_pos[3] = nullptr);
};

} // namespace fr::ib

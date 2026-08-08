/**
 * @file ib_motion.cpp
 * @brief Implementation of Dynamic Body Motion and Kinematic Trajectory Handler.
 */

#include "ib_motion.hpp"
#include "../core/parameters.hpp"
#include <cmath>

namespace fr::ib {

MotionState DynamicMotionHandler::evaluate_motion(const Parameters& p, double time, const double node_pos[3]) {
    MotionState state;

    // Base constant velocity
    state.vel[0] = p.IB_VELOCITY_X;
    state.vel[1] = p.IB_VELOCITY_Y;
    state.vel[2] = 0.0;

    if (!p.ENABLE_IB_DYNAMIC_MOTION) {
        return state;
    }

    static const double PI = 3.14159265358979323846;

    if (p.IB_MOTION_TYPE == "SINUSOIDAL") {
        // Sinusoidal position offsets
        state.pos[0] = p.IB_MOTION_AMPLITUDE_X * std::sin(2.0 * PI * p.IB_MOTION_FREQ_X * time + p.IB_MOTION_PHASE_X);
        state.pos[1] = p.IB_MOTION_AMPLITUDE_Y * std::sin(2.0 * PI * p.IB_MOTION_FREQ_Y * time + p.IB_MOTION_PHASE_Y);
        state.pos[2] = p.IB_MOTION_AMPLITUDE_Z * std::sin(2.0 * PI * p.IB_MOTION_FREQ_Z * time + p.IB_MOTION_PHASE_Z);

        // Sinusoidal velocity time derivatives
        state.vel[0] += p.IB_MOTION_AMPLITUDE_X * 2.0 * PI * p.IB_MOTION_FREQ_X *
                        std::cos(2.0 * PI * p.IB_MOTION_FREQ_X * time + p.IB_MOTION_PHASE_X);
        state.vel[1] += p.IB_MOTION_AMPLITUDE_Y * 2.0 * PI * p.IB_MOTION_FREQ_Y *
                        std::cos(2.0 * PI * p.IB_MOTION_FREQ_Y * time + p.IB_MOTION_PHASE_Y);
        state.vel[2] += p.IB_MOTION_AMPLITUDE_Z * 2.0 * PI * p.IB_MOTION_FREQ_Z *
                        std::cos(2.0 * PI * p.IB_MOTION_FREQ_Z * time + p.IB_MOTION_PHASE_Z);
    } else if (p.IB_MOTION_TYPE == "TRANSLATIONAL") {
        state.pos[0] = p.IB_VELOCITY_X * time;
        state.pos[1] = p.IB_VELOCITY_Y * time;
        state.pos[2] = 0.0;
    } else if (p.IB_MOTION_TYPE == "ROTATIONAL") {
        state.omega[2] = 2.0 * PI * p.IB_MOTION_FREQ_Z;
        if (node_pos) {
            double rx = node_pos[0] - p.IB_CENTER_X;
            double ry = node_pos[1] - p.IB_CENTER_Y;
            state.vel[0] += -state.omega[2] * ry;
            state.vel[1] +=  state.omega[2] * rx;
        }
    }

    return state;
}

} // namespace fr::ib

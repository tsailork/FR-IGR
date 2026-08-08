#include "../doctest.h"
#include "../../src/ib/ib_motion.hpp"
#include "../../src/core/parameters.hpp"

TEST_CASE("fr::ib::DynamicMotionHandler Motion State Evaluation") {
    Parameters p;
    p.ENABLE_IB_DYNAMIC_MOTION = true;
    p.IB_MOTION_TYPE = "SINUSOIDAL";
    p.IB_MOTION_AMPLITUDE_X = 0.0;
    p.IB_MOTION_AMPLITUDE_Y = 0.5;
    p.IB_MOTION_FREQ_Y = 2.0; // 2 Hz
    p.IB_MOTION_PHASE_Y = 0.0;

    SUBCASE("Stationary at t = 0 (velocity peak)") {
        auto state = fr::ib::DynamicMotionHandler::evaluate_motion(p, 0.0);
        static const double PI = 3.14159265358979323846;
        double expected_v = 0.5 * 2.0 * PI * 2.0; // 2 * pi m/s
        CHECK(state.pos[1] == doctest::Approx(0.0));
        CHECK(state.vel[1] == doctest::Approx(expected_v));
    }

    SUBCASE("Peak displacement at t = 1.0 / (4.0 * freq)") {
        double t_peak = 1.0 / 8.0; // t = 0.125 s
        auto state = fr::ib::DynamicMotionHandler::evaluate_motion(p, t_peak);
        CHECK(state.pos[1] == doctest::Approx(0.5));
        CHECK(state.vel[1] == doctest::Approx(0.0));
    }
}

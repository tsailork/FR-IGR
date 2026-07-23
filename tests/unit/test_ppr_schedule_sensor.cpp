/**
 * @file test_ppr_schedule_sensor.cpp
 * @brief Unit tests for the Ducros sensor, PPR theta schedule interpolation, and magnitude pressure difference.
 */

#include "../doctest.h"
#include "../../src/igr/ducros_sensor.hpp"
#include "../../src/core/parameters.hpp"

TEST_CASE("PPR - Ducros Sensor Unit Evaluation") {
    // Pure compression (div_u != 0, curl_u = 0) -> Ducros sensor should be 1.0
    double s_comp = Sensors::compute_ducros_sensor(-10.0, 0.0);
    CHECK(s_comp == doctest::Approx(1.0));

    // Pure shear layer (div_u = 0, curl_u != 0) -> Ducros sensor should be 0.0
    double s_shear = Sensors::compute_ducros_sensor(0.0, 15.0);
    CHECK(s_shear == doctest::Approx(0.0));

    // Equal compression and vorticity -> Ducros sensor should be 0.5
    double s_mixed = Sensors::compute_ducros_sensor(5.0, 5.0);
    CHECK(s_mixed == doctest::Approx(0.5));
}

TEST_CASE("PPR - Piecewise Linear Schedule Interpolation") {
    std::vector<double> sens_sched  = {-0.2, 0.0, 1.0, 1.5};
    std::vector<double> theta_sched = { 1.0, 0.0, 10.0, 50.0};

    // Below floor
    CHECK(Sensors::interpolate_schedule(-0.5, sens_sched, theta_sched) == doctest::Approx(1.0));

    // Exact floor breakpoint
    CHECK(Sensors::interpolate_schedule(-0.2, sens_sched, theta_sched) == doctest::Approx(1.0));

    // Zero breakpoint
    CHECK(Sensors::interpolate_schedule(0.0, sens_sched, theta_sched) == doctest::Approx(0.0));

    // Interpolation between 0.0 and 1.0 (at 0.5 -> 5.0)
    CHECK(Sensors::interpolate_schedule(0.5, sens_sched, theta_sched) == doctest::Approx(5.0));

    // Breakpoint at 1.0 -> 10.0
    CHECK(Sensors::interpolate_schedule(1.0, sens_sched, theta_sched) == doctest::Approx(10.0));

    // Interpolation between 1.0 and 1.5 (at 1.25 -> 30.0)
    CHECK(Sensors::interpolate_schedule(1.25, sens_sched, theta_sched) == doctest::Approx(30.0));

    // Above ceiling
    CHECK(Sensors::interpolate_schedule(2.0, sens_sched, theta_sched) == doctest::Approx(50.0));
}

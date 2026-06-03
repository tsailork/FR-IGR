#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include <cmath>


TEST_CASE("Conservation - blast_walls") {
    auto p = load_test_config("blast_walls");
    Solver solver(p);
    IC::apply(solver);
    double current_time = 0.0;
    
    double m0 = compute_total_mass(solver);
    double e0 = compute_total_energy(solver);
    
    while (current_time < p.T_FINAL - 1e-12) {
        double dt = solver.compute_dt();
        if (current_time + dt > p.T_FINAL) dt = p.T_FINAL - current_time;
        solver.step_rk3(dt);
        current_time += dt;
    }
    
    double m1 = compute_total_mass(solver);
    double e1 = compute_total_energy(solver);
    
    CHECK(std::abs(m1 - m0) < 1e-10);
    CHECK(std::abs(e1 - e0) < 1e-10);
}

TEST_CASE("Conservation - sine_periodic") {
    auto p = load_test_config("sine_periodic");
    Solver solver(p);
    IC::apply(solver);
    double current_time = 0.0;
    
    double m0 = compute_total_mass(solver);
    
    while (current_time < p.T_FINAL - 1e-12) {
        double dt = solver.compute_dt();
        if (current_time + dt > p.T_FINAL) dt = p.T_FINAL - current_time;
        solver.step_rk3(dt);
        current_time += dt;
    }
    
    double m1 = compute_total_mass(solver);
    CHECK(std::abs(m1 - m0) < 1e-10);
}

#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"

TEST_CASE("Boundary Integration - Characteristic BC") {
    auto p = load_test_config("characteristic_bc");
    Solver solver(p);
    IC::apply(solver);
    
    double initial_mass = compute_total_mass(solver);
    
    // Step a few times to ensure BCs are stable
    for (int i = 0; i < 5; ++i) {
        solver.step_rk3(solver.compute_dt());
    }
    
    assert_no_nan(solver);
}

TEST_CASE("Boundary Integration - Blast Walls") {
    auto p = load_test_config("blast_walls");
    Solver solver(p);
    IC::apply(solver);
    
    double initial_mass = compute_total_mass(solver);
    
    for (int i = 0; i < 5; ++i) {
        solver.step_rk3(solver.compute_dt());
    }
    
    assert_no_nan(solver);
    
    // With wall BCs, mass should be conserved exactly or very closely
    double final_mass = compute_total_mass(solver);
    CHECK(final_mass == doctest::Approx(initial_mass).epsilon(1e-10));
}

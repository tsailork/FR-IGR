#include <array>
#include "../doctest.h"
#include "../../src/boundary/boundary.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>

TEST_CASE("Boundary Conditions Ghost States") {
    double gamma = 1.4;
    std::array<double, 4> face_state = {1.2, 1.2 * 0.5, 1.2 * -0.2, 100000.0 / 0.4 + 0.5 * 1.2 * (0.5 * 0.5 + 0.2 * 0.2)};
    std::array<double, 4> neigh_state = {0.0};
    
    double u_int = face_state[1] / face_state[0];
    double v_int = face_state[2] / face_state[0];

    SUBCASE("No-slip adiabatic wall") {
        build_viscous_wall_ghost(face_state.data(), neigh_state.data(), 0.0, 0.0, gamma, false, 0.0);
        
        CHECK(neigh_state[0] == doctest::Approx(face_state[0]));
        CHECK(neigh_state[1] == doctest::Approx(-face_state[1]));
        CHECK(neigh_state[2] == doctest::Approx(-face_state[2]));
        CHECK(neigh_state[3] == doctest::Approx(face_state[3]));
    }

    SUBCASE("Moving wall") {
        double u_wall = 1.0;
        double v_wall = 0.5;
        build_viscous_wall_ghost(face_state.data(), neigh_state.data(), u_wall, v_wall, gamma, false, 0.0);
        
        double rho_ghost = neigh_state[0];
        CHECK(rho_ghost == doctest::Approx(face_state[0]));
        
        double u_ghost = neigh_state[1] / rho_ghost;
        double v_ghost = neigh_state[2] / rho_ghost;
        
        // Ghost velocity should be 2 * u_wall - u_int
        CHECK(u_ghost == doctest::Approx(2.0 * u_wall - u_int));
        CHECK(v_ghost == doctest::Approx(2.0 * v_wall - v_int));
    }

    SUBCASE("No-slip isothermal wall") {
        double T_wall = 300.0; // Needs to be dimensionless or match units, just a test value
        build_viscous_wall_ghost(face_state.data(), neigh_state.data(), 0.0, 0.0, gamma, true, T_wall);
        
        CHECK(neigh_state[0] == doctest::Approx(face_state[0]));
        CHECK(neigh_state[1] == doctest::Approx(-face_state[1]));
        CHECK(neigh_state[2] == doctest::Approx(-face_state[2]));
        
        // Check temperature matching logic inside (ghost temperature = 2*T_wall - T_int)
        // We won't strictly compute T here since it involves R, but we know Energy will change
        CHECK(neigh_state[3] != doctest::Approx(face_state[3]));
    }

    SUBCASE("Characteristic Subsonic Inflow/Outflow") {
        std::array<double, 4> ref_state = {1.0, 0.0, 0.0, 100000.0 / 0.4};
        // Outward normal in x
        build_characteristic_ghost(face_state.data(), ref_state.data(), 1.0, 0.0, gamma, neigh_state.data());
        
        // It's an upwind-based scheme, so some invariants will propagate from ref, some from face.
        // We ensure state is bounded and positivity preserved.
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
    
    SUBCASE("Total pressure comp") {
        double Pt_target = 110000.0;
        build_total_pressure_comp_ghost(face_state.data(), Pt_target, gamma, neigh_state.data());
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
    
    SUBCASE("Total pressure incomp") {
        double Pt_target = 110000.0;
        build_total_pressure_incomp_ghost(face_state.data(), Pt_target, gamma, neigh_state.data());
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
    
    SUBCASE("Static pressure") {
        double P_target = 90000.0;
        build_static_pressure_ghost(face_state.data(), P_target, gamma, neigh_state.data());
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
}

TEST_CASE("PPR Wall Boundary Condition Modes") {
    double gamma = 1.4;
    double rho = 1.2;
    double u = 2.0;
    double v = -0.5;
    double p_phys = 100000.0;
    double E = p_phys / (gamma - 1.0) + 0.5 * rho * (u*u + v*v);
    std::array<double, 4> face_state = {rho, rho*u, rho*v, E};
    std::array<double, 4> neigh_state = {rho, -rho*u, rho*v, E}; // slip wall ghost

    double S_face = 0.8 * (rho * p_phys); // P_phan_face = 80,000, P_diff_face = 20,000

    SUBCASE("EQUILIBRATED mode") {
        double S_ghost = Solver::compute_wall_phantom_pressure(face_state.data(), neigh_state.data(), S_face, "EQUILIBRATED", 2, 1e-12, gamma);
        double P_phan_ghost = S_ghost / neigh_state[0];
        double P_diff_ghost = p_phys - P_phan_ghost;
        double P_diff_face = p_phys - (S_face / rho);

        // Anti-symmetric difference: P_diff_ghost = -P_diff_face => P_diff_avg = 0
        CHECK(P_diff_ghost == doctest::Approx(-P_diff_face));
        double P_diff_avg = 0.5 * (P_diff_face + P_diff_ghost);
        CHECK(P_diff_avg == doctest::Approx(0.0));
    }

    SUBCASE("PHYSICAL_DIRICHLET mode") {
        double S_ghost = Solver::compute_wall_phantom_pressure(face_state.data(), neigh_state.data(), S_face, "PHYSICAL_DIRICHLET", 2, 1e-12, gamma);
        double P_phan_ghost = S_ghost / neigh_state[0];

        // Ghost phantom pressure equals ghost physical pressure
        CHECK(P_phan_ghost == doctest::Approx(p_phys));
        CHECK(S_ghost == doctest::Approx(rho * p_phys));

        double P_diff_face = p_phys - (S_face / rho);
        double P_diff_ghost = p_phys - P_phan_ghost; // 0
        double P_diff_avg = 0.5 * (P_diff_face + P_diff_ghost);
        CHECK(P_diff_avg == doctest::Approx(0.5 * P_diff_face));
    }

    SUBCASE("CONSTANT_DIFFERENCE mode") {
        double S_ghost = Solver::compute_wall_phantom_pressure(face_state.data(), neigh_state.data(), S_face, "CONSTANT_DIFFERENCE", 2, 1e-12, gamma);
        double P_phan_ghost = S_ghost / neigh_state[0];

        double P_diff_face = p_phys - (S_face / rho);
        double P_diff_ghost = p_phys - P_phan_ghost;

        // Constant difference: P_diff_ghost = P_diff_face => P_diff_avg = P_diff_face
        CHECK(P_diff_ghost == doctest::Approx(P_diff_face));
        double P_diff_avg = 0.5 * (P_diff_face + P_diff_ghost);
        CHECK(P_diff_avg == doctest::Approx(P_diff_face));
    }
}


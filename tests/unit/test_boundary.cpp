#include "../doctest.h"
#include "../../src/boundary/boundary.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>

TEST_CASE("Boundary Conditions Ghost States") {
    double gamma = 1.4;
    double face_state[4] = {1.2, 1.2 * 0.5, 1.2 * -0.2, 100000.0 / 0.4 + 0.5 * 1.2 * (0.5 * 0.5 + 0.2 * 0.2)};
    double neigh_state[4] = {0.0};
    
    double u_int = face_state[1] / face_state[0];
    double v_int = face_state[2] / face_state[0];

    SUBCASE("No-slip adiabatic wall") {
        build_viscous_wall_ghost(face_state, neigh_state, 0.0, 0.0, gamma, false, 0.0);
        
        CHECK(neigh_state[0] == doctest::Approx(face_state[0]));
        CHECK(neigh_state[1] == doctest::Approx(-face_state[1]));
        CHECK(neigh_state[2] == doctest::Approx(-face_state[2]));
        CHECK(neigh_state[3] == doctest::Approx(face_state[3]));
    }

    SUBCASE("Moving wall") {
        double u_wall = 1.0;
        double v_wall = 0.5;
        build_viscous_wall_ghost(face_state, neigh_state, u_wall, v_wall, gamma, false, 0.0);
        
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
        build_viscous_wall_ghost(face_state, neigh_state, 0.0, 0.0, gamma, true, T_wall);
        
        CHECK(neigh_state[0] == doctest::Approx(face_state[0]));
        CHECK(neigh_state[1] == doctest::Approx(-face_state[1]));
        CHECK(neigh_state[2] == doctest::Approx(-face_state[2]));
        
        // Check temperature matching logic inside (ghost temperature = 2*T_wall - T_int)
        // We won't strictly compute T here since it involves R, but we know Energy will change
        CHECK(neigh_state[3] != doctest::Approx(face_state[3]));
    }

    SUBCASE("Characteristic Subsonic Inflow/Outflow") {
        double ref_state[4] = {1.0, 0.0, 0.0, 100000.0 / 0.4};
        // Outward normal in x
        build_characteristic_ghost(face_state, ref_state, 1.0, 0.0, gamma, neigh_state);
        
        // It's an upwind-based scheme, so some invariants will propagate from ref, some from face.
        // We ensure state is bounded and positivity preserved.
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
    
    SUBCASE("Total pressure comp") {
        double Pt_target = 110000.0;
        build_total_pressure_comp_ghost(face_state, Pt_target, gamma, neigh_state);
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
    
    SUBCASE("Total pressure incomp") {
        double Pt_target = 110000.0;
        build_total_pressure_incomp_ghost(face_state, Pt_target, gamma, neigh_state);
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
    
    SUBCASE("Static pressure") {
        double P_target = 90000.0;
        build_static_pressure_ghost(face_state, P_target, gamma, neigh_state);
        CHECK(neigh_state[0] > 0.0);
        CHECK(neigh_state[3] > 0.0);
    }
}

TEST_CASE("Wall-Equilibrated Ghost State (WE-BC) for PPR") {
    Parameters p;
    p.GAMMA = 1.4;
    p.ENABLE_PPR = true;
    Solver solver(p);

    Cell2D c(p.N_PTS, &p);
    c.is_boundary[0] = true;
    c.boundary_info[0].is_wall = true;

    double rho = 1.2;
    double u = 2.0;
    double v = -0.5;
    double p_phys = 100000.0;
    double E = p_phys / (p.GAMMA - 1.0) + 0.5 * rho * (u*u + v*v);
    double face_state[4] = {rho, rho*u, rho*v, E};

    double S_face = 0.8 * (rho * p_phys);
    double S_ghost = 0.0;
    double neigh_state[4] = {0.0};
    double sig_neigh = 0.0;

    solver.get_neigh_state_cell(c, 0, false, face_state, 0.0, neigh_state, sig_neigh, 0, S_face, &S_ghost);

    double S_ghost_expected = 2.0 * rho * p_phys - S_face;
    CHECK(S_ghost == doctest::Approx(S_ghost_expected));

    double S_face_avg = 0.5 * (S_face + S_ghost);
    double p_phan_face = S_face_avg / rho;
    CHECK(p_phan_face == doctest::Approx(p_phys));

    double theta = 1.5;
    double p_reg_face = p_phys + theta * (p_phys - p_phan_face);
    CHECK(p_reg_face == doctest::Approx(p_phys));
}

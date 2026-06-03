#include "../doctest.h"
#include "../../src/io/initial_conditions.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>

TEST_CASE("Initial Conditions Application") {
    Parameters p;
    p.GAMMA = 1.4;
    p.N_PTS = 1;
    
    BlockConfig bc;
    bc.id = 0; bc.N_ELEM_X = 2; bc.N_ELEM_Y = 2;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    p.blocks.push_back(bc);

    SUBCASE("UNIFORM") {
        p.IC_TYPE = "UNIFORM";
        Solver solver(p);
        IC::apply(solver);
        
        double E_expected = 1.0 / 0.4;
        CHECK(solver.blocks[0].U(0, 0, 0, 0, 0) == doctest::Approx(1.0));
        CHECK(solver.blocks[0].U(1, 0, 0, 0, 0) == doctest::Approx(0.0));
        CHECK(solver.blocks[0].U(2, 0, 0, 0, 0) == doctest::Approx(0.0));
        CHECK(solver.blocks[0].U(3, 0, 0, 0, 0) == doctest::Approx(E_expected));
    }

    SUBCASE("FREESTREAM") {
        p.IC_TYPE = "FREESTREAM";
        p.RHO_INF = 1.2;
        p.U_INF = 0.5;
        p.V_INF = -0.5;
        p.P_INF = 100000.0;
        
        Solver solver(p);
        IC::apply(solver);
        
        double u2 = p.U_INF * p.U_INF + p.V_INF * p.V_INF;
        double E_expected = p.P_INF / 0.4 + 0.5 * p.RHO_INF * u2;
        
        CHECK(solver.blocks[0].U(0, 0, 0, 0, 0) == doctest::Approx(1.2));
        CHECK(solver.blocks[0].U(1, 0, 0, 0, 0) == doctest::Approx(1.2 * 0.5));
        CHECK(solver.blocks[0].U(2, 0, 0, 0, 0) == doctest::Approx(1.2 * -0.5));
        CHECK(solver.blocks[0].U(3, 0, 0, 0, 0) == doctest::Approx(E_expected));
    }

    SUBCASE("SINE_WAVE") {
        p.IC_TYPE = "SINE_WAVE";
        Solver solver(p);
        IC::apply(solver);
        
        // Element (0,0) center point x = 0.25 (since N_ELEM=2, width=0.5, dx=0.5 -> center=0.25)
        double rho_expected = 1.0 + 0.2 * std::sin(2.0 * M_PI * 0.25);
        double E_expected = 1.0 / 0.4 + 0.5 * rho_expected * 1.0;
        
        CHECK(solver.blocks[0].U(0, 0, 0, 0, 0) == doctest::Approx(rho_expected));
        CHECK(solver.blocks[0].U(1, 0, 0, 0, 0) == doctest::Approx(rho_expected * 1.0));
        CHECK(solver.blocks[0].U(2, 0, 0, 0, 0) == doctest::Approx(0.0));
        CHECK(solver.blocks[0].U(3, 0, 0, 0, 0) == doctest::Approx(E_expected));
    }

    SUBCASE("BLAST") {
        p.IC_TYPE = "BLAST";
        Solver solver(p);
        IC::apply(solver);
        
        CHECK(solver.blocks[0].U(0, 0, 0, 0, 0) == doctest::Approx(1.0));
        CHECK(solver.blocks[0].U(1, 0, 0, 0, 0) == doctest::Approx(0.0));
        CHECK(solver.blocks[0].U(2, 0, 0, 0, 0) == doctest::Approx(0.0));
        // Pressure is varying, we just check that Energy > 0
        CHECK(solver.blocks[0].U(3, 0, 0, 0, 0) > 0.0);
    }
    
    SUBCASE("LID_DRIVEN_CAVITY") {
        p.IC_TYPE = "LID_DRIVEN_CAVITY";
        p.RHO_INF = 1.0;
        p.P_INF = 1.0;
        Solver solver(p);
        IC::apply(solver);
        
        double E_expected = 1.0 / 0.4;
        CHECK(solver.blocks[0].U(0, 0, 0, 0, 0) == doctest::Approx(1.0));
        CHECK(solver.blocks[0].U(1, 0, 0, 0, 0) == doctest::Approx(0.0));
        CHECK(solver.blocks[0].U(2, 0, 0, 0, 0) == doctest::Approx(0.0));
        CHECK(solver.blocks[0].U(3, 0, 0, 0, 0) == doctest::Approx(E_expected));
    }
    
    SUBCASE("RIEMANN_2D_C3") {
        p.IC_TYPE = "RIEMANN_2D_C3";
        Solver solver(p);
        IC::apply(solver);
        
        // Element (0,0) is in bottom left
        // Element (0,1) is in bottom right
        // Element (1,0) is in top left
        // Element (1,1) is in top right
        
        // Bottom left should be near rBL = 0.138
        CHECK(solver.blocks[0].U(0, 0, 0, 0, 0) > 0.0);
        // Top right should be near rTR = 1.5
        CHECK(solver.blocks[0].U(0, 1, 1, 0, 0) > 0.0);
    }
}

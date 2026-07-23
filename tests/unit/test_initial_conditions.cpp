#include "../doctest.h"
#include "../../src/io/initial_conditions.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>

TEST_CASE("Initial Conditions Application") {
    Parameters p;
    p.GAMMA = 1.4;
    p.N_PTS = 1;
    int npts = p.N_PTS;
    
    BlockConfig bc;
    bc.id = 0; bc.N_ELEM_X = 2; bc.N_ELEM_Y = 2;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    p.blocks.push_back(bc);

    SUBCASE("UNIFORM") {
        p.IC_TYPE = "UNIFORM";
        Solver solver(p);
        IC::apply(solver);
        
        REQUIRE(solver.cells.size() >= 1);
        Cell* cell = solver.cells[0];
        
        double E_expected = 1.0 / 0.4;
        CHECK(cell->U[0 * npts * npts + 0 * npts + 0] == doctest::Approx(1.0));
        CHECK(cell->U[1 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        CHECK(cell->U[2 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        CHECK(cell->U[3 * npts * npts + 0 * npts + 0] == doctest::Approx(E_expected));
    }

    SUBCASE("FREESTREAM") {
        p.IC_TYPE = "FREESTREAM";
        p.RHO_INF = 1.2;
        p.U_INF = 0.5;
        p.V_INF = -0.5;
        p.P_INF = 100000.0;
        
        Solver solver(p);
        IC::apply(solver);
        
        REQUIRE(solver.cells.size() >= 1);
        Cell* cell = solver.cells[0];
        
        double u2 = p.U_INF * p.U_INF + p.V_INF * p.V_INF;
        double E_expected = p.P_INF / 0.4 + 0.5 * p.RHO_INF * u2;
        
        CHECK(cell->U[0 * npts * npts + 0 * npts + 0] == doctest::Approx(1.2));
        CHECK(cell->U[1 * npts * npts + 0 * npts + 0] == doctest::Approx(1.2 * 0.5));
        CHECK(cell->U[2 * npts * npts + 0 * npts + 0] == doctest::Approx(1.2 * -0.5));
        CHECK(cell->U[3 * npts * npts + 0 * npts + 0] == doctest::Approx(E_expected));
    }

    SUBCASE("SINE_WAVE") {
        p.IC_TYPE = "SINE_WAVE";
        Solver solver(p);
        IC::apply(solver);
        
        REQUIRE(solver.cells.size() >= 1);
        Cell* cell = solver.cells[0];
        
        // Element (0,0) center point x = 0.25 (since N_ELEM=2, width=0.5, dx=0.5 -> center=0.25)
        double rho_expected = 1.0 + 0.2 * std::sin(2.0 * M_PI * 0.25);
        double E_expected = 1.0 / 0.4 + 0.5 * rho_expected * 1.0;
        
        CHECK(cell->U[0 * npts * npts + 0 * npts + 0] == doctest::Approx(rho_expected));
        CHECK(cell->U[1 * npts * npts + 0 * npts + 0] == doctest::Approx(rho_expected * 1.0));
        CHECK(cell->U[2 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        CHECK(cell->U[3 * npts * npts + 0 * npts + 0] == doctest::Approx(E_expected));
    }

    SUBCASE("BLAST") {
        p.IC_TYPE = "BLAST";
        Solver solver(p);
        IC::apply(solver);
        
        REQUIRE(solver.cells.size() >= 1);
        Cell* cell = solver.cells[0];
        
        CHECK(cell->U[0 * npts * npts + 0 * npts + 0] == doctest::Approx(1.0));
        CHECK(cell->U[1 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        CHECK(cell->U[2 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        // Pressure is varying, we just check that Energy > 0
        CHECK(cell->U[3 * npts * npts + 0 * npts + 0] > 0.0);
    }
    
    SUBCASE("LID_DRIVEN_CAVITY") {
        p.IC_TYPE = "LID_DRIVEN_CAVITY";
        p.RHO_INF = 1.0;
        p.P_INF = 1.0;
        Solver solver(p);
        IC::apply(solver);
        
        REQUIRE(solver.cells.size() >= 1);
        Cell* cell = solver.cells[0];
        
        double E_expected = 1.0 / 0.4;
        CHECK(cell->U[0 * npts * npts + 0 * npts + 0] == doctest::Approx(1.0));
        CHECK(cell->U[1 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        CHECK(cell->U[2 * npts * npts + 0 * npts + 0] == doctest::Approx(0.0));
        CHECK(cell->U[3 * npts * npts + 0 * npts + 0] == doctest::Approx(E_expected));
    }
    
    SUBCASE("RIEMANN_2D_C3") {
        p.IC_TYPE = "RIEMANN_2D_C3";
        Solver solver(p);
        IC::apply(solver);
        
        // Cell ordering in a 2x2 grid (ey outer, ex inner):
        //   cells[0]: ey=0, ex=0  (bottom left)
        //   cells[1]: ey=0, ex=1  (bottom right)
        //   cells[2]: ey=1, ex=0  (top left)
        //   cells[3]: ey=1, ex=1  (top right)
        
        REQUIRE(solver.cells.size() >= 4);
        
        // Bottom left should be near rBL = 0.138
        Cell* cell_bl = solver.cells[0];
        CHECK(cell_bl->U[0 * npts * npts + 0 * npts + 0] > 0.0);
        // Top right should be near rTR = 1.5
        Cell* cell_tr = solver.cells[3];
        CHECK(cell_tr->U[0 * npts * npts + 0 * npts + 0] > 0.0);
    }

    SUBCASE("SHOCK_VORTEX") {
        p.IC_TYPE = "SHOCK_VORTEX";
        p.SHOCK_VORTEX_MS = 1.2;
        p.SHOCK_VORTEX_MV = 0.25;
        p.SHOCK_VORTEX_XS = 0.5;
        p.SHOCK_VORTEX_XV = 0.8;
        p.SHOCK_VORTEX_YV = 0.5;
        p.SHOCK_VORTEX_RC = 0.2;
        Solver solver(p);
        IC::apply(solver);

        REQUIRE(solver.cells.size() >= 4);
        for (const auto* cell : solver.cells) {
            CHECK(cell->U[0] > 0.0); // Density > 0
            CHECK(cell->U[3] > 0.0); // Total energy > 0
        }
    }

    SUBCASE("RICHTMYER_MESHKOV") {
        p.IC_TYPE = "RICHTMYER_MESHKOV";
        p.RMI_MS = 1.5;
        p.RMI_RHO1 = 1.0;
        p.RMI_RHO2 = 3.0;
        p.RMI_XS = 0.2;
        p.RMI_X0 = 0.5;
        p.RMI_AMP = 0.05;
        p.RMI_LY = 1.0;
        p.RMI_SIGMA = 0.01;
        Solver solver(p);
        IC::apply(solver);

        REQUIRE(solver.cells.size() >= 4);
        for (const auto* cell : solver.cells) {
            CHECK(cell->U[0] > 0.0); // Density > 0
            CHECK(cell->U[3] > 0.0); // Total energy > 0
        }
    }
}

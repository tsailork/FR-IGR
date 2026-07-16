#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>

TEST_CASE("Euler flux computation") {
    Parameters p;
    p.GAMMA = 1.4;
    Solver solver(p);
    
    BlockConfig bc;
    bc.id = 0; bc.N_ELEM_X = 1; bc.N_ELEM_Y = 1;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0; bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    bc.BC_L = "WALL"; bc.BC_R = "WALL"; bc.BC_B = "WALL"; bc.BC_T = "WALL";
    Block b(bc, 1);
    
    SUBCASE("State at rest") {
        b.U(0, 0, 0, 0, 0) = 1.0; // rho
        b.U(1, 0, 0, 0, 0) = 0.0; // rho*u
        b.U(2, 0, 0, 0, 0) = 0.0; // rho*v
        b.U(3, 0, 0, 0, 0) = 2.5; // E = p/(gamma-1) = 1.0/0.4 = 2.5 (pressure = 1.0)
        
        double F[4], G[4];
        solver.get_flux_pointwise(b, 0, 0, 0, 0, F, G, 0.0);
        
        CHECK(F[0] == doctest::Approx(0.0));
        CHECK(F[1] == doctest::Approx(1.0)); // pressure
        CHECK(F[2] == doctest::Approx(0.0));
        CHECK(F[3] == doctest::Approx(0.0));
        
        CHECK(G[0] == doctest::Approx(0.0));
        CHECK(G[1] == doctest::Approx(0.0));
        CHECK(G[2] == doctest::Approx(1.0)); // pressure
        CHECK(G[3] == doctest::Approx(0.0));
    }
    
    SUBCASE("X-velocity") {
        b.U(0, 0, 0, 0, 0) = 1.0; // rho
        b.U(1, 0, 0, 0, 0) = 2.0; // rho*u (u=2)
        b.U(2, 0, 0, 0, 0) = 0.0; // rho*v
        b.U(3, 0, 0, 0, 0) = 2.5 + 0.5 * 1.0 * 4.0; // E = 4.5 (pressure = 1.0)
        
        double F[4], G[4];
        solver.get_flux_pointwise(b, 0, 0, 0, 0, F, G, 0.0);
        
        CHECK(F[0] == doctest::Approx(2.0));       // rho*u
        CHECK(F[1] == doctest::Approx(4.0 + 1.0)); // rho*u^2 + p = 5.0
        CHECK(F[2] == doctest::Approx(0.0));       // rho*u*v
        CHECK(F[3] == doctest::Approx((4.5 + 1.0) * 2.0)); // (E + p) * u = 11.0
    }
    
    SUBCASE("Y-velocity") {
        b.U(0, 0, 0, 0, 0) = 1.0; // rho
        b.U(1, 0, 0, 0, 0) = 0.0; // rho*u
        b.U(2, 0, 0, 0, 0) = 3.0; // rho*v (v=3)
        b.U(3, 0, 0, 0, 0) = 2.5 + 0.5 * 1.0 * 9.0; // E = 7.0 (pressure = 1.0)
        
        double F[4], G[4];
        solver.get_flux_pointwise(b, 0, 0, 0, 0, F, G, 0.0);
        
        CHECK(G[0] == doctest::Approx(3.0));       // rho*v
        CHECK(G[1] == doctest::Approx(0.0));       // rho*v*u
        CHECK(G[2] == doctest::Approx(9.0 + 1.0)); // rho*v^2 + p = 10.0
        CHECK(G[3] == doctest::Approx((7.0 + 1.0) * 3.0)); // (E + p) * v = 24.0
    }
    
    SUBCASE("With IGR sigma") {
        b.U(0, 0, 0, 0, 0) = 1.0;
        b.U(1, 0, 0, 0, 0) = 1.0;
        b.U(2, 0, 0, 0, 0) = 1.0;
        b.U(3, 0, 0, 0, 0) = 3.5; // p = 1.0
        
        double sigma = 0.5;
        double F[4], G[4];
        solver.get_flux_pointwise(b, 0, 0, 0, 0, F, G, sigma);
        
        CHECK(F[1] == doctest::Approx(1.0 + 1.0 + 0.5)); // rho*u^2 + p + sigma = 2.5
        CHECK(F[3] == doctest::Approx((3.5 + 1.0 + 0.5) * 1.0)); // (E + p + sigma)*u = 5.0
        
        CHECK(G[2] == doctest::Approx(1.0 + 1.0 + 0.5)); // rho*v^2 + p + sigma = 2.5
        CHECK(G[3] == doctest::Approx((3.5 + 1.0 + 0.5) * 1.0)); // (E + p + sigma)*v = 5.0
    }
    
    SUBCASE("Symmetry") {
        double F_comm[4];
        double UL[4] = {1.0, 1.0, 0.0, 2.5 + 0.5}; // rho, u=1, v=0, E=3.0 (p=1)
        double UR[4] = {1.0, -1.0, 0.0, 2.5 + 0.5}; // rho, u=-1, v=0, E=3.0 (p=1)
        
        solver.solve_riemann(UL, UR, F_comm, 0);
        
        // Exact symmetry implies mass flux should be 0 due to symmetric opposing flows
        CHECK(F_comm[0] == doctest::Approx(0.0).epsilon(1e-12));
    }
}

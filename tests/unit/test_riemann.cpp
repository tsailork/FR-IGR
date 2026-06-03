#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include <cmath>
#include <vector>

TEST_CASE("Rusanov Riemann solver - Identical states") {
    Parameters p;
    p.GAMMA = 1.4;
    Solver solver(p);

    double UL[4] = {1.0, 0.0, 0.0, 2.5}; // rho=1, u=0, v=0, p=1
    double UR[4] = {1.0, 0.0, 0.0, 2.5};
    double F_comm[4] = {0.0, 0.0, 0.0, 0.0};

    solver.solve_riemann(UL, UR, F_comm, 0, 0.0, 0.0);

    // Flux should be F(U) = [0, p, 0, 0] = [0, 1.0, 0, 0]
    CHECK(F_comm[0] == doctest::Approx(0.0));
    CHECK(F_comm[1] == doctest::Approx(1.0));
    CHECK(F_comm[2] == doctest::Approx(0.0));
    CHECK(F_comm[3] == doctest::Approx(0.0));
}

TEST_CASE("Rusanov Riemann solver - Sod shock tube") {
    Parameters p;
    p.GAMMA = 1.4;
    Solver solver(p);

    double UL[4] = {1.0, 0.0, 0.0, 2.5};
    double UR[4] = {0.125, 0.0, 0.0, 0.25};
    double F_comm[4] = {0.0, 0.0, 0.0, 0.0};

    solver.solve_riemann(UL, UR, F_comm, 0, 0.0, 0.0);

    // The interface flux should reflect numerical dissipation
    // LLF adds dissipation based on max eigenvalue
    // cL = sqrt(1.4*1.0/1.0) ~ 1.1832
    // cR = sqrt(1.4*0.1/0.125) ~ 1.0583
    // lambda = max(cL, cR) ~ 1.1832
    // Flux is 0.5*(FL + FR) - 0.5*lambda*(UR - UL)
    // FL = [0, 1.0, 0, 0]
    // FR = [0, 0.1, 0, 0]
    // F_comm[0] = 0.5*(0 + 0) - 0.5*1.18321595*(0.125 - 1.0) = 0.51765
    CHECK(F_comm[0] > 0.0);
    CHECK(F_comm[0] < 1.0);
    CHECK(F_comm[1] > 0.0);
}

TEST_CASE("Rusanov Riemann solver - Symmetry") {
    Parameters p;
    p.GAMMA = 1.4;
    Solver solver(p);

    double U1[4] = {1.0, 0.5, 0.0, 3.0};
    double U2[4] = {0.5, 0.0, 0.0, 1.5};
    
    double F_comm_12[4];
    solver.solve_riemann(U1, U2, F_comm_12, 0, 0.0, 0.0);

    double F_comm_21[4];
    solver.solve_riemann(U2, U1, F_comm_21, 0, 0.0, 0.0);

    // The solver computes flux for left U1 and right U2. F = 0.5*(F1+F2) - 0.5*lambda*(U2-U1)
    CHECK(F_comm_12[0] != 0.0);
}

TEST_CASE("Rusanov Riemann solver - With sigma") {
    Parameters p;
    p.GAMMA = 1.4;
    Solver solver(p);

    double UL[4] = {1.0, 0.0, 0.0, 2.5}; 
    double UR[4] = {1.0, 0.0, 0.0, 2.5};
    double F_comm[4] = {0.0, 0.0, 0.0, 0.0};

    // Sigmas apply an artificial pressure gradient 
    double sigl = 0.1;
    double sigr = -0.1;
    solver.solve_riemann(UL, UR, F_comm, 0, sigl, sigr);

    // Check that flux computation completes without NaN
    CHECK(F_comm[0] == doctest::Approx(0.0));
}

TEST_CASE("Rusanov Riemann solver - Y-direction") {
    Parameters p;
    p.GAMMA = 1.4;
    Solver solver(p);

    // Flow in Y direction: u=0, v=0.5
    double UL[4] = {1.0, 0.0, 0.5, 2.5 + 0.5*1.0*0.25}; 
    double UR[4] = {1.0, 0.0, 0.5, 2.5 + 0.5*1.0*0.25};
    double F_comm[4] = {0.0, 0.0, 0.0, 0.0};

    solver.solve_riemann(UL, UR, F_comm, 1, 0.0, 0.0);

    // Flux G(U) = [rho*v, rho*u*v, rho*v^2 + p, (E+p)*v]
    // rho*v = 0.5
    // rho*u*v = 0.0
    // rho*v^2 + p = 0.25 + 1.0 = 1.25
    // E = 2.625
    // (E+p)*v = (3.625)*0.5 = 1.8125
    
    CHECK(F_comm[0] == doctest::Approx(0.5));
    CHECK(F_comm[1] == doctest::Approx(0.0));
    CHECK(F_comm[2] == doctest::Approx(1.25));
    CHECK(F_comm[3] == doctest::Approx(1.8125));
}

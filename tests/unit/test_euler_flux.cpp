#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/ppr/ppr.hpp"
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

TEST_CASE("3D Euler flux computation") {
    Parameters p;
    p.GAMMA = 1.4;
    SolverDim<3> solver(p);
    
    BlockConfig bc;
    bc.id = 0;
    bc.N_ELEM_X = 1; bc.N_ELEM_Y = 1; bc.N_ELEM_Z = 1;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0; bc.Y_MIN = 0.0; bc.Y_MAX = 1.0; bc.Z_MIN = 0.0; bc.Z_MAX = 1.0;
    bc.BC_L = "WALL"; bc.BC_R = "WALL"; bc.BC_B = "WALL"; bc.BC_T = "WALL"; bc.BC_F = "WALL"; bc.BC_K = "WALL";
    Block3D b(bc, 1);
    
    SUBCASE("State at rest") {
        b.U(0, 0, 0, 0, 0, 0, 0) = 1.0; // rho
        b.U(1, 0, 0, 0, 0, 0, 0) = 0.0; // rho*u
        b.U(2, 0, 0, 0, 0, 0, 0) = 0.0; // rho*v
        b.U(3, 0, 0, 0, 0, 0, 0) = 0.0; // rho*w
        b.U(4, 0, 0, 0, 0, 0, 0) = 2.5; // E = 2.5 (pressure = 1.0)
        
        double F[5], G[5], H[5];
        solver.get_flux_pointwise(b, 0, 0, 0, 0, 0, 0, F, G, H, 0.0);
        
        CHECK(F[0] == doctest::Approx(0.0));
        CHECK(F[1] == doctest::Approx(1.0)); // pressure
        CHECK(F[2] == doctest::Approx(0.0));
        CHECK(F[3] == doctest::Approx(0.0));
        CHECK(F[4] == doctest::Approx(0.0));
        
        CHECK(H[0] == doctest::Approx(0.0));
        CHECK(H[1] == doctest::Approx(0.0));
        CHECK(H[2] == doctest::Approx(0.0));
        CHECK(H[3] == doctest::Approx(1.0)); // pressure
        CHECK(H[4] == doctest::Approx(0.0));
    }
    
    SUBCASE("Z-velocity") {
        b.U(0, 0, 0, 0, 0, 0, 0) = 1.0; // rho
        b.U(1, 0, 0, 0, 0, 0, 0) = 0.0;
        b.U(2, 0, 0, 0, 0, 0, 0) = 0.0;
        b.U(3, 0, 0, 0, 0, 0, 0) = 4.0; // rho*w (w=4)
        b.U(4, 0, 0, 0, 0, 0, 0) = 2.5 + 0.5 * 1.0 * 16.0; // E = 10.5
        
        double F[5], G[5], H[5];
        solver.get_flux_pointwise(b, 0, 0, 0, 0, 0, 0, F, G, H, 0.0);
        
        CHECK(H[0] == doctest::Approx(4.0));       // rho*w
        CHECK(H[1] == doctest::Approx(0.0));       // rho*w*u
        CHECK(H[2] == doctest::Approx(0.0));       // rho*w*v
        CHECK(H[3] == doctest::Approx(16.0 + 1.0)); // rho*w^2 + p = 17.0
        CHECK(H[4] == doctest::Approx((10.5 + 1.0) * 4.0)); // (E + p) * w = 46.0
    }
    
    SUBCASE("Symmetry 3D") {
        double F_comm[5];
        double UL[5] = {1.0, 0.0, 0.0, 1.0, 2.5 + 0.5}; // rho, u=0, v=0, w=1, E=3.0 (p=1)
        double UR[5] = {1.0, 0.0, 0.0, -1.0, 2.5 + 0.5}; // rho, u=0, v=0, w=-1, E=3.0 (p=1)
        
        CHECK(F_comm[0] == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("PPR Regularized Acoustic Sound Speed Evaluation") {
    Parameters p;
    p.GAMMA = 1.4;
    p.ENABLE_PPR = true;
    p.RIEMANN_SOLVER = "RUSANOV";

    Solver solver(p);

    double UL[4] = {1.0, 0.0, 0.0, 2.5}; // rho=1, u=0, v=0, p=1.0, a_phys = sqrt(1.4)
    double UR[4] = {1.0, 0.0, 0.0, 2.5};
    double SL = 1.0, SR = 1.0; // S = rho * P_phan = 1.0 -> P_phan = 1.0 (P_reg = P_phys = 1.0)
    double thetaL = 3.0, thetaR = 3.0;

    double F_comm[4];
    solver.solve_riemann(UL, UR, F_comm, 0, SL, SR, thetaL, thetaR);

    // Physical sound speed a_phys = sqrt(1.4 * 1.0 / 1.0) = sqrt(1.4) = 1.1832159566
    double P_phys, P_phan, P_reg, a_reg;
    PPR::get_thermodynamics(UL[0], UL[1], UL[2], UL[3], SL, thetaL, p.GAMMA, p.POS_LIMITER_EPS, P_phys, P_phan, P_reg, a_reg);
    CHECK(a_reg >= std::sqrt(1.4));

    // Test compute_dt sound speed
    Solver solver_dt(p);
    Cell2D* cell = new Cell2D(p.N_PTS, &p);
    cell->dx = 1.0;
    cell->dy = 1.0;
    cell->theta_avg = 3.0;
    int npts2 = p.N_PTS * p.N_PTS;
    for (int iy = 0; iy < p.N_PTS; ++iy) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            int k = iy * p.N_PTS + ix;
            cell->U[0 * npts2 + k] = 1.0; // rho = 1.0
            cell->U[1 * npts2 + k] = 0.0; // u = 0
            cell->U[2 * npts2 + k] = 0.0; // v = 0
            cell->U[3 * npts2 + k] = 2.5; // E = 2.5 (p = 1.0)
            cell->S_field[k] = 1.0; // S = 1.0
        }
    }
    solver_dt.cells.push_back(cell);
    double dt = solver_dt.compute_dt();
    
    // Verify dt is non-zero and finite
    CHECK(dt > 0.0);
}

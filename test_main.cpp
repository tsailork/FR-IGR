#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

#include "parameters.hpp"
#include "basis.hpp"
#include "state.hpp"
#include "solver.hpp"

// Minimal testing framework
int tests_failed = 0;
int tests_passed = 0;

#define TEST_CASE(name) \
    void name()

#define CHECK(cond) \
    if (!(cond)) { \
        std::cout << "FAILED" << std::endl; \
        std::cerr << "Check failed: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        tests_failed++; \
        return; \
    }

#define CHECK_CLOSE(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cout << "FAILED" << std::endl; \
        std::cerr << "Check failed: " #a " (" << (a) << ") != " #b " (" << (b) << ") within tol " << (tol) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        tests_failed++; \
        return; \
    }

#define RUN_TEST(name) \
    std::cout << "Running test: " << #name << "... "; \
    { \
        int initial_failed = tests_failed; \
        name(); \
        if (initial_failed == tests_failed) { \
            std::cout << "PASSED" << std::endl; \
            tests_passed++; \
        } \
    }

// ----------------------------------------------------------------------------
// 1. BASIS TESTS
// ----------------------------------------------------------------------------

TEST_CASE(test_basis_p0) {
    Parameters p;
    p.P_DEG = 0;
    p.N_PTS = 1;
    Basis b(p);

    CHECK(b.z.size() == 1);
    CHECK_CLOSE(b.z[0], 0.0, 1e-12);
    
    CHECK(b.l_L.size() == 1);
    CHECK_CLOSE(b.l_L[0], 1.0, 1e-12);
    CHECK_CLOSE(b.l_R[0], 1.0, 1e-12);

    CHECK(b.D.size() == 1);
    CHECK_CLOSE(b.D[0][0], 0.0, 1e-12);

    // Correction derivatives for P=0
    // dgl = -0.5, dgr = 0.5
    CHECK_CLOSE(b.dgl[0], -0.5, 1e-12);
    CHECK_CLOSE(b.dgr[0], 0.5, 1e-12);
}

TEST_CASE(test_basis_p1) {
    Parameters p;
    p.P_DEG = 1;
    p.N_PTS = 2;
    Basis b(p);

    CHECK(b.z.size() == 2);
    double z_expected = 1.0 / std::sqrt(3.0);
    CHECK_CLOSE(b.z[0], -z_expected, 1e-12);
    CHECK_CLOSE(b.z[1], z_expected, 1e-12);

    // Derivative matrix for P=1
    // D = [[-sqrt(3)/2, sqrt(3)/2], [-sqrt(3)/2, sqrt(3)/2]]
    // Wait, D_ij = l'_j(z_i)
    // l_0(x) = (x - z1) / (z0 - z1) => l_0' = 1 / (z0 - z1) = 1 / (-2/sqrt(3)) = -sqrt(3)/2
    // l_1(x) = (x - z0) / (z1 - z0) => l_1' = 1 / (z1 - z0) = 1 / (2/sqrt(3)) = sqrt(3)/2
    double D_val = std::sqrt(3.0) / 2.0;
    CHECK_CLOSE(b.D[0][0], -D_val, 1e-12);
    CHECK_CLOSE(b.D[0][1], D_val, 1e-12);
    CHECK_CLOSE(b.D[1][0], -D_val, 1e-12);
    CHECK_CLOSE(b.D[1][1], D_val, 1e-12);
}

// ----------------------------------------------------------------------------
// 2. STATE TESTS
// ----------------------------------------------------------------------------

TEST_CASE(test_state_indexing) {
    Parameters p;
    p.N_ELEM_X = 2;
    p.N_ELEM_Y = 2;
    p.P_DEG = 1;
    p.N_PTS = 2;
    State s(p);

    // [Var][Ey][Ex][Iy][Ix]
    // Total size = 4 * 2 * 2 * 2 * 2 = 64
    CHECK(s.data.size() == 64);

    s(0, 0, 0, 0, 0) = 1.0;
    s(3, 1, 1, 1, 1) = 42.0;

    CHECK_CLOSE(s(0, 0, 0, 0, 0), 1.0, 1e-12);
    CHECK_CLOSE(s(3, 1, 1, 1, 1), 42.0, 1e-12);
    
    // Check that they don't overlap
    CHECK_CLOSE(s(0, 1, 1, 1, 1), 0.0, 1e-12);
}

// ----------------------------------------------------------------------------
// 3. SOLVER TESTS
// ----------------------------------------------------------------------------

TEST_CASE(test_thomas_algorithm) {
    // Solve Ax = d
    // [ 2  -1   0 ] [ x0 ]   [ 1 ]
    // [ -1  2  -1 ] [ x1 ] = [ 0 ]
    // [ 0  -1   2 ] [ x2 ]   [ 1 ]
    // Solution: x = [ 1, 1, 1 ]
    std::vector<double> a = {0, -1, -1};
    std::vector<double> b = {2, 2, 2};
    std::vector<double> c = {-1, -1, 0};
    std::vector<double> d = {1, 0, 1};
    std::vector<double> x(3);

    solve_tridiagonal(a, b, c, d, x);

    CHECK_CLOSE(x[0], 1.0, 1e-12);
    CHECK_CLOSE(x[1], 1.0, 1e-12);
    CHECK_CLOSE(x[2], 1.0, 1e-12);
}

TEST_CASE(test_riemann_rusanov) {
    Parameters p;
    Solver solver(p);
    
    double UL[4] = {1.0, 0.0, 0.0, 2.5};    // rho=1, u=0, v=0, E=2.5 => p=1
    double UR[4] = {0.125, 0.0, 0.0, 0.25}; // rho=0.125, u=0, v=0, E=0.25 => p=0.1
    double F_comm[4];
    
    // dir=0 (X-direction)
    solver.solve_riemann(UL, UR, F_comm, 0, 0.0, 0.0);
    
    // Expected:
    // pL = 1.0, cL = sqrt(1.4*1/1) = 1.1832159566199232
    // pR = 0.1, cR = sqrt(1.4*0.1/0.125) = 1.0583005244258363
    // max_wave = 1.1832159566199232
    // FL = [0, 1, 0, 0]
    // FR = [0, 0.1, 0, 0]
    // F_comm[0] = 0.5*(0+0) - 0.5*max_wave*(0.125 - 1.0) = 0.5 * 1.1832159566199232 * 0.875 = 0.5176569810212164
    // F_comm[1] = 0.5*(1+0.1) - 0.5*max_wave*(0 - 0) = 0.55
    
    CHECK_CLOSE(F_comm[0], 0.5176569810212164, 1e-10);
    CHECK_CLOSE(F_comm[1], 0.55, 1e-10);
    CHECK_CLOSE(F_comm[2], 0.0, 1e-10);
    CHECK_CLOSE(F_comm[3], 1.3311179511974136, 1e-10);
}

TEST_CASE(test_rhs_uniform) {
    Parameters p;
    p.N_ELEM_X = 5;
    p.N_ELEM_Y = 5;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.ENABLE_IGR = false; // Disable IGR for this simple test
    
    Solver solver(p);
    
    // Set uniform state
    for(int ey=0; ey<p.N_ELEM_Y; ++ey) {
        for(int ex=0; ex<p.N_ELEM_X; ++ex) {
            for(int iy=0; iy<p.N_PTS; ++iy) {
                for(int ix=0; ix<p.N_PTS; ++ix) {
                    solver.U(0, ey, ex, iy, ix) = 1.0;
                    solver.U(1, ey, ex, iy, ix) = 0.1;
                    solver.U(2, ey, ex, iy, ix) = 0.2;
                    solver.U(3, ey, ex, iy, ix) = 2.5;
                }
            }
        }
    }
    
    solver.compute_rhs();
    
    // RHS should be zero for uniform state
    for(size_t i=0; i<solver.RHS.data.size(); ++i) {
        CHECK_CLOSE(solver.RHS.data[i], 0.0, 1e-12);
    }
}

TEST_CASE(test_helmholtz_1d) {
    Parameters p;
    p.N_ELEM_X = 10;
    p.N_ELEM_Y = 1;
    p.P_DEG = 1; 
    p.N_PTS = 2;
    p.ENABLE_IGR = true;
    p.ALPHA_SCALE = 0.1;
    p.X_MIN = 0.0;
    p.X_MAX = 1.0;
    p.GAMMA = 1.4;
    
    Solver solver(p);
    
    // Set density = 1.0, velocity u = -100*x to trigger sensor
    double dx_elem = (p.X_MAX - p.X_MIN) / p.N_ELEM_X;
    for(int ex=0; ex<p.N_ELEM_X; ++ex) {
        for(int ix=0; ix<p.N_PTS; ++ix) {
            double x_local = solver.basis.z[ix] * 0.5 * dx_elem + (ex + 0.5) * dx_elem;
            solver.U(0, 0, ex, 0, ix) = 1.0; // rho
            solver.U(1, 0, ex, 0, ix) = -100.0 * x_local; // rhou
            solver.U(2, 0, ex, 0, ix) = 0.0; // rhov
            solver.U(3, 0, ex, 0, ix) = 10000.0; // E (large enough for stability)
        }
    }
    
    solver.compute_entropic_pressure();
    
    // Sigma field should be non-zero
    bool non_zero_sigma = false;
    for(size_t i=0; i<solver.sigma_field.size(); ++i) {
        if (solver.sigma_field[i] > 1e-8) non_zero_sigma = true;
    }
    CHECK(non_zero_sigma);
}

int main() {
    std::cout << "Running unit tests..." << std::endl;
    
    RUN_TEST(test_basis_p0);
    RUN_TEST(test_basis_p1);
    RUN_TEST(test_state_indexing);
    RUN_TEST(test_thomas_algorithm);
    RUN_TEST(test_helmholtz_1d);
    RUN_TEST(test_riemann_rusanov);
    RUN_TEST(test_rhs_uniform);

    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    return (tests_failed == 0) ? 0 : 1;
}

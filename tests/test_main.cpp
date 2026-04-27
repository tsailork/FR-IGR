#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

#include "../src/core/parameters.hpp"
#include "../src/core/basis.hpp"
#include "../src/core/state.hpp"
#include "../src/core/solver.hpp"
#include "../src/igr/adi_solver.hpp"

// Minimal testing framework
int tests_failed = 0;
int tests_passed = 0;

#define TEST_CASE(name) void name()

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

    CHECK(s.data.size() == 64);

    s(0, 0, 0, 0, 0) = 1.0;
    s(3, 1, 1, 1, 1) = 42.0;

    CHECK_CLOSE(s(0, 0, 0, 0, 0), 1.0, 1e-12);
    CHECK_CLOSE(s(3, 1, 1, 1, 1), 42.0, 1e-12);
    CHECK_CLOSE(s(0, 1, 1, 1, 1), 0.0, 1e-12);
}

// ----------------------------------------------------------------------------
// 3. SOLVER TESTS
// ----------------------------------------------------------------------------

TEST_CASE(test_thomas_algorithm) {
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
    
    double UL[4] = {1.0, 0.0, 0.0, 2.5};
    double UR[4] = {0.125, 0.0, 0.0, 0.25};
    double F_comm[4];
    
    solver.solve_riemann(UL, UR, F_comm, 0, 0.0, 0.0);
    
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
    p.ENABLE_IGR = false;
    
    Solver solver(p);
    
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
    
    double dx_elem = (p.X_MAX - p.X_MIN) / p.N_ELEM_X;
    for(int ex=0; ex<p.N_ELEM_X; ++ex) {
        for(int ix=0; ix<p.N_PTS; ++ix) {
            double x_local = solver.basis.z[ix] * 0.5 * dx_elem + (ex + 0.5) * dx_elem;
            solver.U(0, 0, ex, 0, ix) = 1.0;
            solver.U(1, 0, ex, 0, ix) = -100.0 * x_local;
            solver.U(2, 0, ex, 0, ix) = 0.0;
            solver.U(3, 0, ex, 0, ix) = 10000.0;
        }
    }
    
    solver.compute_entropic_pressure();
    
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

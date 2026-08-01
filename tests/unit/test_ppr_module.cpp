#include "../doctest.h"
#include "../../src/ppr/ppr.hpp"
#include "../../src/core/parameters.hpp"
#include "../../src/core/basis.hpp"
#include "../../src/core/cell.hpp"
#include <cmath>
#include <vector>

TEST_CASE("PPR - Thermodynamics & Regularized Wave Speed") {
    double rho = 1.0;
    double rhou = 2.0; // u = 2.0
    double rhov = 0.0;
    double E = 10.0;
    double S = 1.0; // P_phan = 1.0
    double gamma = 1.4;
    double eps = 1e-12;

    // P_phys = 0.4 * (10.0 - 0.5 * 1.0 * 4.0) = 0.4 * 8.0 = 3.2
    double P_phys, P_phan, P_reg, a_reg;

    SUBCASE("Zero Theta (Equilibrium)") {
        double theta = 0.0;
        PPR::get_thermodynamics(rho, rhou, rhov, E, S, theta, gamma, eps, P_phys, P_phan, P_reg, a_reg);

        CHECK(P_phys == doctest::Approx(3.2));
        CHECK(P_phan == doctest::Approx(1.0));
        CHECK(P_reg == doctest::Approx(3.2));
        CHECK(a_reg == doctest::Approx(std::sqrt(1.4 * 3.2 / 1.0)));
    }

    SUBCASE("Positive Theta (Non-Equilibrium Regularization)") {
        double theta = 1.0;
        PPR::get_thermodynamics(rho, rhou, rhov, E, S, theta, gamma, eps, P_phys, P_phan, P_reg, a_reg);

        // P_reg = 3.2 + 1.0 * (3.2 - 1.0) = 5.4
        CHECK(P_reg == doctest::Approx(5.4));
        CHECK(a_reg > std::sqrt(1.4 * 3.2 / 1.0));
    }
}

TEST_CASE("PPR - Analytical Relaxation & Limiter Execution") {
    Parameters p;
    p.ENABLE_PPR = true;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.POS_LIMITER_EPS = 1e-12;
    p.PPR_C_TAU = 0.25;
    p.PPR_N_CELLS_SHOCK = 2.5;

    Basis basis(p.P_DEG);
    CellDim<2> cell(p.N_PTS, &p);
    cell.dx = 0.1;
    cell.dy = 0.1;

    // Initialize state
    for (int iy = 0; iy < p.N_PTS; ++iy) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            cell.get_U(0, iy, ix, p.N_PTS) = 1.0;
            cell.get_U(1, iy, ix, p.N_PTS) = 0.0;
            cell.get_U(2, iy, ix, p.N_PTS) = 0.0;
            cell.get_U(3, iy, ix, p.N_PTS) = 2.5; // P_phys = 1.0
            cell.S_field[iy * p.N_PTS + ix] = 2.0; // P_phan = 2.0 (out of equilibrium)
        }
    }

    SUBCASE("Analytical Relaxation Step") {
        double dt_stage = 0.01;
        PPR::relax_phantom_pressure_2d(cell, dt_stage, basis, p);

        // S should decay towards S_eq = rho * P_phys = 1.0 * 1.0 = 1.0
        for (int k = 0; k < 4; ++k) {
            CHECK(cell.S_field[k] < 2.0);
            CHECK(cell.S_field[k] > 1.0);
        }
    }

    SUBCASE("Phantom Pressure Bound Limiter") {
        std::vector<CellDim<2>*> cells = { &cell };
        PPR::apply_phantom_pressure_limiter_2d(cells, basis, p);

        // P_phan should remain clipped within bounds
        for (int k = 0; k < 4; ++k) {
            CHECK(cell.S_field[k] >= 0.0);
        }
    }
}

TEST_CASE("PPR - Multidimensional Feature Switches") {
    Parameters p;
    p.ENABLE_PPR = true;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.POS_LIMITER_EPS = 1e-12;
    p.PPR_C_TAU = 0.25;
    p.PPR_N_CELLS_SHOCK = 2.5;

    Basis basis(p.P_DEG);
    CellDim<2> cell(p.N_PTS, &p);
    cell.dx = 0.1;
    cell.dy = 0.1;

    for (int iy = 0; iy < p.N_PTS; ++iy) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            cell.get_U(0, iy, ix, p.N_PTS) = 1.0;
            cell.get_U(1, iy, ix, p.N_PTS) = -2.0 * (ix + 1); // Compression along X
            cell.get_U(2, iy, ix, p.N_PTS) = 0.0;
            cell.get_U(3, iy, ix, p.N_PTS) = 10.0;
            cell.S_field[iy * p.N_PTS + ix] = 1.0;
        }
    }

    std::vector<CellDim<2>*> cells = { &cell };

    SUBCASE("Shock-Normal Mach vs Total Mach Switch") {
        p.PPR_USE_SHOCK_NORMAL_MACH = true;
        PPR::compute_element_theta_2d(cells, basis, p);
        double theta_normal = cell.theta_avg;

        p.PPR_USE_SHOCK_NORMAL_MACH = false;
        PPR::compute_element_theta_2d(cells, basis, p);
        double theta_total = cell.theta_avg;

        CHECK(theta_normal >= 0.0);
        CHECK(theta_total >= 0.0);
    }

    SUBCASE("Soft-Max Indicator vs Node-Max Switch") {
        p.PPR_USE_SOFTMAX_INDICATOR = true;
        p.PPR_SOFTMAX_P = 4.0;
        PPR::compute_element_theta_2d(cells, basis, p);
        double theta_softmax = cell.theta_avg;

        p.PPR_USE_SOFTMAX_INDICATOR = false;
        PPR::compute_element_theta_2d(cells, basis, p);
        double theta_nodemax = cell.theta_avg;

        CHECK(theta_softmax >= 0.0);
        CHECK(theta_nodemax >= 0.0);
    }

    SUBCASE("Sub-cell Linear Theta Representation Switch") {
        p.PPR_USE_SUBCELL_LINEAR_THETA = true;
        PPR::compute_element_theta_2d(cells, basis, p);
        CHECK(cell.theta_avg >= 0.0);

        p.PPR_USE_SUBCELL_LINEAR_THETA = false;
        PPR::compute_element_theta_2d(cells, basis, p);
        CHECK(cell.theta_ax == 0.0);
        CHECK(cell.theta_ay == 0.0);
    }

    SUBCASE("Dynamic C_tau Guide Rule Switch") {
        p.PPR_USE_DYNAMIC_C_TAU = true;
        PPR::compute_element_theta_2d(cells, basis, p);
        CHECK(cell.theta_avg >= 0.0);

        p.PPR_USE_DYNAMIC_C_TAU = false;
        PPR::compute_element_theta_2d(cells, basis, p);
        CHECK(cell.theta_avg >= 0.0);
    }
}

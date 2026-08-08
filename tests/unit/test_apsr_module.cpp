#include <array>
#include "../doctest.h"
#include "../../src/apsr/apsr.hpp"
#include "../../src/core/parameters.hpp"
#include "../../src/core/basis.hpp"
#include "../../src/core/cell.hpp"
#include <cmath>
#include <vector>

TEST_CASE("APSR - Non-Negative SmoothReLU & Ducros Filter") {
    SUBCASE("SmoothReLU Non-Negativity") {
        double eps = 1e-12;
        CHECK(APSR::smooth_relu(-10.0, eps) >= 0.0);
        CHECK(APSR::smooth_relu(-1e-5, eps) >= 0.0);
        CHECK(APSR::smooth_relu(0.0, eps) == doctest::Approx(0.5 * eps));
        CHECK(APSR::smooth_relu(10.0, eps) == doctest::Approx(10.0));
    }

    SUBCASE("Ducros Vorticity Filter") {
        // Pure compressional shock: div_u = -100.0, curl_u_sq = 0.0 -> filter = 1.0
        double f_shock = APSR::compute_ducros_filter(-100.0, 0.0);
        CHECK(f_shock == doctest::Approx(1.0));

        // Pure shear / vortex: div_u = 0.0, curl_u_sq = 1000.0 -> filter = 0.0
        double f_vort = APSR::compute_ducros_filter(0.0, 1000.0);
        CHECK(f_vort == doctest::Approx(0.0));
    }
}

TEST_CASE("APSR - Thermodynamics & Regularized Whitham Wave Speeds") {
    Parameters p;
    p.ENABLE_PPR = true;
    p.ENABLE_APSR = true;
    p.GAMMA = 1.4;
    p.POS_LIMITER_EPS = 1e-12;
    p.PPR_CONSTANT_THETA = 1.0;

    std::array<double, 4> U = {1.0, 2.0, 0.0, 10.0}; // rho=1.0, u=2.0, v=0.0, E=10.0 -> P_phys = 3.2
    double S_val = 1.0; // P_phan = 1.0

    double P_phys, P_phan, P_reg, a_reg;
    APSR::get_thermodynamics_apsr_2d(U.data(), S_val, p, P_phys, P_phan, P_reg, a_reg);

    CHECK(P_phys == doctest::Approx(3.2));
    CHECK(P_phan == doctest::Approx(1.0));
    CHECK(P_reg == doctest::Approx(5.4));
    CHECK(a_reg == doctest::Approx(std::sqrt(10.72)));
}

TEST_CASE("APSR - 1D Shock Width Theta Calculation") {
    Parameters p;
    p.ENABLE_PPR = true;
    p.ENABLE_APSR = true;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.PPR_C_TAU = 0.25;
    p.PPR_N_CELLS_SHOCK = 2.5;

    Basis basis(p.P_DEG);
    CellDim<2> cell(p.N_PTS, &p);
    cell.dx = 0.1;
    cell.dy = 0.1;

    cell.U.resize(4 * p.N_PTS * p.N_PTS, 0.0);
    cell.S_field.resize(p.N_PTS * p.N_PTS, 1.0);

    for (int iy = 0; iy < p.N_PTS; ++iy) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            int idx_base = iy * p.N_PTS + ix;
            cell.U[0 * p.N_PTS * p.N_PTS + idx_base] = 1.0;
            cell.U[1 * p.N_PTS * p.N_PTS + idx_base] = 2.0;
            cell.U[2 * p.N_PTS * p.N_PTS + idx_base] = 0.0;
            cell.U[3 * p.N_PTS * p.N_PTS + idx_base] = 10.0;
            cell.S_field[idx_base] = 1.0;
        }
    }

    std::vector<CellDim<2>*> cells = { &cell };
    APSR::compute_element_theta_apsr_2d(cells, basis, p);

    CHECK(cell.theta_avg >= 0.0);
}

TEST_CASE("APSR - Simple Divergence Theta Sensor") {
    Parameters p;
    p.ENABLE_PPR = true;
    p.ENABLE_APSR = true;
    p.P_DEG = 1; // P = 1 => P + 1 = 2
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    p.PPR_C_TAU = 0.25;
    p.PPR_N_CELLS_SHOCK = 2.0; // theta_max = (2.0 * 2.0) / (4.0 * 0.25) = 4.0
    p.PPR_USE_DUCROS = false;

    // Expected theta_max = 2.0 * (1 + 1) / (4.0 * 0.25) = 4.0
    double expected_theta_max = 4.0;
    double C_tau_0 = 0.25;
    double theta_max_calc = (p.PPR_N_CELLS_SHOCK * (p.P_DEG + 1)) / (4.0 * C_tau_0);
    CHECK(theta_max_calc == doctest::Approx(expected_theta_max));

    Basis basis(p.P_DEG);
    CellDim<2> cell(p.N_PTS, &p);
    cell.dx = 0.1;
    cell.dy = 0.1;
    cell.U.resize(4 * p.N_PTS * p.N_PTS, 0.0);
    cell.S_field.resize(p.N_PTS * p.N_PTS, 1.0);

    for (int iy = 0; iy < p.N_PTS; ++iy) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            int idx_base = iy * p.N_PTS + ix;
            cell.U[0 * p.N_PTS * p.N_PTS + idx_base] = 1.0;
            cell.U[1 * p.N_PTS * p.N_PTS + idx_base] = 2.0;
            cell.U[2 * p.N_PTS * p.N_PTS + idx_base] = 0.0;
            cell.U[3 * p.N_PTS * p.N_PTS + idx_base] = 10.0;
            cell.S_field[idx_base] = 1.0;
        }
    }

    std::vector<CellDim<2>*> cells = { &cell };
    APSR::compute_element_theta_apsr_2d(cells, basis, p);

    CHECK(cell.theta_avg >= 0.0);
    CHECK(cell.theta_avg <= expected_theta_max);
}

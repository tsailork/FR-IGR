/**
 * @file test_analytical_jacobians.cpp
 * @brief Unit test verifying exact analytical physical Euler Jacobians against central finite differences.
 */

#include "../doctest.h"
#include "../../src/time/implicit_precond.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

TEST_CASE("Analytical Euler Flux Jacobians vs Finite Differences") {
    double gamma = 1.4;
    double U[4] = {1.0, 0.5, -0.2, 2.5}; // Physical state: rho=1, u=0.5, v=-0.2, P > 0

    // Compute Analytical Jacobians
    double A_ana[16], B_ana[16];
    fr::implicit::compute_euler_jacobian_x_2d(U, gamma, A_ana);
    fr::implicit::compute_euler_jacobian_y_2d(U, gamma, B_ana);

    // Compute Fluxes helper
    auto euler_flux_x = [gamma](const double u_vec[4], double F[4]) {
        double r = std::max(u_vec[0], 1e-12);
        double u = u_vec[1] / r;
        double v = u_vec[2] / r;
        double E = u_vec[3];
        double P = std::max((gamma - 1.0) * (E - 0.5 * r * (u * u + v * v)), 1e-12);
        F[0] = u_vec[1];
        F[1] = u_vec[1] * u + P;
        F[2] = u_vec[1] * v;
        F[3] = (E + P) * u;
    };

    auto euler_flux_y = [gamma](const double u_vec[4], double G[4]) {
        double r = std::max(u_vec[0], 1e-12);
        double u = u_vec[1] / r;
        double v = u_vec[2] / r;
        double E = u_vec[3];
        double P = std::max((gamma - 1.0) * (E - 0.5 * r * (u * u + v * v)), 1e-12);
        G[0] = u_vec[2];
        G[1] = u_vec[1] * v;
        G[2] = u_vec[2] * v + P;
        G[3] = (E + P) * v;
    };

    double eps = 1e-7;
    double A_fd[16], B_fd[16];

    for (int col = 0; col < 4; ++col) {
        double U_plus[4], U_minus[4];
        for (int v = 0; v < 4; ++v) {
            U_plus[v] = U[v];
            U_minus[v] = U[v];
        }
        U_plus[col] += eps;
        U_minus[col] -= eps;

        double F_plus[4], F_minus[4];
        euler_flux_x(U_plus, F_plus);
        euler_flux_x(U_minus, F_minus);

        double G_plus[4], G_minus[4];
        euler_flux_y(U_plus, G_plus);
        euler_flux_y(U_minus, G_minus);

        for (int row = 0; row < 4; ++row) {
            A_fd[row * 4 + col] = (F_plus[row] - F_minus[row]) / (2.0 * eps);
            B_fd[row * 4 + col] = (G_plus[row] - G_minus[row]) / (2.0 * eps);
        }
    }

    // Verify error norm
    double max_err_A = 0.0;
    double max_err_B = 0.0;
    for (int i = 0; i < 16; ++i) {
        max_err_A = std::max(max_err_A, std::abs(A_ana[i] - A_fd[i]));
        max_err_B = std::max(max_err_B, std::abs(B_ana[i] - B_fd[i]));
    }

    CHECK(max_err_A < 1e-6);
    CHECK(max_err_B < 1e-6);
}

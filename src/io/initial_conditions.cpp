/**
 * @file initial_conditions.cpp
 * @brief Initial condition setup implementation.
 *
 * Implements standard fluid dynamics benchmark initializations such as 2D Riemann problems,
 * blast waves, sine waves, and freestream uniform flow.
 *
 * @see IC::apply
 */

#include "initial_conditions.hpp"
#include "../core/solver.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double IC::sigmoid(double x, double x0, double delta) {
    return 1.0 / (1.0 + std::exp(-(x - x0) / delta));
}

void IC::apply(Solver& solver) {
    const Parameters& p = solver.p;
    const Basis& basis = solver.basis;

    for (Cell* c : solver.cells) {
        // Do not overwrite elements that are fully inside the IB
        if (p.ENABLE_IB && c->solid_mask) continue;

        double delta = 0.5 * std::min(c->dx, c->dy);

        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double x = c->x_min + 0.5 * (1.0 + basis.z[ix]) * c->dx;
                double y = c->y_min + 0.5 * (1.0 + basis.z[iy]) * c->dy;

                double rho, u, v, press;

                if (p.IC_TYPE == "RIEMANN_2D_C3") {
                    double x0 = 0.5, y0 = 0.5;
                    double rTR = 1.5, uTR = 0.0, vTR = 0.0, pTR = 1.5;
                    double rTL = 0.5323, uTL = 1.206, vTL = 0.0, pTL = 0.3;
                    double rBL = 0.138, uBL = 1.206, vBL = 1.206, pBL = 0.029;
                    double rBR = 0.5323, uBR = 0.0, vBR = 1.206, pBR = 0.3;

                    double wx = sigmoid(x, x0, delta);
                    double wy = sigmoid(y, y0, delta);

                    rho = (1 - wy) * ((1 - wx) * rBL + wx * rBR) +
                          wy * ((1 - wx) * rTL + wx * rTR);
                    u = (1 - wy) * ((1 - wx) * uBL + wx * uBR) +
                        wy * ((1 - wx) * uTL + wx * uTR);
                    v = (1 - wy) * ((1 - wx) * vBL + wx * vBR) +
                        wy * ((1 - wx) * vTL + wx * vTR);
                    press = (1 - wy) * ((1 - wx) * pBL + wx * pBR) +
                            wy * ((1 - wx) * pTL + wx * pTR);

                } else if (p.IC_TYPE == "BLAST") {
                    double r0 = std::sqrt(x * x + y * y);
                    rho = 1.0;
                    u = 0.0;
                    v = 0.0;
                    double w = 1.0 - sigmoid(r0, 0.1, delta);
                    press = 0.1 + w * 0.9;

                } else if (p.IC_TYPE == "SINE_WAVE") {
                    rho = 1.0 + 0.2 * std::sin(2.0 * M_PI * x);
                    u = 1.0;
                    v = 0.0;
                    press = 1.0;

                } else if (p.IC_TYPE == "FREESTREAM") {
                    rho = p.RHO_INF;
                    u = p.U_INF;
                    v = p.V_INF;
                    press = p.P_INF;

                } else if (p.IC_TYPE == "KELVIN_HELMHOLTZ" || p.IC_TYPE == "KHI") {
                    // 2D Kelvin-Helmholtz Shear Layer Instability
                    double y1 = 0.25, y2 = 0.75;
                    double sigma = 0.025;
                    double U0 = 0.5;
                    double eps = 0.05;

                    double k_wave = 3.0; // 3 disturbance cycles across domain x in [0, 1]
                    u = U0 * (std::tanh((y - y1) / sigma) - std::tanh((y - y2) / sigma) - 1.0);
                    v = eps * U0 * std::sin(2.0 * M_PI * k_wave * x) * 
                        (std::exp(-std::pow((y - y1) / sigma, 2)) + std::exp(-std::pow((y - y2) / sigma, 2)));
                    rho = 1.0 + 0.5 * (std::tanh((y - y1) / sigma) - std::tanh((y - y2) / sigma));
                    press = 2.5;

                } else if (p.IC_TYPE == "SHOCK_VORTEX") {
                    // 2D Shock-Vortex Interaction (Inoue & Hattori benchmark)
                    // Shock at xs = 0.5 moving right toward vortex at (xv, yv) = (1.5, 0.0)
                    double xs = 0.5;            // Shock initial position
                    double xv = 1.5, yv = 0.0;  // Vortex center position
                    double Mv = 0.5;            // Vortex Mach number
                    double rc = 0.2;            // Core radius scale
                    double Ms = 1.2;            // Shock Mach number
                    double gamma = p.GAMMA;

                    double rx = x - xv;
                    double ry = y - yv;
                    double r = std::sqrt(rx * rx + ry * ry);
                    double r_norm = r / rc;
                    double v_theta = Mv * r_norm * std::exp(0.5 * (1.0 - r_norm * r_norm));

                    double u_vort = (r > 1e-12) ? -v_theta * (ry / r) : 0.0;
                    double v_vort = (r > 1e-12) ?  v_theta * (rx / r) : 0.0;
                    double T_r = 1.0 - 0.5 * (gamma - 1.0) * Mv * Mv * std::exp(1.0 - r_norm * r_norm);
                    if (T_r < 1e-4) T_r = 1e-4;

                    double rho_R = std::pow(T_r, 1.0 / (gamma - 1.0));
                    double p_R = (1.0 / gamma) * std::pow(T_r, gamma / (gamma - 1.0));

                    // Rankine-Hugoniot post-shock state for Ms = 1.2 moving in +x direction
                    double p0 = 1.0 / gamma;
                    double a0 = 1.0; // sqrt(gamma * p0 / rho0) = 1.0
                    double rho_L = 1.0 * ((gamma + 1.0) * Ms * Ms) / ((gamma - 1.0) * Ms * Ms + 2.0);
                    double p_L = p0 * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (Ms * Ms - 1.0));
                    double u_L = (2.0 * a0 / (gamma + 1.0)) * (Ms - 1.0 / Ms);

                    double w = sigmoid(x, xs, delta);
                    rho   = (1.0 - w) * rho_L + w * rho_R;
                    u     = (1.0 - w) * u_L   + w * u_vort;
                    v     = (1.0 - w) * 0.0   + w * v_vort;
                    press = (1.0 - w) * p_L   + w * p_R;

                } else if (p.IC_TYPE == "DECAYING_TURBULENCE") {
                    // 2D Decaying Isotropic Solenoidal Compressible Turbulence
                    double x_tilde = 2.0 * M_PI * x;
                    double y_tilde = 2.0 * M_PI * y;
                    double u_sum = 0.0, v_sum = 0.0;
                    double kp = 4.0;
                    double U0 = 0.2; // Solenoidal turbulent Mach scale ~0.4

                    for (int kx = 1; kx <= 6; ++kx) {
                        for (int ky = 1; ky <= 6; ++ky) {
                            double k = std::sqrt((double)(kx * kx + ky * ky));
                            double A_k = (k / kp) * (k / kp) * std::exp(1.0 - (k / kp) * (k / kp));
                            double phi_x = std::sin(kx * 12.345 + ky * 67.89);
                            double phi_y = std::cos(kx * 54.321 + ky * 98.76);

                            u_sum += -A_k * (ky / k) * std::sin(kx * x_tilde + phi_x) * std::cos(ky * y_tilde + phi_y);
                            v_sum +=  A_k * (kx / k) * std::cos(kx * x_tilde + phi_x) * std::sin(ky * y_tilde + phi_y);
                        }
                    }
                    rho = 1.0;
                    u = U0 * u_sum;
                    v = U0 * v_sum;
                    press = 1.0 / p.GAMMA;

                } else if (p.IC_TYPE == "RICHTMYER_MESHKOV" || p.IC_TYPE == "RMI") {
                    // 2D Richtmyer-Meshkov Instability
                    double xs = 0.2;       // Incident shock position
                    double x0 = 0.5;       // Mean density interface position
                    double amp = 0.05;     // Perturbation amplitude
                    double Ly = 1.0;       // Domain height / wavelength
                    double sigma_i = 0.01; // Interface smoothing thickness
                    double Ms = 1.5;       // Shock Mach number
                    double gamma = p.GAMMA;

                    double xi = x0 + amp * std::sin(2.0 * M_PI * y / Ly);
                    double rho1 = 1.0;     // Light fluid
                    double rho2 = 3.0;     // Heavy fluid (Atwood number = 0.5)

                    double rho_unshocked = rho1 + 0.5 * (rho2 - rho1) * (1.0 + std::tanh((x - xi) / sigma_i));
                    double p_unshocked = 1.0 / gamma;

                    double a1 = 1.0;
                    double rho_L = rho1 * ((gamma + 1.0) * Ms * Ms) / ((gamma - 1.0) * Ms * Ms + 2.0);
                    double p_L = p_unshocked * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (Ms * Ms - 1.0));
                    double u_L = (2.0 * a1 / (gamma + 1.0)) * (Ms - 1.0 / Ms);

                    double w = sigmoid(x, xs, delta);
                    rho   = (1.0 - w) * rho_L + w * rho_unshocked;
                    u     = (1.0 - w) * u_L   + w * 0.0;
                    v     = 0.0;
                    press = (1.0 - w) * p_L   + w * p_unshocked;

                } else if (p.IC_TYPE == "ORSZAG_TANG") {
                    // 2D Compressible Hydrodynamic Orszag-Tang Vortex
                    double xt = 2.0 * M_PI * x;
                    double yt = 2.0 * M_PI * y;
                    double gamma = p.GAMMA;

                    rho = gamma * gamma;
                    u = -std::sin(yt);
                    v =  std::sin(xt);
                    press = gamma + 0.25 * gamma * (std::cos(2.0 * xt) + 2.0 * std::cos(yt));

                } else if (p.IC_TYPE == "LID_DRIVEN_CAVITY") {
                    rho = p.RHO_INF;
                    u = 0.0;
                    v = 0.0;
                    press = p.P_INF;

                } else { // UNIFORM default
                    rho = 1.0;
                    u = 0.0;
                    v = 0.0;
                    press = 1.0;
                }

                int npts = p.N_PTS;
                c->get_U(0, iy, ix, npts) = rho;
                c->get_U(1, iy, ix, npts) = rho * u;
                c->get_U(2, iy, ix, npts) = rho * v;
                c->get_U(3, iy, ix, npts) = press / (p.GAMMA - 1.0) + 0.5 * rho * (u * u + v * v);
                c->S_field[iy * npts + ix] = rho * press;
            }
        }
    }
}

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

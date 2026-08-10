#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/time/esdirk34.hpp"
#include "../../src/limiters/limiter_smooth.hpp"
#include <cmath>

TEST_SUITE("Implicit ESDIRK34 Solver") {
    TEST_CASE("ESDIRK34 - Physical State Preservation") {
        auto p = make_params(2, 2, 2);
        p.TIME_INTEGRATOR = "ESDIRK34";
        p.IMPLICIT_CFL = 10.0;
        p.IMPLICIT_NEWTON_TOL = 1e-5;
        p.IMPLICIT_GMRES_TOL = 1e-4;

        Solver solver(p);
        Basis basis(p.P_DEG);

        // Initialize uniform flow
        int npts = p.N_PTS;
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int idx = iy * npts + ix;
                    c->U[0 * npts * npts + idx] = 1.0;
                    c->U[1 * npts * npts + idx] = 0.5;
                    c->U[2 * npts * npts + idx] = 0.0;
                    c->U[3 * npts * npts + idx] = 2.5;
                }
            }
        }

        double dt = 0.01;
        solver.step_esdirk34(dt);

        // Verify state remains physical and stable
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int idx = iy * npts + ix;
                    double rho = c->U[0 * npts * npts + idx];
                    double u = c->U[1 * npts * npts + idx] / rho;
                    double v = c->U[2 * npts * npts + idx] / rho;
                    double E = c->U[3 * npts * npts + idx];
                    double P = (p.GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v));

                    CHECK(rho > 0.0);
                    CHECK(P > 0.0);
                    CHECK(rho == doctest::Approx(1.0).epsilon(1e-4));
                }
            }
        }
    }

    TEST_CASE("C1-Smooth Limiter - Density and Pressure Positivity") {
        auto p = make_params(2, 2, 2);
        Solver solver(p);
        Basis basis(p.P_DEG);

        int npts = p.N_PTS;
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int idx = iy * npts + ix;
                    c->U[0 * npts * npts + idx] = 2.0;
                    c->U[1 * npts * npts + idx] = 0.0;
                    c->U[2 * npts * npts + idx] = 0.0;
                    c->U[3 * npts * npts + idx] = 2.5;
                }
            }
        }

        // Introduce negative density dip
        solver.cells[0]->U[0 * npts * npts + 0] = -0.2;

        Limiters::apply_full_limiter_2d(*solver.cells[0], basis, p.GAMMA, p.POS_LIMITER_EPS);

        // Verify positivity is restored
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                int idx = iy * npts + ix;
                double rho = solver.cells[0]->U[0 * npts * npts + idx];
                double E = solver.cells[0]->U[3 * npts * npts + idx];
                double u = solver.cells[0]->U[1 * npts * npts + idx] / rho;
                double v = solver.cells[0]->U[2 * npts * npts + idx] / rho;
                double P = (p.GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v));

                CHECK(rho >= p.POS_LIMITER_EPS);
                CHECK(P >= p.POS_LIMITER_EPS);
            }
        }
    }
}

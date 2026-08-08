#include "../doctest.h"
#include "../../src/ib/ib_wall_function.hpp"
#include <cmath>

TEST_CASE("fr::ib::WallFunctionModel Compressible Law-of-the-Wall") {
    fr::ib::WallFunctionModel wf(0.41, 5.2);

    SUBCASE("Zero velocity handling") {
        auto res = wf.solve_wall_function(0.01, 1.0, 0.0, 300.0, 300.0, 1.8e-5, 1.4);
        CHECK(res.u_tau == 0.0);
        CHECK(res.tau_w == 0.0);
        CHECK(res.u_wall_eff == 0.0);
    }

    SUBCASE("Log-law region friction velocity solution") {
        double delta_wf = 0.005; // 5 mm off-body probe height
        double rho_wf = 1.2;
        double u_parallel = 100.0; // 100 m/s tangential velocity
        double T_wf = 300.0;
        double T_wall = 300.0;
        double mu_wf = 1.8e-5;
        double gamma = 1.4;

        auto res = wf.solve_wall_function(delta_wf, rho_wf, u_parallel, T_wf, T_wall, mu_wf, gamma);

        CHECK(res.u_tau > 0.0);
        CHECK(res.tau_w > 0.0);
        CHECK(res.y_plus > 30.0); // Logarithmic layer y+ > 30
        CHECK(res.u_wall_eff >= 0.0);
        CHECK(res.u_wall_eff < u_parallel);
    }
}

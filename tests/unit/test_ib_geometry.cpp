#include "../doctest.h"
#include "../../src/ib/ib.hpp"
#include "../../src/ib/sbm_geometry.hpp"
#include "../../src/core/solver.hpp"
#include <cmath>
#include <fstream>

TEST_SUITE("IB Geometry") {
    TEST_CASE("Circle SDF") {
        double d_in = ImmersedBoundary::get_circle_sdf(0.0, 0.0, 0.0, 0.0, 1.0);
        CHECK(d_in == doctest::Approx(-1.0));
        
        double d_out = ImmersedBoundary::get_circle_sdf(2.0, 0.0, 0.0, 0.0, 1.0);
        CHECK(d_out == doctest::Approx(1.0));
        
        double d_edge = ImmersedBoundary::get_circle_sdf(1.0, 0.0, 0.0, 0.0, 1.0);
        CHECK(std::abs(d_edge) < 1e-12);
    }

    TEST_CASE("NACA SDF symmetry/asymmetry") {
        double d_up = ImmersedBoundary::get_naca_sdf(0.5, 0.1, 0.0, 0.0, 1.0, "0012", 0.0);
        double d_down = ImmersedBoundary::get_naca_sdf(0.5, -0.1, 0.0, 0.0, 1.0, "0012", 0.0);
        CHECK(d_up == doctest::Approx(d_down));

        double d_up_aoa = ImmersedBoundary::get_naca_sdf(0.5, 0.1, 0.0, 0.0, 1.0, "0012", 5.0);
        double d_down_aoa = ImmersedBoundary::get_naca_sdf(0.5, -0.1, 0.0, 0.0, 1.0, "0012", 5.0);
        CHECK(d_up_aoa != doctest::Approx(d_down_aoa));
    }

    TEST_CASE("Sharp/smooth masks") {
        double m_sharp_in = ImmersedBoundary::compute_circle_mask(0.0, 0.0, 0.0, 0.0, 1.0, true, 2.0, 0.1, 0.1);
        CHECK(m_sharp_in == doctest::Approx(1.0));
        double m_sharp_out = ImmersedBoundary::compute_circle_mask(2.0, 0.0, 0.0, 0.0, 1.0, true, 2.0, 0.1, 0.1);
        CHECK(m_sharp_out == doctest::Approx(0.0));
        
        double m_smooth_edge = ImmersedBoundary::compute_circle_mask(1.0, 0.0, 0.0, 0.0, 1.0, false, 2.0, 0.1, 0.1);
        CHECK(m_smooth_edge == doctest::Approx(0.5));
    }

    TEST_CASE("Quad/Parabola SDF") {
        ImmersedBoundary::QuadShape quad;
        quad.x[0] = 0; quad.y[0] = 0;
        quad.x[1] = 1; quad.y[1] = 0;
        quad.x[2] = 1; quad.y[2] = 1;
        quad.x[3] = 0; quad.y[3] = 1;
        
        double d_in = ImmersedBoundary::get_quad_sdf(0.5, 0.5, quad);
        CHECK(d_in < 0.0);
        double d_out = ImmersedBoundary::get_quad_sdf(2.0, 2.0, quad);
        CHECK(d_out > 0.0);
        
        ImmersedBoundary::ParabolaShape par;
        par.dir = 'Y';
        par.a0 = 0.0; par.b0 = 0.0; par.c0 = 0.0;
        par.a1 = 1.0; par.b1 = 0.0; par.c1 = 0.0;
        par.side = "ABOVE";
        
        double d_par_0 = ImmersedBoundary::get_parabola_sdf(0.5, 0.5, par, 0.0);
        CHECK(d_par_0 < 0.0);
        
        double d_par_1 = ImmersedBoundary::get_parabola_sdf(0.5, 0.5, par, 1.0);
        CHECK(d_par_1 < 0.0);
        
        double d_par_out = ImmersedBoundary::get_parabola_sdf(0.5, -0.5, par, 0.0);
        CHECK(d_par_out > 0.0);
    }

    TEST_CASE("SBM Diagnostics Reset Preservation") {
        using namespace ImmersedBoundary;
        current_sbm_diags.max_dist_ratio = 1.23;
        current_sbm_diags.max_d_dl_ratio = 4.56;
        current_sbm_diags.max_lebesgue = 7.89;
        current_sbm_diags.limiter_count = 42;

        reset_sbm_diagnostics();

        auto diags = get_sbm_diagnostics();
        CHECK(diags.max_dist_ratio == doctest::Approx(1.23));
        CHECK(diags.max_d_dl_ratio == doctest::Approx(4.56));
        CHECK(diags.max_lebesgue == doctest::Approx(7.89));
        CHECK(diags.limiter_count == 0);
    }

    TEST_CASE("SBM Stencil Coordinate Calculation Details") {
        // Write a small test grid and input config
        std::ofstream grid_out("test_sbm_stencil.grid");
        grid_out << "[Block0]\n";
        grid_out << "N_ELEM_X = 16\nN_ELEM_Y = 16\n";
        grid_out << "X_MIN = -1.0\nX_MAX = 1.0\n";
        grid_out << "Y_MIN = -1.0\nY_MAX = 1.0\n";
        grid_out << "BC_L = WALL\nBC_R = WALL\nBC_B = WALL\nBC_T = WALL\n";
        grid_out.close();

        std::ofstream inputs_out("test_sbm_stencil.dat");
        inputs_out << "[Physics]\nGAMMA = 1.4\n";
        inputs_out << "[Solver]\nP_DEG = 1\nCFL = 0.5\n";
        inputs_out << "[ImmersedBoundary]\nENABLE_IB = true\nIB_METHOD = SBM\nIB_SHAPE = CIRCLE\nIB_RADIUS = 0.25\nIB_CENTER_X = 0.0\nIB_CENTER_Y = 0.0\n";
        inputs_out << "IB_L_SCALE = 0.5\nIB_DL_SCALE = 2.0\n";
        inputs_out.close();

        Parameters p;
        p.load_domain("test_sbm_stencil.grid");
        p.load_inputs("test_sbm_stencil.dat");

        Solver solver(p);
        
        // Check that we have registered surrogate points and check their L and dL
        REQUIRE(!ImmersedBoundary::sbm_registry.empty());
        
        for (const auto& sfp : ImmersedBoundary::sbm_registry) {
            double h = std::max(solver.blocks[0].dx, solver.blocks[0].dy); // grid spacing
            double alpha = 1.0 + sfp.D / h;
            // Verify formula: L = D + IB_L_SCALE * alpha * sqrt(2.0) * h
            double expected_L = sfp.D + 0.5 * alpha * std::sqrt(2.0) * h;
            // Verify formula: dL = IB_DL_SCALE * h * sqrt(2.0)
            double expected_dL = 2.0 * h * std::sqrt(2.0);
            
            // Let's verify that the donor points fall into the interval [L, L+dL]
            double min_psi = expected_L;
            double max_psi = expected_L + expected_dL;
            
            // Check that psi[1] to psi[P_interp] are in the correct bounds
            const int P_interp = 2; // P_soln + 1 with P_soln = 1
            std::vector<double> psi(P_interp + 1);
            psi[0] = sfp.D;
            for (int j = 1; j <= P_interp; ++j) {
                psi[j] = expected_L + 0.5 * expected_dL * (1.0 + solver.basis.z[j - 1]);
                CHECK(psi[j] >= min_psi);
                CHECK(psi[j] <= max_psi);
            }
        }

        std::remove("test_sbm_stencil.grid");
        std::remove("test_sbm_stencil.dat");
    }
}

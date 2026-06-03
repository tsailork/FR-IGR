#include "../doctest.h"
#include "../../src/ib/ib.hpp"
#include <cmath>

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
}

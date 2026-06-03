#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/limiters/positivity.hpp"
#include "../../src/limiters/entropy.hpp"

TEST_SUITE("Limiters") {
    TEST_CASE("Positivity Limiter - Physical state unchanged") {
        auto p = make_params(2, 1, 1);
        Solver solver(p);
        Basis basis(p.P_DEG);
        
        for(int iy=0; iy<p.N_PTS; ++iy) {
            for(int ix=0; ix<p.N_PTS; ++ix) {
                solver.blocks[0].U(0, 0, 0, iy, ix) = 1.0;
                solver.blocks[0].U(1, 0, 0, iy, ix) = 0.0;
                solver.blocks[0].U(2, 0, 0, iy, ix) = 0.0;
                solver.blocks[0].U(3, 0, 0, iy, ix) = 2.5; 
            }
        }
        
        auto stats = Limiters::apply_positivity_limiter(solver.blocks[0].U, basis, p);
        CHECK(stats.num_limited == 0);
        
        for(int iy=0; iy<p.N_PTS; ++iy) {
            for(int ix=0; ix<p.N_PTS; ++ix) {
                CHECK(solver.blocks[0].U(0, 0, 0, iy, ix) == doctest::Approx(1.0));
            }
        }
    }
    
    TEST_CASE("Positivity Limiter - Negative density/pressure fixed") {
        auto p = make_params(2, 1, 1);
        Solver solver(p);
        Basis basis(p.P_DEG);
        
        for(int iy=0; iy<p.N_PTS; ++iy) {
            for(int ix=0; ix<p.N_PTS; ++ix) {
                solver.blocks[0].U(0, 0, 0, iy, ix) = 2.0;
                solver.blocks[0].U(1, 0, 0, iy, ix) = 0.0;
                solver.blocks[0].U(2, 0, 0, iy, ix) = 0.0;
                solver.blocks[0].U(3, 0, 0, iy, ix) = 2.5;
            }
        }

        solver.blocks[0].U(0, 0, 0, 0, 0) = -0.5;
        solver.blocks[0].U(3, 0, 0, 0, 0) = 2.5;
        
        auto stats = Limiters::apply_positivity_limiter(solver.blocks[0].U, basis, p);
        CHECK(stats.num_limited > 0);
        
        for(int iy=0; iy<p.N_PTS; ++iy) {
            for(int ix=0; ix<p.N_PTS; ++ix) {
                CHECK(solver.blocks[0].U(0, 0, 0, iy, ix) >= p.POS_LIMITER_EPS);
            }
        }
    }
    
    TEST_CASE("Entropy Limiter - Physical state unchanged") {
        auto p = make_params(2, 2, 2);
        Solver solver(p);
        
        for(auto& b : solver.blocks) {
            for(int ey=0; ey<b.ny; ++ey) {
                for(int ex=0; ex<b.nx; ++ex) {
                    for(int iy=0; iy<p.N_PTS; ++iy) {
                        for(int ix=0; ix<p.N_PTS; ++ix) {
                            b.U(0, ey, ex, iy, ix) = 1.0;
                            b.U(1, ey, ex, iy, ix) = 0.0;
                            b.U(2, ey, ex, iy, ix) = 0.0;
                            b.U(3, ey, ex, iy, ix) = 2.5; 
                        }
                    }
                }
            }
        }
        
        auto stats = Limiters::apply_entropy_limiter(solver);
        CHECK(stats.num_limited == 0);
    }
}

#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/limiters/positivity.hpp"
#include "../../src/limiters/entropy.hpp"

TEST_SUITE("Limiters") {
    TEST_CASE("Positivity Limiter - Physical state unchanged") {
        auto p = make_params(2, 1, 1);
        Solver solver(p);
        Basis basis(p.P_DEG);
        
        // Set all cells to uniform physical state (rho=1, u=0, v=0, E=2.5)
        for (Cell* c : solver.cells) {
            int npts = p.N_PTS;
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    c->U[0*npts*npts + iy*npts + ix] = 1.0;
                    c->U[1*npts*npts + iy*npts + ix] = 0.0;
                    c->U[2*npts*npts + iy*npts + ix] = 0.0;
                    c->U[3*npts*npts + iy*npts + ix] = 2.5;
                }
            }
        }
        
        auto stats = Limiters::apply_positivity_limiter(solver.cells, basis, p);
        CHECK(stats.num_limited == 0);
        
        // Verify state is unchanged
        for (Cell* c : solver.cells) {
            int npts = p.N_PTS;
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    CHECK(c->U[0*npts*npts + iy*npts + ix] == doctest::Approx(1.0));
                }
            }
        }
    }
    
    TEST_CASE("Positivity Limiter - Negative density/pressure fixed") {
        auto p = make_params(2, 1, 1);
        Solver solver(p);
        Basis basis(p.P_DEG);
        
        int npts = p.N_PTS;
        
        // Set all cells to uniform physical state
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    c->U[0*npts*npts + iy*npts + ix] = 2.0;
                    c->U[1*npts*npts + iy*npts + ix] = 0.0;
                    c->U[2*npts*npts + iy*npts + ix] = 0.0;
                    c->U[3*npts*npts + iy*npts + ix] = 2.5;
                }
            }
        }

        // Set a negative density at one node of the first cell
        solver.cells[0]->U[0*npts*npts + 0*npts + 0] = -0.5;
        solver.cells[0]->U[3*npts*npts + 0*npts + 0] = 2.5;
        
        auto stats = Limiters::apply_positivity_limiter(solver.cells, basis, p);
        CHECK(stats.num_limited > 0);
        
        // Verify all densities are above threshold
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    CHECK(c->U[0*npts*npts + iy*npts + ix] >= p.POS_LIMITER_EPS);
                }
            }
        }
    }
    
    TEST_CASE("Entropy Limiter - Physical state unchanged") {
        auto p = make_params(2, 2, 2);
        Solver solver(p);
        
        int npts = p.N_PTS;
        
        // Set all cells to uniform physical state
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    c->U[0*npts*npts + iy*npts + ix] = 1.0;
                    c->U[1*npts*npts + iy*npts + ix] = 0.0;
                    c->U[2*npts*npts + iy*npts + ix] = 0.0;
                    c->U[3*npts*npts + iy*npts + ix] = 2.5;
                }
            }
        }
        
        auto stats = Limiters::apply_entropy_limiter(solver);
        CHECK(stats.num_limited == 0);
    }
}

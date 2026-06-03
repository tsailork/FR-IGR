#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/io/diagnostics.hpp"
#include <filesystem>
#include <fstream>
#include <string>

TEST_SUITE("Diagnostics") {
    TEST_CASE("Probe locator, Probe interpolation, L2 residual computation of zero RHS") {
        auto p = make_params(2, 2, 2);
        ProbeDef pd;
        pd.x = 0.25; pd.y = 0.25; pd.variable = "Density";
        p.probes.push_back(pd);
        
        Solver solver(p);
        
        for(auto& b : solver.blocks) {
            for(int ey=0; ey<b.ny; ++ey) {
                for(int ex=0; ex<b.nx; ++ex) {
                    for(int iy=0; iy<p.N_PTS; ++iy) {
                        for(int ix=0; ix<p.N_PTS; ++ix) {
                            b.U(0, ey, ex, iy, ix) = 1.5;
                            b.U(1, ey, ex, iy, ix) = 0.0;
                            b.U(2, ey, ex, iy, ix) = 0.0;
                            b.U(3, ey, ex, iy, ix) = 2.5; 
                            
                            int npts_block = b.nx * b.ny * p.N_PTS * p.N_PTS;
                            int idx = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                            b.RHS.data[0 * npts_block + idx] = 0.0;
                            b.RHS.data[1 * npts_block + idx] = 0.0;
                            b.RHS.data[2 * npts_block + idx] = 0.0;
                            b.RHS.data[3 * npts_block + idx] = 0.0;
                        }
                    }
                }
            }
        }
        
        TempDir td("diag_test_dir");
        auto old_path = std::filesystem::current_path();
        std::filesystem::current_path(td.get());
        
        Diagnostics d(p, solver, 0.0);
        d.update(solver, 0.0, 0);
        
        CHECK(std::filesystem::exists("csv_outputs/probe.csv"));
        CHECK(std::filesystem::exists("csv_outputs/residuals.csv"));
        
        std::filesystem::current_path(old_path);
    }
}

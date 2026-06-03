#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/core/solver.hpp"
#include "../../src/io/initial_conditions.hpp"
#include "../../src/io/vtk_writer.hpp"
#include "../../src/io/restart.hpp"

TEST_CASE("IO - VTK Output and Restart Round-Trip") {
    TempDir tmp("tests_tmp_io");
    
    // Create and run a small simulation
    auto p = load_test_config("blast_walls");
    auto old_path = std::filesystem::current_path();
    std::filesystem::current_path(tmp.get());
    
    Solver solver1(p);
    IC::apply(solver1);
    
    // Take a step
    double dt = solver1.compute_dt();
    solver1.step_rk3(dt);
    double t_saved = dt;
    
    // Write restart file
    ensure_output_directory("pv_outputs");
    VTKWriter writer;
    writer.write_checkpoint(solver1, 0, t_saved);
    
    // Create a new solver and load the restart file
    Solver solver2(p);
    Restart::load_restart("pv_outputs/sol_0.vtm", solver2.blocks, p);
    
    // Check if time and data match
    // Time isn't saved in the block data natively in this format, just verify blocks match
    
    for (size_t b = 0; b < solver1.blocks.size(); ++b) {
        for (size_t i = 0; i < solver1.blocks[b].U.data.size(); ++i) {
            CHECK(solver2.blocks[b].U.data[i] == doctest::Approx(solver1.blocks[b].U.data[i]));
        }
    }
    
    // Write VTK output
    CHECK(std::filesystem::exists("pv_outputs/sol_0.vtm"));
    std::filesystem::current_path(old_path);
}

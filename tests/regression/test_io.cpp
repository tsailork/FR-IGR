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
    Restart::load_restart("pv_outputs/sol_0.vtu", solver2);
    
    // Check if cell states match
    for (size_t c = 0; c < solver1.cells.size(); ++c) {
        for (size_t i = 0; i < solver1.cells[c]->U.size(); ++i) {
            CHECK(solver2.cells[c]->U[i] == doctest::Approx(solver1.cells[c]->U[i]));
        }
    }
    
    // Write VTK output
    CHECK(std::filesystem::exists("pv_outputs/sol_0.vtu"));
    std::filesystem::current_path(old_path);
}

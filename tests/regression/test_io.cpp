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
    std::string vtu_path = "pv_outputs/sol_0.vtu";
    CHECK(std::filesystem::exists(vtu_path));

    std::ifstream vtu(vtu_path);
    CHECK(vtu.is_open());
    std::string line;
    bool found_binary_tag = false;
    while (std::getline(vtu, line)) {
        if (line.find("format=\"binary\"") != std::string::npos) {
            found_binary_tag = true;
            break;
        }
    }
    CHECK(found_binary_tag);
    vtu.close();

    std::filesystem::current_path(old_path);
}

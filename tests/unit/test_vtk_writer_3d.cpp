#include "../doctest.h"
#include "../../src/core/solver.hpp"
#include "../../src/core/parameters.hpp"
#include "../../src/io/vtk_writer.hpp"
#include "../../src/io/restart.hpp"
#include <fstream>

TEST_CASE("3D VTK Binary Writer and Restart Loader") {
    Parameters p;
    p.P_DEG = 1;
    p.N_PTS = 2;
    p.GAMMA = 1.4;
    SolverDim<3> solver(p);

    BlockConfig bc;
    bc.id = 0;
    bc.N_ELEM_X = 1; bc.N_ELEM_Y = 1; bc.N_ELEM_Z = 1;
    bc.X_MIN = 0.0; bc.X_MAX = 1.0;
    bc.Y_MIN = 0.0; bc.Y_MAX = 1.0;
    bc.Z_MIN = 0.0; bc.Z_MAX = 1.0;
    bc.BC_L = "WALL"; bc.BC_R = "WALL";
    bc.BC_B = "WALL"; bc.BC_T = "WALL";
    bc.BC_F = "WALL"; bc.BC_K = "WALL";

    solver.blocks.emplace_back(bc, p.N_PTS);
    solver.initialize_cells();

    CHECK(solver.cells.size() == 1);
    Cell3D* c = solver.cells[0];

    // Set initial 5-component state array
    int npts3 = 8;
    for (int i = 0; i < npts3; ++i) {
        c->U[0 * npts3 + i] = 1.25 + 0.1 * i; // rho
        c->U[1 * npts3 + i] = 2.50 + 0.2 * i; // rho_u
        c->U[2 * npts3 + i] = -1.5 + 0.1 * i; // rho_v
        c->U[3 * npts3 + i] = 0.75 + 0.05 * i;// rho_w
        c->U[4 * npts3 + i] = 10.0 + 0.5 * i; // rho_E
    }

    ensure_output_directory("pv_outputs");

    VTKWriter writer("test_sol_3d");
    writer.write_checkpoint(solver, 9999, 0.5);

    std::string vtu_path = "pv_outputs/sol_9999.vtu";
    std::ifstream vtu(vtu_path);
    CHECK(vtu.is_open());

    // Verify binary format tag present
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

    // Re-initialize solver state and load restart file
    for (int i = 0; i < 5 * npts3; ++i) {
        c->U[i] = 0.0;
    }

    bool success = Restart::load_restart(vtu_path, solver);
    CHECK(success);

    // Verify values restored exact double precision / float precision match
    for (int i = 0; i < npts3; ++i) {
        CHECK(c->U[0 * npts3 + i] == doctest::Approx(1.25 + 0.1 * i).epsilon(1e-5));
        CHECK(c->U[1 * npts3 + i] == doctest::Approx(2.50 + 0.2 * i).epsilon(1e-5));
        CHECK(c->U[2 * npts3 + i] == doctest::Approx(-1.5 + 0.1 * i).epsilon(1e-5));
        CHECK(c->U[3 * npts3 + i] == doctest::Approx(0.75 + 0.05 * i).epsilon(1e-5));
        CHECK(c->U[4 * npts3 + i] == doctest::Approx(10.0 + 0.5 * i).epsilon(1e-5));
    }
}

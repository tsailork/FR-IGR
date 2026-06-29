#include "../doctest.h"
#include "../../src/core/parameters.hpp"
#include "../../src/core/solver.hpp"
#include <fstream>
#include <cstdio>

TEST_CASE("Parameters - Default values") {
    Parameters p;
    CHECK(p.CFL == doctest::Approx(0.5));
    CHECK(p.GAMMA == doctest::Approx(1.4));
    CHECK(p.ENABLE_IGR == false);
}

TEST_CASE("Parameters - Parse basic INI") {
    std::ofstream out("test_ini.txt");
    out << "[Section1]\nKey1=Value1\nKey2 = 3.14 \n";
    out << "[Section2]\nKeyA=StringVal\n";
    out.close();

    auto map = Parameters::parse_ini("test_ini.txt");
    CHECK(map["Section1"]["Key1"] == "Value1");
    CHECK(map["Section1"]["Key2"] == "3.14");
    CHECK(map["Section2"]["KeyA"] == "StringVal");
    
    std::remove("test_ini.txt");
}

TEST_CASE("Parameters - domain.grid single/multiblock and BC strings") {
    std::ofstream out("test_domain.grid");
    out << "[Block0]\n";
    out << "N_ELEM_X = 10\nN_ELEM_Y = 20\n";
    out << "X_MIN = -1.0\nX_MAX = 1.0\n";
    out << "Y_MIN = 0.0\nY_MAX = 2.0\n";
    out << "BC_L = WALL\nBC_R = OUTFLOW\nBC_B = INFLOW\nBC_T = PERIODIC\n";
    out.close();

    Parameters p;
    p.load_domain("test_domain.grid");
    
    REQUIRE(p.blocks.size() == 1);
    CHECK(p.blocks[0].N_ELEM_X == 10);
    CHECK(p.blocks[0].N_ELEM_Y == 20);
    CHECK(p.blocks[0].X_MIN == doctest::Approx(-1.0));
    CHECK(p.blocks[0].X_MAX == doctest::Approx(1.0));
    CHECK(p.blocks[0].BC_L == "WALL");
    CHECK(p.blocks[0].BC_R == "OUTFLOW");
    
    std::remove("test_domain.grid");
}

TEST_CASE("Parameters - Load inputs.dat") {
    std::ofstream out("test_inputs.dat");
    out << "[Physics]\nGAMMA = 1.3\n";
    out << "[Solver]\nCFL = 0.8\nENABLE_MULTIRATE = true\nMAX_MULTIRATE_LEVEL = 4\n";
    out << "[Regularization]\nENABLE_IGR = true\nIGR_TYPE = PARABOLIC\nUSE_DUCROS_SWITCH = true\nUSE_PRESSURE_SENSOR = true\nUSE_MOMENTUM_DIV = true\nUSE_PRESSURE_SOURCE_CAP = false\nSOURCE_CAP_COEFF = 2.5\n";
    out << "[ImmersedBoundary]\nIB_L_SCALE = 0.75\nIB_DL_SCALE = 1.5\nIB_CHORD = 3.2\n";
    out.close();

    Parameters p;
    p.load_inputs("test_inputs.dat");
    
    CHECK(p.GAMMA == doctest::Approx(1.3));
    CHECK(p.CFL == doctest::Approx(0.8));
    CHECK(p.ENABLE_MULTIRATE == true);
    CHECK(p.MAX_MULTIRATE_LEVEL == 4);
    CHECK(p.ENABLE_IGR == true);
    CHECK(p.IGR_TYPE == "PARABOLIC");
    CHECK(p.USE_DUCROS_SWITCH == true);
    CHECK(p.USE_PRESSURE_SENSOR == true);
    CHECK(p.USE_MOMENTUM_DIV == true);
    CHECK(p.USE_PRESSURE_SOURCE_CAP == false);
    CHECK(p.SOURCE_CAP_COEFF == doctest::Approx(2.5));
    CHECK(p.IB_L_SCALE == doctest::Approx(0.75));
    CHECK(p.IB_DL_SCALE == doctest::Approx(1.5));
    CHECK(p.IB_CHORD == doctest::Approx(3.2));
    
    std::remove("test_inputs.dat");
}

TEST_CASE("Parameters - WALL_SLIP alias test") {
    std::ofstream out("test_domain_wall_slip.grid");
    out << "[Block0]\n";
    out << "N_ELEM_X = 10\nN_ELEM_Y = 20\n";
    out << "X_MIN = -1.0\nX_MAX = 1.0\n";
    out << "Y_MIN = 0.0\nY_MAX = 2.0\n";
    out << "BC_L = WALL_SLIP\nBC_R = WALL\nBC_B = INFLOW\nBC_T = PERIODIC\n";
    out.close();

    Parameters p;
    p.load_domain("test_domain_wall_slip.grid");

    Solver solver(p);
    
    REQUIRE(solver.cells.size() > 0);

    // Find cells on the left domain boundary and verify WALL_SLIP → is_wall
    bool found_left_wall = false;
    bool found_right_wall = false;
    for (const auto* cell : solver.cells) {
        if (cell->is_boundary[0]) {  // Left face is a domain boundary
            CHECK(cell->boundary_info[0].is_wall == true);
            found_left_wall = true;
        }
        if (cell->is_boundary[1]) {  // Right face is a domain boundary
            CHECK(cell->boundary_info[1].is_wall == true);
            found_right_wall = true;
        }
    }
    CHECK(found_left_wall);
    CHECK(found_right_wall);

    std::remove("test_domain_wall_slip.grid");
}

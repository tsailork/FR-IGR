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
    out << "[Solver]\nCFL = 0.8\n";
    out << "[Regularization]\nENABLE_IGR = true\nIGR_TYPE = PARABOLIC\n";
    out << "[ImmersedBoundary]\nIB_L_SCALE = 0.75\nIB_DL_SCALE = 1.5\n";
    out.close();

    Parameters p;
    p.load_inputs("test_inputs.dat");
    
    CHECK(p.GAMMA == doctest::Approx(1.3));
    CHECK(p.CFL == doctest::Approx(0.8));
    CHECK(p.ENABLE_IGR == true);
    CHECK(p.IGR_TYPE == "PARABOLIC");
    CHECK(p.IB_L_SCALE == doctest::Approx(0.75));
    CHECK(p.IB_DL_SCALE == doctest::Approx(1.5));
    
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
    
    REQUIRE(solver.blocks.size() == 1);
    CHECK(solver.blocks[0].ni_l.is_wall == true);
    CHECK(solver.blocks[0].ni_r.is_wall == true);

    std::remove("test_domain_wall_slip.grid");
}

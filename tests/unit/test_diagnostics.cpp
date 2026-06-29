#include "../doctest.h"
#include "../test_helpers.hpp"
#include "../../src/io/diagnostics.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

TEST_SUITE("Diagnostics") {
    TEST_CASE("Probe locator, Probe interpolation, L2 residual computation of zero RHS") {
        auto p = make_params(2, 2, 2);
        p.ENABLE_IGR = true;
        
        std::vector<std::string> vars = {
            "Density", "XMomentum", "YMomentum", "Energy", 
            "Pressure", "Temperature", "Mach", "Sigma"
        };
        for (const auto& v : vars) {
            ProbeDef pd;
            pd.x = 0.25; pd.y = 0.25; pd.variable = v;
            p.probes.push_back(pd);
        }
        
        Solver solver(p);
        
        int npts = p.N_PTS;
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int pt = iy * npts + ix;
                    c->U[0 * npts * npts + pt] = 1.5;
                    c->U[1 * npts * npts + pt] = 0.3; // rhou = 0.3 -> u = 0.2
                    c->U[2 * npts * npts + pt] = 0.0;
                    c->U[3 * npts * npts + pt] = 2.5;

                    c->RHS[0 * npts * npts + pt] = 0.0;
                    c->RHS[1 * npts * npts + pt] = 0.0;
                    c->RHS[2 * npts * npts + pt] = 0.0;
                    c->RHS[3 * npts * npts + pt] = 0.0;

                    c->sigma_field[pt] = 0.12;
                }
            }
        }
        
        TempDir td("diag_test_dir");
        auto old_path = std::filesystem::current_path();
        std::filesystem::current_path(td.get());
        
        Diagnostics d(p, solver, 0.0);
        d.update(solver, 0.001, 1);
        
        std::filesystem::current_path(old_path);
        
        std::string probe_path = td.get() + "/csv_outputs/probe.csv";
        std::string res_path = td.get() + "/csv_outputs/residuals.csv";
        
        CHECK(std::filesystem::exists(probe_path));
        CHECK(std::filesystem::exists(res_path));
        
        // Read and parse probe.csv
        std::ifstream file(probe_path);
        REQUIRE(file.is_open());
        
        std::string header, data_line;
        REQUIRE(std::getline(file, header));
        REQUIRE(std::getline(file, data_line));
        
        std::stringstream ss(data_line);
        std::string token;
        std::vector<double> vals;
        while (std::getline(ss, token, ',')) {
            vals.push_back(std::stod(token));
        }
        
        // Expected: Time, Density, XMomentum, YMomentum, Energy, Pressure, Temperature, Mach, Sigma
        REQUIRE(vals.size() == 9);
        CHECK(vals[0] == doctest::Approx(0.001)); // Time
        CHECK(vals[1] == doctest::Approx(1.5)); // Density
        CHECK(vals[2] == doctest::Approx(0.3)); // XMomentum
        CHECK(vals[3] == doctest::Approx(0.0)); // YMomentum
        CHECK(vals[4] == doctest::Approx(2.5)); // Energy
        
        // Pressure = (gamma - 1) * (E - 0.5 * rhou * u) = 0.4 * (2.5 - 0.5 * 0.3 * 0.2) = 0.4 * 2.47 = 0.988
        CHECK(vals[5] == doctest::Approx(0.988)); // Pressure
        
        // Temperature = p / rho = 0.988 / 1.5 = 0.6586666...
        CHECK(vals[6] == doctest::Approx(0.988 / 1.5)); // Temperature
        
        // Mach = u / c = 0.2 / sqrt(1.4 * 0.988 / 1.5)
        CHECK(vals[7] == doctest::Approx(0.2 / std::sqrt(1.4 * 0.988 / 1.5))); // Mach
        
        CHECK(vals[8] == doctest::Approx(0.12)); // Sigma
    }

    TEST_CASE("VPM Aerodynamic Force Integration") {
        auto p = make_params(2, 2, 2);
        p.ENABLE_IB = true;
        p.IB_METHOD = "VPM";
        p.IB_PENALIZATION_ETA = 1e-3;
        p.IB_VELOCITY_X = 0.1;
        p.IB_VELOCITY_Y = 0.0;
        p.IB_CHORD = 1.0;
        p.RHO_INF = 1.0;
        p.U_INF = 1.0;
        p.V_INF = 0.0;
        
        Solver solver(p);
        
        // Populate mask and flow variables
        int npts = p.N_PTS;
        for (Cell* c : solver.cells) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int pt = iy * npts + ix;
                    // Set left half of elements to be inside the solid body
                    c->ib_mask[pt] = (c->ex < 1) ? 1.0 : 0.0;

                    c->U[0 * npts * npts + pt] = 1.0; // rho = 1.0
                    c->U[1 * npts * npts + pt] = 0.5; // rhou = 0.5 (u = 0.5)
                    c->U[2 * npts * npts + pt] = 0.2; // rhov = 0.2 (v = 0.2)
                    c->U[3 * npts * npts + pt] = 2.0;
                }
            }
        }
        
        TempDir td("force_test_dir");
        auto old_path = std::filesystem::current_path();
        std::filesystem::current_path(td.get());
        
        Diagnostics d(p, solver, 0.0);
        d.update(solver, 0.001, 1);
        
        std::filesystem::current_path(old_path);
        
        std::string forces_path = td.get() + "/csv_outputs/forces.csv";
        CHECK(std::filesystem::exists(forces_path));
        
        std::ifstream file(forces_path);
        REQUIRE(file.is_open());
        
        std::string line1, line2, data_line;
        REQUIRE(std::getline(file, line1));
        REQUIRE(std::getline(file, line2));
        REQUIRE(std::getline(file, data_line));
        
        std::stringstream ss(data_line);
        std::string token;
        std::vector<double> vals;
        while (std::getline(ss, token, ',')) {
            vals.push_back(std::stod(token));
        }
        
        // Expected columns: Time, DragForce, LiftForce, Cd, Cl
        REQUIRE(vals.size() == 5);
        CHECK(vals[0] == doctest::Approx(0.001));
        
        // Analytical check:
        // fx = (chi / eta) * (rhou - rho * u_s) = (1.0 / 1e-3) * (0.5 - 1.0 * 0.1) = 1000 * 0.4 = 400.0
        // fy = (chi / eta) * (rhov - rho * v_s) = (1.0 / 1e-3) * (0.2 - 1.0 * 0.0) = 1000 * 0.2 = 200.0
        // We integrate over elements where ex < nx/2 (which is exactly half the elements).
        // Total integrated area is half the domain area.
        // For a 2x2 element grid from x_min=0 to x_max=1, y_min=0 to y_max=1, total area is 1.0.
        // Half the area is 0.5.
        // Integrated DragForce = 400.0 * 0.5 = 200.0
        // Integrated LiftForce = 200.0 * 0.5 = 100.0
        // Cd = DragForce / (0.5 * rho_inf * U_inf^2 * chord) = 200.0 / (0.5 * 1.0 * 1.0^2 * 1.0) = 400.0
        // Cl = LiftForce / (0.5 * rho_inf * U_inf^2 * chord) = 100.0 / (0.5 * 1.0 * 1.0^2 * 1.0) = 200.0
        CHECK(vals[1] == doctest::Approx(200.0));
        CHECK(vals[2] == doctest::Approx(100.0));
        CHECK(vals[3] == doctest::Approx(400.0));
        CHECK(vals[4] == doctest::Approx(200.0));
    }
}

#pragma once
#include "doctest.h"
#include "../src/core/parameters.hpp"
#include "../src/core/solver.hpp"
#include "../src/io/initial_conditions.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

// Creates minimal parameters with defaults
inline Parameters make_params(int p_deg, int nx, int ny) {
    Parameters p;
    p.P_DEG = p_deg;
    p.N_PTS = p_deg + 1;
    
    BlockConfig b;
    b.id = 0;
    b.N_ELEM_X = nx;
    b.N_ELEM_Y = ny;
    b.X_MIN = 0.0;
    b.X_MAX = 1.0;
    b.Y_MIN = 0.0;
    b.Y_MAX = 1.0;
    b.BC_L = "OUTFLOW_SUPERSONIC";
    b.BC_R = "OUTFLOW_SUPERSONIC";
    b.BC_B = "OUTFLOW_SUPERSONIC";
    b.BC_T = "OUTFLOW_SUPERSONIC";
    
    p.blocks.push_back(b);
    return p;
}

// Loads a test configuration from the fixtures directory
inline Parameters load_test_config(const std::string& test_name) {
    Parameters p;
    std::string base_path = "tests/fixtures/" + test_name + "/";
    p.load_domain(base_path + "domain.grid");
    p.load_inputs(base_path + "inputs.dat");
    return p;
}

// Temporary directory manager for tests
class TempDir {
    std::string path;
public:
    TempDir(const std::string& p) : path(p) {
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::filesystem::remove_all(path);
    }
    std::string get() const { return path; }
};

// Computes total mass in a solver
inline double compute_total_mass(const Solver& solver) {
    double mass = 0.0;
    for (const auto& b : solver.blocks) {
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < solver.p.N_PTS; ++iy) {
                    for (int ix = 0; ix < solver.p.N_PTS; ++ix) {
                        double weight = solver.basis.w[ix] * solver.basis.w[iy] * b.dx * b.dy / 4.0;
                        mass += b.U(0, ey, ex, iy, ix) * weight;
                    }
                }
            }
        }
    }
    return mass;
}

// Computes total energy in a solver
inline double compute_total_energy(const Solver& solver) {
    double energy = 0.0;
    for (const auto& b : solver.blocks) {
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < solver.p.N_PTS; ++iy) {
                    for (int ix = 0; ix < solver.p.N_PTS; ++ix) {
                        double weight = solver.basis.w[ix] * solver.basis.w[iy] * b.dx * b.dy / 4.0;
                        energy += b.U(3, ey, ex, iy, ix) * weight;
                    }
                }
            }
        }
    }
    return energy;
}

// Computes L2 norm of a conservative variable over a block
inline double compute_L2_norm(const Block& b, const Basis& basis, int var) {
    double norm_sq = 0.0;
    int npts = basis.w.size();
    for (int ey = 0; ey < b.ny; ++ey) {
        for (int ex = 0; ex < b.nx; ++ex) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    double weight = basis.w[ix] * basis.w[iy] * b.dx * b.dy / 4.0;
                    double val = b.U(var, ey, ex, iy, ix);
                    norm_sq += val * val * weight;
                }
            }
        }
    }
    return std::sqrt(norm_sq);
}

// Verifies no NaN or Inf in state
inline double compute_L2_norm(const Solver& solver) {
    double norm = 0.0;
    for (const auto& b : solver.blocks) {
        for (size_t i = 0; i < b.U.data.size(); ++i) {
            norm += b.U.data[i] * b.U.data[i];
        }
    }
    return std::sqrt(norm);
}

inline void setup_solver_ic(Solver& solver) {
    IC::apply(solver);
    if (solver.p.ENABLE_IGR && solver.p.IGR_TYPE == "PARABOLIC") {
        solver.compute_sensor_source();
        for (auto& b : solver.blocks) {
            b.sigma_field = b.S_buf;
        }
    }
}

inline void assert_no_nan(const Solver& solver) {
    for (const auto& b : solver.blocks) {
        for (size_t i = 0; i < b.U.data.size(); ++i) {
            CHECK(std::isfinite(b.U.data[i]));
        }
    }
}

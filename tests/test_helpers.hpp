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

// Computes total mass in a solver (cell-local version)
inline double compute_total_mass(const Solver& solver) {
    double mass = 0.0;
    int npts = solver.p.N_PTS;
    for (const Cell* c : solver.cells) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double weight = solver.basis.w[ix] * solver.basis.w[iy] * c->dx * c->dy / 4.0;
                mass += c->U[0*npts*npts + iy*npts + ix] * weight;
            }
        }
    }
    return mass;
}

// Computes total energy in a solver (cell-local version)
inline double compute_total_energy(const Solver& solver) {
    double energy = 0.0;
    int npts = solver.p.N_PTS;
    for (const Cell* c : solver.cells) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double weight = solver.basis.w[ix] * solver.basis.w[iy] * c->dx * c->dy / 4.0;
                energy += c->U[3*npts*npts + iy*npts + ix] * weight;
            }
        }
    }
    return energy;
}

// Computes L2 norm of a conservative variable over a cell
inline double compute_L2_norm_cell(const Cell& c, const Basis& basis, int var) {
    double norm_sq = 0.0;
    int npts = (int)basis.w.size();
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            double weight = basis.w[ix] * basis.w[iy] * c.dx * c.dy / 4.0;
            double val = c.U[var*npts*npts + iy*npts + ix];
            norm_sq += val * val * weight;
        }
    }
    return std::sqrt(norm_sq);
}

// Computes global L2 norm of all conserved variables (cell-local version)
inline double compute_L2_norm(const Solver& solver) {
    double norm = 0.0;
    for (const Cell* c : solver.cells) {
        for (size_t i = 0; i < c->U.size(); ++i) {
            norm += c->U[i] * c->U[i];
        }
    }
    return std::sqrt(norm);
}

inline void setup_solver_ic(Solver& solver) {
    IC::apply(solver);
    if (solver.p.ENABLE_IGR && solver.p.IGR_TYPE == "PARABOLIC") {
        solver.compute_sensor_source();
        for (Cell* c : solver.cells) {
            c->sigma_field = c->S_buf;
        }
    }
}

inline void assert_no_nan(const Solver& solver) {
    for (const Cell* c : solver.cells) {
        for (size_t i = 0; i < c->U.size(); ++i) {
            CHECK(std::isfinite(c->U[i]));
        }
    }
}

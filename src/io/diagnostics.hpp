/// @file diagnostics.hpp
/// @brief Handles diagnostic tracking: residuals, point probes, and terminal output.

#pragma once
#include "../core/solver.hpp"
#include "../core/parameters.hpp"
#include <fstream>
#include <chrono>

class Diagnostics {
public:
    Diagnostics(const Parameters& p, const Solver& solver);
    ~Diagnostics();

    void update(const Solver& solver, double t, int step, double dt);

private:
    const Parameters& params;
    double next_residual_output;
    double next_probe_output;
    double next_print_output;
    
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> last_print_wall_time;

    std::ofstream res_file;
    std::ofstream probe_file;

    // Pre-computed spatial locators for probes
    struct ProbeLocator {
        const ProbeDef* def;
        int ey, ex;
        double L_xi[MAX_PTS];
        double L_eta[MAX_PTS];
    };
    std::vector<ProbeLocator> locators;

    double evaluate_probe(const Solver& solver, const ProbeLocator& loc) const;
};

/**
 * @file diagnostics.hpp
 * @brief Handles diagnostic tracking: residuals, point probes, and terminal output on decoupled cells.
 */

#pragma once
#include "../core/parameters.hpp"
#include "../core/cell.hpp"
#include <fstream>
#include <chrono>
#include <vector>

class Solver;

/**
 * @class Diagnostics
 * @brief Manages simulation health metrics, probe sampling, and user-facing terminal logs.
 */
class Diagnostics {
public:
    /**
     * @brief Constructs the Diagnostics tracker and opens output file streams.
     */
    Diagnostics(const Parameters& p, const Solver& solver, double startTime);

    /**
     * @brief Destructor that cleanly flushes and closes active file streams.
     */
    ~Diagnostics();

    /**
     * @brief Updates all diagnostic metrics and writes to disk if intervals are met.
     */
    void update(const Solver& solver, double t, int step);

private:
    const Parameters& params;
    double sim_start_time;
    double next_residual_output;
    double next_probe_output;
    double next_print_output;
    
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> last_print_wall_time;

    std::ofstream res_file;
    std::ofstream probe_file;
    std::ofstream force_file;

    // Pre-computed spatial locators for probes
    struct ProbeLocator {
        const ProbeDef* def;
        Cell* cell_ptr;
        std::vector<double> L_xi;
        std::vector<double> L_eta;
    };
    std::vector<ProbeLocator> locators;

    double evaluate_probe(const Solver& solver, const ProbeLocator& loc) const;
};

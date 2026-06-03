/**
 * @file diagnostics.hpp
 * @brief Handles diagnostic tracking: residuals, point probes, and terminal output.
 *
 * Provides a unified interface for evaluating L2 norms of the flux residuals, extracting 
 * pointwise time-series data, and formatting the console status outputs during the simulation.
 */

#pragma once
#include "../core/solver.hpp"
#include "../core/parameters.hpp"
#include <fstream>
#include <chrono>

/**
 * @class Diagnostics
 * @brief Manages simulation health metrics, probe sampling, and user-facing terminal logs.
 *
 * Tracks the \f$L_2\f$ residual of the SSP-RK3 temporal updates, samples physical fields 
 * at specified geometrical coordinates, and provides real-time progress updates including 
 * ETA estimations.
 */
class Diagnostics {
public:
    /**
     * @brief Constructs the Diagnostics tracker and opens output file streams.
     *
     * @param[in] p Global simulation parameters defining diagnostic intervals.
     * @param[in] solver The main solver instance.
     * @param[in] startTime The initial physical time of the simulation.
     */
    Diagnostics(const Parameters& p, const Solver& solver, double startTime);

    /**
     * @brief Destructor that cleanly flushes and closes active file streams.
     */
    ~Diagnostics();

    /**
     * @brief Updates all diagnostic metrics and writes to disk if intervals are met.
     *
     * Evaluates time-step conditions to determine if residuals should be calculated, 
     * probes should be sampled, or terminal output should be printed.
     *
     * @param[in] solver The main solver instance.
     * @param[in] t Current physical time.
     * @param[in] step Current iteration step number.
     * @see Parameters::RESIDUAL_INTERVAL
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

    // Pre-computed spatial locators for probes
    struct ProbeLocator {
        const ProbeDef* def;
        int block_id;
        int ey, ex;
        double L_xi[MAX_PTS];
        double L_eta[MAX_PTS];
    };
    std::vector<ProbeLocator> locators;

    double evaluate_probe(const Solver& solver, const ProbeLocator& loc) const;
};

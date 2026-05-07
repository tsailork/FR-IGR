/// @file main.cpp
/// @brief Main entry point for the Flux Reconstruction Euler solver.

#include "core/parameters.hpp"
#include "core/solver.hpp"
#include "io/initial_conditions.hpp"
#include "io/restart.hpp"
#include "io/vtk_writer.hpp"
#include "io/diagnostics.hpp"
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    // 1. Load Parameters
    Parameters params;
    params.load_domain("domain.grid");
    params.load_inputs("inputs.dat");

    std::cout << "Starting FR-IGR Solver\n";
#ifdef _OPENMP
    std::cout << "OpenMP Enabled: " << omp_get_max_threads() << " threads\n";
#else
    std::cout << "OpenMP Disabled (Serial Mode)\n";
#endif

    // 2. Setup Solver
    Solver solver(params);

    // 3. Initialize Output Directory and Writer
    ensure_output_directory("pv_outputs");
    VTKWriter writer("solution");

    // 4. Initialization: restart from file, or apply fresh IC
    double t = 0.0;
    int output_count = 0;

    if (!params.RESTART_FILE.empty()) {
        std::cout << "Restarting from: " << params.RESTART_FILE << "\n";
        if (!Restart::load_restart(params.RESTART_FILE, solver.blocks, params)) {
            std::cerr << "[FATAL] Restart failed. Aborting.\n";
            return 1;
        }
        t = params.RESTART_TIME;
        output_count = static_cast<int>(std::round(t / params.OUTPUT_DT));
        std::cout << "Resuming at t=" << t << ", output_count=" << output_count << "\n";
    } else {
        std::cout << "Initializing solver with IC: " << params.IC_TYPE << "...\n";
        IC::apply(solver);

        // Initialize Sigma field for Parabolic mode
        if (params.ENABLE_IGR && params.IGR_TYPE == "PARABOLIC") {
            solver.compute_sensor_source();
            for (auto& b : solver.blocks) {
                b.sigma_field = b.S_buf;
            }
        }
    }

    // 5. Time Stepping Loop
    int step = 0;
    double next_output = output_count * params.OUTPUT_DT + params.OUTPUT_DT;

    writer.write_snapshot(solver, output_count++, t);
    Diagnostics diag(params, solver);

    while (t < params.T_FINAL) {
        double dt = solver.compute_dt();

        if (t + dt > next_output) dt = next_output - t;
        if (t + dt > params.T_FINAL) dt = params.T_FINAL - t;

        solver.step_rk3(dt);
        t += dt;
        step++;

        diag.update(solver, t, step, dt);

        if (std::abs(t - next_output) < 1e-12) {
            writer.write_snapshot(solver, output_count++, t);
            next_output += params.OUTPUT_DT;
        }
    }

    std::cout << "Simulation complete.\n";
    return 0;
}

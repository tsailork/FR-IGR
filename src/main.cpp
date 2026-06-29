/**
 * @file main.cpp
 * @brief Main entry point for the Flux Reconstruction Euler solver.
 *
 * Orchestrates the full CFD simulation pipeline:
 *  1. Parameter and domain grid loading.
 *  2. Solver initialization (applying initial conditions or VTK format restart serialization).
 *  3. Explicit time-stepping loop using dynamic CFL stability conditions and SSP-RK3 integration.
 *  4. Periodic execution of diagnostics, state checkpoints, and ParaView plot generation.
 *
 * @see Solver::step_rk3
 * @see VTKWriter::write_plot
 */

#include "core/parameters.hpp"
#include "core/solver.hpp"
#include "io/initial_conditions.hpp"
#include "io/restart.hpp"
#include "io/vtk_writer.hpp"
#include "io/diagnostics.hpp"
#include "ib/sbm_geometry.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    // 1. Load Parameters
    Parameters params;
    params.load_domain("domain.grid");
    params.load_inputs("inputs.dat");

    // Redirect stdout and stderr to out.log, disable buffering
    if (!params.RESTART_FILE.empty()) {
        std::freopen("out.log", "a", stdout);
        std::freopen("out.log", "a", stderr);
    } else {
        std::freopen("out.log", "w", stdout);
        std::freopen("out.log", "w", stderr);
    }
    std::setvbuf(stdout, NULL, _IONBF, 0);
    std::setvbuf(stderr, NULL, _IONBF, 0);
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

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
    ensure_output_directory("csv_outputs");
    VTKWriter writer("solution");

    // 4. Initialization: restart from file, or apply fresh IC
    double t = 0.0;
    int checkpoint_count = 0;
    int plot_count = 0;

    if (!params.RESTART_FILE.empty()) {
        std::cout << "Restarting from: " << params.RESTART_FILE << "\n";
        writer.load_existing_pvd();
        if (!Restart::load_restart(params.RESTART_FILE, solver)) {
            std::cerr << "[FATAL] Restart failed. Aborting.\n";
            return 1;
        }
        t = params.RESTART_TIME;
        checkpoint_count = static_cast<int>(std::round(t / params.RESTART_INTERVAL));
        plot_count = static_cast<int>(std::round(t / params.OUTPUT_INTERVAL));
        std::cout << "Resuming at t=" << t << ", checkpoint_count=" << checkpoint_count 
                  << ", plot_count=" << plot_count << "\n";
    } else {
        std::cout << "Initializing solver with IC: " << params.IC_TYPE << "...\n";
        IC::apply(solver);

        // Initialize Sigma field for Parabolic mode
        if (params.ENABLE_IGR && params.IGR_TYPE == "PARABOLIC") {
            solver.compute_sensor_source();
            for (Cell* c : solver.cells) {
                c->sigma_field = c->S_buf;
            }
        }
        
        if (params.ENABLE_SBM_DIAGNOSTICS && params.IB_METHOD == "SBM") {
            std::ofstream sbm_file("csv_outputs/sbm_diagnostics.csv");
            if (sbm_file.is_open()) {
                sbm_file << "Time,Max_Lebesgue,Limiter_Count,Max_Dist_Ratio,Max_D_dL_Ratio\n";
            }
        }
    }

    // 5. Time Stepping Loop
    int step = 0;
    double next_checkpoint = checkpoint_count * params.RESTART_INTERVAL + params.RESTART_INTERVAL;
    double next_plot       = plot_count * params.OUTPUT_INTERVAL + params.OUTPUT_INTERVAL;
    double next_residual_sbm = (std::floor(t / params.RESIDUAL_INTERVAL) + 1.0) * params.RESIDUAL_INTERVAL;

    // Write initial states
    writer.write_checkpoint(solver, checkpoint_count++, t);
    writer.write_plot(solver, plot_count++, t);
    
    Diagnostics diag(params, solver, t);

    while (t < params.T_FINAL) {
        if (std::filesystem::exists("STOP")) {
            std::cout << "\n[STOP] STOP file detected. Shutting down solver cleanly...\n";
            std::cout.flush();
            std::filesystem::remove("STOP");
            break;
        }

        double dt = solver.compute_dt();

        // Limit dt to hit output intervals exactly
        if (t + dt > next_checkpoint) dt = next_checkpoint - t;
        if (t + dt > next_plot)       dt = next_plot - t;
        if (t + dt > params.T_FINAL)  dt = params.T_FINAL - t;

        solver.step_rk3(dt);
        t += dt;
        step++;

        if (solver.sbm_nonphysical_count > 0) {
            std::cout << "[WARNING] " << solver.sbm_nonphysical_count 
                      << " nonphysical SBM states (rho <= 0 or E <= 0) detected at step " << step << "\n";
            solver.sbm_nonphysical_count = 0;
        }

        diag.update(solver, t, step);

        // Write SBM Diagnostics if scheduled
        if (t >= next_residual_sbm) {
            if (params.ENABLE_SBM_DIAGNOSTICS && params.IB_METHOD == "SBM") {
                auto sbm_diags = ImmersedBoundary::get_sbm_diagnostics();
                std::ofstream sbm_file("csv_outputs/sbm_diagnostics.csv", std::ios::app);
                if (sbm_file.is_open()) {
                    sbm_file << std::scientific << t << "," 
                             << sbm_diags.max_lebesgue << "," 
                             << sbm_diags.limiter_count << "," 
                             << sbm_diags.max_dist_ratio << ","
                             << sbm_diags.max_d_dl_ratio << "\n";
                }
                std::cout << "\n[SBM DIAGNOSTICS] Max Lebesgue: " << sbm_diags.max_lebesgue 
                          << " | Limiter Triggers: " << sbm_diags.limiter_count 
                          << " | Max D/L: " << sbm_diags.max_dist_ratio 
                          << " | Max D/dL: " << sbm_diags.max_d_dl_ratio << "\n";
                ImmersedBoundary::reset_sbm_diagnostics();
            }
            next_residual_sbm += params.RESIDUAL_INTERVAL;
        }

        // Write checkpoint if scheduled
        if (std::abs(t - next_checkpoint) < 1e-12) {
            writer.write_checkpoint(solver, checkpoint_count++, t);
            next_checkpoint += params.RESTART_INTERVAL;
        }

        // Write plot if scheduled
        if (std::abs(t - next_plot) < 1e-12) {
            writer.write_plot(solver, plot_count++, t);
            next_plot += params.OUTPUT_INTERVAL;
        }
    }

    std::cout << "Simulation complete.\n";
    return 0;
}

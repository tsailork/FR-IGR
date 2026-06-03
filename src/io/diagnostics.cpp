/**
 * @file diagnostics.cpp
 * @brief Implementation of diagnostic tracking and performance logging.
 *
 * Calculates the \f$L_2\f$ norm of the solution right-hand side (RHS) across all blocks 
 * using OpenMP reductions, and extracts sub-element scalar values for point probes 
 * using high-order polynomial interpolation.
 */

#include "diagnostics.hpp"
#include <iomanip>
#include <iostream>
#include <filesystem>

Diagnostics::Diagnostics(const Parameters& p, const Solver& solver, double startTime) 
    : params(p), sim_start_time(startTime) {
    
    start_time = std::chrono::steady_clock::now();
    last_print_wall_time = start_time;

    // Synchronize interval counters with startTime
    next_residual_output = (std::floor(startTime / params.RESIDUAL_INTERVAL) + 1.0) * params.RESIDUAL_INTERVAL;
    next_probe_output    = (std::floor(startTime / params.PROBE_INTERVAL) + 1.0) * params.PROBE_INTERVAL;
    next_print_output    = (std::floor(startTime / params.PRINT_INTERVAL) + 1.0) * params.PRINT_INTERVAL;

    std::filesystem::create_directories("csv_outputs");

    // Initialize Residuals File (Append if restarting)
    if (startTime > 0) {
        res_file.open("csv_outputs/residuals.csv", std::ios::app);
    } else {
        res_file.open("csv_outputs/residuals.csv");
        if (res_file.is_open()) {
            res_file << "# FR-IGR Residual Tracker (Multiblock)\n";
            res_file << "Time, L2_rho, L2_rhou, L2_rhov, L2_E\n";
        }
    }

    // Initialize Probes File (Append if restarting)
    if (!p.probes.empty()) {
        if (startTime > 0) {
            probe_file.open("csv_outputs/probe.csv", std::ios::app);
        } else {
            probe_file.open("csv_outputs/probe.csv");
            if (probe_file.is_open()) {
                probe_file << "Time";
                for (size_t i = 0; i < p.probes.size(); ++i) {
                    probe_file << ", Probe" << i << "_" << p.probes[i].variable;
                }
                probe_file << "\n";
                probe_file.flush();
            }
        }
        
        // Always rebuild locators (independent of file mode)
        for (const auto& probe_def : p.probes) {
            ProbeLocator loc;
            loc.def = &probe_def;
            loc.block_id = -1;

            for (const auto& b : solver.blocks) {
                double x_max = b.x_min + b.nx * b.dx;
                double y_max = b.y_min + b.ny * b.dy;
                if (loc.def->x >= b.x_min && loc.def->x <= x_max &&
                    loc.def->y >= b.y_min && loc.def->y <= y_max) {
                    loc.block_id = b.id;
                    
                    double ex_frac = (loc.def->x - b.x_min) / b.dx;
                    double ey_frac = (loc.def->y - b.y_min) / b.dy;
                    loc.ex = std::clamp(static_cast<int>(ex_frac), 0, b.nx - 1);
                    loc.ey = std::clamp(static_cast<int>(ey_frac), 0, b.ny - 1);

                    double xc = b.x_min + (loc.ex + 0.5) * b.dx;
                    double yc = b.y_min + (loc.ey + 0.5) * b.dy;
                    double xi  = (loc.def->x - xc) / (0.5 * b.dx);
                    double eta = (loc.def->y - yc) / (0.5 * b.dy);

                    for (int j = 0; j < p.N_PTS; ++j) {
                        loc.L_xi[j] = 1.0;
                        loc.L_eta[j] = 1.0;
                        for (int k = 0; k < p.N_PTS; ++k) {
                            if (j != k) {
                                loc.L_xi[j]  *= (xi  - solver.basis.z[k]) / (solver.basis.z[j] - solver.basis.z[k]);
                                loc.L_eta[j] *= (eta - solver.basis.z[k]) / (solver.basis.z[j] - solver.basis.z[k]);
                            }
                        }
                    }
                    break;
                }
            }
            
            if (loc.block_id != -1) locators.push_back(loc);
            else std::cerr << "[DIAG] Warning: Probe at (" << loc.def->x << "," << loc.def->y << ") outside domain.\n";
        }
    }
}

Diagnostics::~Diagnostics() {
    if (res_file.is_open()) res_file.close();
    if (probe_file.is_open()) probe_file.close();
}

double Diagnostics::evaluate_probe(const Solver& solver, const ProbeLocator& loc) const {
    const Block* target_block = nullptr;
    for (const auto& b : solver.blocks) if (b.id == loc.block_id) { target_block = &b; break; }
    if (!target_block) return 0.0;

    double u_int[4] = {0,0,0,0};
    double sig_int = 0;

    for (int iy = 0; iy < params.N_PTS; ++iy) {
        for (int ix = 0; ix < params.N_PTS; ++ix) {
            double w = loc.L_eta[iy] * loc.L_xi[ix];
            u_int[0] += w * target_block->U(0, loc.ey, loc.ex, iy, ix);
            u_int[1] += w * target_block->U(1, loc.ey, loc.ex, iy, ix);
            u_int[2] += w * target_block->U(2, loc.ey, loc.ex, iy, ix);
            u_int[3] += w * target_block->U(3, loc.ey, loc.ex, iy, ix);
            
            if (params.ENABLE_IGR) {
                sig_int += w * target_block->sigma_field[target_block->get_flat_idx(loc.ey, loc.ex, iy, ix, params.N_PTS)];
            }
        }
    }

    double rho = std::max(1e-12, u_int[0]);
    double u = u_int[1] / rho, v = u_int[2] / rho;
    double p = (params.GAMMA - 1.0) * (u_int[3] - 0.5 * rho * (u*u + v*v));

    if (loc.def->variable == "Density") return rho;
    if (loc.def->variable == "Pressure") return p;
    if (loc.def->variable == "Mach") return std::sqrt(u*u + v*v) / std::sqrt(params.GAMMA * std::abs(p) / rho);
    if (loc.def->variable == "Sigma") return sig_int;
    return 0.0;
}

void Diagnostics::update(const Solver& solver, double t, int step) {
    bool need_residual = (t >= next_residual_output && res_file.is_open()) || (t >= next_print_output);
    std::vector<double> res(4, 0.0);
    
    if (need_residual) {
        double l2_rho = 0, l2_rhou = 0, l2_rhov = 0, l2_E = 0;
        long long total_pts = 0;

        for (const auto& b : solver.blocks) {
            int npts_block = b.nx * b.ny * params.N_PTS * params.N_PTS;
            total_pts += npts_block;
            
            #pragma omp parallel for reduction(+:l2_rho, l2_rhou, l2_rhov, l2_E)
            for (int i = 0; i < npts_block; ++i) {
                l2_rho  += b.RHS.data[0 * npts_block + i] * b.RHS.data[0 * npts_block + i];
                l2_rhou += b.RHS.data[1 * npts_block + i] * b.RHS.data[1 * npts_block + i];
                l2_rhov += b.RHS.data[2 * npts_block + i] * b.RHS.data[2 * npts_block + i];
                l2_E    += b.RHS.data[3 * npts_block + i] * b.RHS.data[3 * npts_block + i];
            }
        }
        
        res[0] = std::sqrt(l2_rho / total_pts);
        res[1] = std::sqrt(l2_rhou / total_pts);
        res[2] = std::sqrt(l2_rhov / total_pts);
        res[3] = std::sqrt(l2_E / total_pts);
    }

    if (t >= next_residual_output && res_file.is_open()) {
        res_file << std::scientific << std::setprecision(6) << t << ", "
                 << res[0] << ", " << res[1] << ", " << res[2] << ", " << res[3] << "\n";
        res_file.flush();
        next_residual_output += params.RESIDUAL_INTERVAL;
    }

    if (t >= next_probe_output && probe_file.is_open()) {
        probe_file << std::scientific << std::setprecision(6) << t;
        for (const auto& loc : locators) probe_file << ", " << evaluate_probe(solver, loc);
        probe_file << "\n";
        probe_file.flush();
        next_probe_output += params.PROBE_INTERVAL;
    }

    if (t >= next_print_output) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> wall_elapsed = now - last_print_wall_time;
        std::chrono::duration<double> total_elapsed = now - start_time;
        last_print_wall_time = now;

        double l2_sum = res[0] + res[1] + res[2] + res[3];
        double progress = (t / params.T_FINAL) * 100.0;
        
        double dt_run = t - sim_start_time;
        double eta = 0.0;
        if (dt_run > 1e-8) {
            eta = (total_elapsed.count() / dt_run) * (params.T_FINAL - t);
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "[Step " << std::setw(5) << step << "] "
                  << "t: " << std::setw(6) << t << " | "
                  << std::setw(5) << progress << "% | "
                  << "ETA: " << std::setw(5) << eta << "s | "
                  << std::scientific << std::setprecision(3)
                  << "L2_Sum: " << l2_sum;

        if (params.ENABLE_POS_LIMITER || params.ENABLE_ENTROPY_LIMITER) {
            double avg_theta = (solver.current_limiter_stats.num_limited > 0) ? 
                                solver.current_limiter_stats.sum_theta / solver.current_limiter_stats.num_limited : 0.0;
            std::cout << " | Lim: " << solver.current_limiter_stats.num_limited 
                      << " (avg_th: " << std::fixed << std::setprecision(4) << avg_theta << ")";
        }
        std::cout << "\n";
        next_print_output += params.PRINT_INTERVAL;
    }
}

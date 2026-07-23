/**
 * @file diagnostics.cpp
 * @brief Implementation of diagnostic tracking and performance logging on decoupled cells.
 */

#include "diagnostics.hpp"
#include "../core/solver.hpp"
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

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
            res_file << "# FR-IGR Residual Tracker\n";
            res_file << "Time, L2_rho, L2_rhou, L2_rhov, L2_E\n";
        }
    }

    // Initialize Forces File (Append if restarting)
    if (params.ENABLE_IB && (params.IB_METHOD == "VPM" || params.IB_METHOD == "VPM_ANALYTICAL" || params.IB_METHOD == "VPM_EXPLICIT")) {
        if (startTime > 0) {
            force_file.open("csv_outputs/forces.csv", std::ios::app);
        } else {
            force_file.open("csv_outputs/forces.csv");
            if (force_file.is_open()) {
                force_file << "# VPM Aerodynamic Forces\n";
                force_file << "Time, DragForce, LiftForce, Cd, Cl\n";
            }
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
        
        // Find hosting Cell and build 1D Lagrange interpolants
        for (const auto& probe_def : p.probes) {
            ProbeLocator loc;
            loc.def = &probe_def;
            loc.cell_ptr = nullptr;

            for (Cell* c : solver.cells) {
                double x_max = c->x_min + c->dx;
                double y_max = c->y_min + c->dy;
                if (loc.def->x >= c->x_min && loc.def->x <= x_max &&
                    loc.def->y >= c->y_min && loc.def->y <= y_max) {
                    loc.cell_ptr = c;
                    
                    double xc = c->x_min + 0.5 * c->dx;
                    double yc = c->y_min + 0.5 * c->dy;
                    double xi  = (loc.def->x - xc) / (0.5 * c->dx);
                    double eta = (loc.def->y - yc) / (0.5 * c->dy);

                    loc.L_xi.assign(p.N_PTS, 1.0);
                    loc.L_eta.assign(p.N_PTS, 1.0);
                    for (int j = 0; j < p.N_PTS; ++j) {
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
            
            if (loc.cell_ptr != nullptr) {
                locators.push_back(loc);
            } else {
                std::cerr << "[DIAG] Warning: Probe at (" << loc.def->x << "," << loc.def->y << ") outside domain.\n";
            }
        }
    }
}

Diagnostics::~Diagnostics() {
    if (res_file.is_open()) res_file.close();
    if (probe_file.is_open()) probe_file.close();
    if (force_file.is_open()) force_file.close();
}

double Diagnostics::evaluate_probe(const Solver& solver, const ProbeLocator& loc) const {
    const Cell* c = loc.cell_ptr;
    if (!c) return 0.0;

    double u_int[4] = {0.0, 0.0, 0.0, 0.0};
    double sig_int = 0.0;

    for (int iy = 0; iy < params.N_PTS; ++iy) {
        for (int ix = 0; ix < params.N_PTS; ++ix) {
            double w = loc.L_eta[iy] * loc.L_xi[ix];
            u_int[0] += w * c->get_U(0, iy, ix, params.N_PTS);
            u_int[1] += w * c->get_U(1, iy, ix, params.N_PTS);
            u_int[2] += w * c->get_U(2, iy, ix, params.N_PTS);
            u_int[3] += w * c->get_U(3, iy, ix, params.N_PTS);
            
            if (params.ENABLE_IGR) {
                sig_int += w * c->sigma_field[iy * params.N_PTS + ix];
            }
        }
    }

    double rho = std::max(1e-12, u_int[0]);
    double u = u_int[1] / rho, v = u_int[2] / rho;
    double p = (params.GAMMA - 1.0) * (u_int[3] - 0.5 * rho * (u*u + v*v));

    if (loc.def->variable == "Density") return rho;
    if (loc.def->variable == "XMomentum") return u_int[1];
    if (loc.def->variable == "YMomentum") return u_int[2];
    if (loc.def->variable == "Energy") return u_int[3];
    if (loc.def->variable == "Pressure") return p;
    if (loc.def->variable == "Temperature") return p / rho;
    if (loc.def->variable == "Mach") return std::sqrt(u*u + v*v) / std::sqrt(params.GAMMA * std::abs(p) / rho);
    if (loc.def->variable == "Sigma") return sig_int;
    return 0.0;
}

void Diagnostics::update(const Solver& solver, double t, int step) {
    bool need_residual = (t >= next_residual_output && res_file.is_open()) || (t >= next_print_output);
    std::vector<double> res(4, 0.0);
    
    if (need_residual) {
        double l2_rho = 0.0, l2_rhou = 0.0, l2_rhov = 0.0, l2_E = 0.0;
        long long total_pts = solver.cells.size() * params.N_PTS * params.N_PTS;

        #pragma omp parallel for reduction(+:l2_rho, l2_rhou, l2_rhov, l2_E)
        for (size_t i = 0; i < solver.cells.size(); ++i) {
            Cell* c = solver.cells[i];
            int npts2 = params.N_PTS * params.N_PTS;
            for (int k = 0; k < npts2; ++k) {
                l2_rho  += c->RHS[0 * npts2 + k] * c->RHS[0 * npts2 + k];
                l2_rhou += c->RHS[1 * npts2 + k] * c->RHS[1 * npts2 + k];
                l2_rhov += c->RHS[2 * npts2 + k] * c->RHS[2 * npts2 + k];
                l2_E    += c->RHS[3 * npts2 + k] * c->RHS[3 * npts2 + k];
            }
        }
        
        if (total_pts > 0) {
            res[0] = std::sqrt(l2_rho / total_pts);
            res[1] = std::sqrt(l2_rhou / total_pts);
            res[2] = std::sqrt(l2_rhov / total_pts);
            res[3] = std::sqrt(l2_E / total_pts);
        }
    }

    if (t >= next_residual_output && res_file.is_open()) {
        res_file << std::scientific << std::setprecision(6) << t << ", "
                 << res[0] << ", " << res[1] << ", " << res[2] << ", " << res[3] << "\n";
        res_file.flush();
        
        // VPM Aerodynamic Forces evaluation and logging
        if (params.ENABLE_IB && (params.IB_METHOD == "VPM" || params.IB_METHOD == "VPM_ANALYTICAL" || params.IB_METHOD == "VPM_EXPLICIT") && force_file.is_open()) {
            double drag_force = 0.0;
            double lift_force = 0.0;
            double eta = params.IB_PENALIZATION_ETA;
            
            if (eta > 1e-15) {
                #pragma omp parallel for reduction(+:drag_force, lift_force)
                for (size_t i = 0; i < solver.cells.size(); ++i) {
                    Cell* c = solver.cells[i];
                    double weight_factor = c->dx * c->dy / 4.0;
                    for (int iy = 0; iy < params.N_PTS; ++iy) {
                        for (int ix = 0; ix < params.N_PTS; ++ix) {
                            int idx = iy * params.N_PTS + ix;
                            double chi = c->ib_mask[idx];
                            chi = std::max(0.0, chi - 0.5);
                            if (chi <= 0.0) continue;
                            
                            double rho  = c->get_U(0, iy, ix, params.N_PTS);
                            double rhou = c->get_U(1, iy, ix, params.N_PTS);
                            double rhov = c->get_U(2, iy, ix, params.N_PTS);
                            
                            double target_rhou = rho * params.IB_VELOCITY_X;
                            double target_rhov = rho * params.IB_VELOCITY_Y;
                            
                            double fx = (chi / eta) * (rhou - target_rhou);
                            double fy = (chi / eta) * (rhov - target_rhov);
                            
                            double w = solver.basis.w[ix] * solver.basis.w[iy] * weight_factor;
                            drag_force += fx * w;
                            lift_force += fy * w;
                        }
                    }
                }
            }
            
            double rho_inf = params.RHO_INF;
            double u_inf = params.U_INF;
            double v_inf = params.V_INF;
            double q_inf = 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf);
            double chord = params.IB_CHORD;
            
            double cd = 0.0, cl = 0.0;
            if (std::abs(q_inf) > 1e-12 && std::abs(chord) > 1e-12) {
                cd = drag_force / (q_inf * chord);
                cl = lift_force / (q_inf * chord);
            }
            
            force_file << std::scientific << std::setprecision(6) << t << ", "
                       << drag_force << ", " << lift_force << ", " << cd << ", " << cl << "\n";
            force_file.flush();
        }
        
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

Diagnostics::Diagnostics(const Parameters& p, const SolverDim<3>& solver, double startTime)
    : params(p), sim_start_time(startTime), next_residual_output(startTime), next_probe_output(startTime), next_print_output(startTime)
{
    start_time = std::chrono::steady_clock::now();
    last_print_wall_time = start_time;
    bool is_restart = (startTime > 0.0);
    std::ios_base::openmode mode = is_restart ? (std::ios::out | std::ios::app) : std::ios::out;
    res_file.open("csv_outputs/residuals.csv", mode);
    if (!is_restart && res_file.is_open()) {
        res_file << "Time, L2_Rho, L2_RhoU, L2_RhoV, L2_RhoW, L2_E\n";
    }
}

void Diagnostics::update(const SolverDim<3>& solver, double t, int step) {
    if (t >= next_print_output) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> total_elapsed = now - start_time;
        last_print_wall_time = now;
        double progress = (t / params.T_FINAL) * 100.0;
        double dt_run = t - sim_start_time;
        double eta = (dt_run > 1e-8) ? (total_elapsed.count() / dt_run) * (params.T_FINAL - t) : 0.0;

        std::cout << std::fixed << std::setprecision(4)
                  << "[Step " << std::setw(5) << step << "] "
                  << "t: " << std::setw(6) << t << " | "
                  << std::setw(5) << progress << "% | "
                  << "ETA: " << std::setw(5) << eta << "s\n";
        next_print_output += params.PRINT_INTERVAL;
    }
}

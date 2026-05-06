/// @file diagnostics.cpp
/// @brief Implementation of diagnostic tracking.

#include "diagnostics.hpp"
#include <iomanip>
#include <iostream>

Diagnostics::Diagnostics(const Parameters& p, const Solver& solver) 
    : params(p) {
    
    start_time = std::chrono::steady_clock::now();
    last_print_wall_time = start_time;

    next_residual_output = params.RESIDUAL_INTERVAL;
    next_probe_output = params.PROBE_INTERVAL;
    next_print_output = params.PRINT_INTERVAL;

    // Initialize Residuals File
    res_file.open("residuals.dat");
    if (res_file.is_open()) {
        res_file << "# FR-IGR Residual Tracker\n";
        res_file << "# Grid: " << p.N_ELEM_X << " x " << p.N_ELEM_Y << "\n";
        res_file << "# Polynomial Degree: " << p.P_DEG << "\n";
        res_file << "Time, L2_rho, L2_rhou, L2_rhov, L2_E\n";
    }

    // Initialize Probes File
    if (!p.probes.empty()) {
        probe_file.open("probe.csv");
        if (probe_file.is_open()) {
            probe_file << "Time";
            for (size_t i = 0; i < p.probes.size(); ++i) {
                probe_file << ", Probe" << i << "_" << p.probes[i].variable;
                
                // Set up locator
                ProbeLocator loc;
                loc.def = &p.probes[i];
                
                // Find element
                double ex_frac = (loc.def->x - p.X_MIN) / solver.dx;
                double ey_frac = (loc.def->y - p.Y_MIN) / solver.dy;
                
                loc.ex = std::clamp(static_cast<int>(ex_frac), 0, p.N_ELEM_X - 1);
                loc.ey = std::clamp(static_cast<int>(ey_frac), 0, p.N_ELEM_Y - 1);
                
                // Find local coordinates [-1, 1]
                double xc = p.X_MIN + (loc.ex + 0.5) * solver.dx;
                double yc = p.Y_MIN + (loc.ey + 0.5) * solver.dy;
                double xi  = (loc.def->x - xc) / (0.5 * solver.dx);
                double eta = (loc.def->y - yc) / (0.5 * solver.dy);

                // Pre-calculate Lagrange weights at the probe location
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
                
                locators.push_back(loc);
            }
            probe_file << "\n";
            probe_file.flush();
        }
    }
    if (res_file.is_open()) res_file.flush();
}

Diagnostics::~Diagnostics() {
    if (res_file.is_open()) res_file.close();
    if (probe_file.is_open()) probe_file.close();
}

double Diagnostics::evaluate_probe(const Solver& solver, const ProbeLocator& loc) const {
    // Interpolate the 4 conserved variables and sigma using pre-calculated weights

    // Interpolate the 4 conserved variables and sigma
    double u_int[4] = {0.0, 0.0, 0.0, 0.0};
    double sig_int = 0.0;

    for (int iy = 0; iy < params.N_PTS; ++iy) {
        for (int ix = 0; ix < params.N_PTS; ++ix) {
            double w = loc.L_eta[iy] * loc.L_xi[ix];
            u_int[0] += w * solver.U(0, loc.ey, loc.ex, iy, ix);
            u_int[1] += w * solver.U(1, loc.ey, loc.ex, iy, ix);
            u_int[2] += w * solver.U(2, loc.ey, loc.ex, iy, ix);
            u_int[3] += w * solver.U(3, loc.ey, loc.ex, iy, ix);
            
            if (params.ENABLE_IGR && params.IGR_TYPE == "PARABOLIC") {
                sig_int += w * solver.sigma_field[solver.get_flat_idx(loc.ey, loc.ex, iy, ix)];
            }
        }
    }

    double rho = std::max(1e-10, u_int[0]);
    double u = u_int[1] / rho;
    double v = u_int[2] / rho;
    double E = u_int[3];
    double press = (params.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v));

    if (loc.def->variable == "Density") return rho;
    if (loc.def->variable == "XMomentum") return u_int[1];
    if (loc.def->variable == "YMomentum") return u_int[2];
    if (loc.def->variable == "Energy") return E;
    if (loc.def->variable == "Pressure") return press;
    if (loc.def->variable == "Temperature") return press / (rho * 287.058); // Assuming R_specific
    if (loc.def->variable == "Mach") {
        double c = std::sqrt(params.GAMMA * std::max(1e-10, press) / rho);
        return std::sqrt(u*u + v*v) / c;
    }
    if (loc.def->variable == "Sigma") return sig_int;

    return 0.0;
}

void Diagnostics::update(const Solver& solver, double t, int step, double dt) {
    
    bool need_residual = (t >= next_residual_output && res_file.is_open()) || (t >= next_print_output);
    std::vector<double> res(4, 0.0);
    
    if (need_residual) {
        double l2_rho = 0.0, l2_rhou = 0.0, l2_rhov = 0.0, l2_E = 0.0;
        int N_pts = params.N_ELEM_X * params.N_ELEM_Y * params.N_PTS * params.N_PTS;
        
        #pragma omp parallel for reduction(+:l2_rho, l2_rhou, l2_rhov, l2_E)
        for (int i = 0; i < N_pts; ++i) {
            double r0 = solver.RHS.data[0 * N_pts + i];
            double r1 = solver.RHS.data[1 * N_pts + i];
            double r2 = solver.RHS.data[2 * N_pts + i];
            double r3 = solver.RHS.data[3 * N_pts + i];
            l2_rho  += r0 * r0;
            l2_rhou += r1 * r1;
            l2_rhov += r2 * r2;
            l2_E    += r3 * r3;
        }
        
        res[0] = std::sqrt(l2_rho / N_pts);
        res[1] = std::sqrt(l2_rhou / N_pts);
        res[2] = std::sqrt(l2_rhov / N_pts);
        res[3] = std::sqrt(l2_E / N_pts);
    }

    // --- Residual Tracking ---
    if (t >= next_residual_output && res_file.is_open()) {
        res_file << std::scientific << std::setprecision(6) << t << ", "
                 << res[0] << ", "
                 << res[1] << ", "
                 << res[2] << ", "
                 << res[3] << "\n";
        res_file.flush();
        next_residual_output += params.RESIDUAL_INTERVAL;
    }

    // --- Point Probes ---
    if (t >= next_probe_output && probe_file.is_open()) {
        probe_file << std::scientific << std::setprecision(6) << t;
        for (const auto& loc : locators) {
            probe_file << ", " << evaluate_probe(solver, loc);
        }
        probe_file << "\n";
        probe_file.flush();
        next_probe_output += params.PROBE_INTERVAL;
    }

    // --- Terminal Statistics ---
    if (t >= next_print_output) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> wall_elapsed = now - last_print_wall_time;
        std::chrono::duration<double> total_elapsed = now - start_time;
        last_print_wall_time = now;

        double l2_sum = res[0] + res[1] + res[2] + res[3];
        
        double progress = (t / params.T_FINAL) * 100.0;
        double eta = (total_elapsed.count() / t) * (params.T_FINAL - t);

        std::cout << std::fixed << std::setprecision(4)
                  << "[Step " << std::setw(5) << step << "] "
                  << "t: " << std::setw(6) << t << " | "
                  << std::setw(5) << progress << "% | "
                  << "ETA: " << std::setw(5) << eta << "s | "
                  << "Wall/print: " << std::setw(5) << wall_elapsed.count() << "s | "
                  << std::scientific << std::setprecision(3)
                  << "L2_Sum: " << l2_sum;

        if (params.ENABLE_POS_LIMITER || params.ENABLE_ENTROPY_LIMITER) {
            double avg_theta = 0.0;
            if (solver.current_limiter_stats.num_limited > 0) {
                avg_theta = solver.current_limiter_stats.sum_theta / solver.current_limiter_stats.num_limited;
            }
            std::cout << " | Lim: " << solver.current_limiter_stats.num_limited 
                      << " cells (avg_th: " << std::fixed << std::setprecision(4) << avg_theta << ")";
        }
        std::cout << "\n";

        next_print_output += params.PRINT_INTERVAL;
    }
}

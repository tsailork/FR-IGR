/**
 * @file rk3.cpp
 * @brief Strong Stability Preserving Runge-Kutta 3rd Order (SSP-RK3) time-stepping on decoupled Cells.
 */

#include "../core/solver.hpp"
#include "../limiters/entropy.hpp"
#include "../limiters/positivity.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::step_rk3(double dt) {
    if (p.ENABLE_MULTIRATE) {
        compute_local_dt();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            double dt_elem = c->element_dt;
            int m = 0;
            if (dt > 1e-15) {
                m = std::clamp(static_cast<int>(std::floor(std::log2(dt_elem / dt))), 0, p.MAX_MULTIRATE_LEVEL);
            }
            c->element_dt = (1 << m) * dt;
            if (current_time + dt >= c->element_time + c->element_dt - 1e-12) {
                c->element_active = true;
            } else {
                c->element_active = false;
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i]->element_active = true;
            cells[i]->element_dt = dt;
        }
    }

    // Save previous stage states in cell-local U_old and sigma_old buffers
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        c->U_old = c->U;
        c->sigma_old = c->sigma_field;
        if (p.ENABLE_PPR) {
            c->S_old = c->S_field;
        }
    }

    auto relax_phantom_pressure = [&](double dt_stage_ratio, double alpha, double beta) {
        if (!p.ENABLE_PPR) return;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            if (p.ENABLE_MULTIRATE && !c->element_active) continue;
            double dt_stage = dt_stage_ratio * c->element_dt;
            
            // 1. Explicit advection update for S
            for (size_t k = 0; k < c->S_field.size(); ++k) {
                c->S_field[k] = alpha * c->S_old[k] + beta * (c->S_field[k] + dt_stage * c->S_RHS[k]);
            }
            
            // 2. Analytical relaxation step
            double r_avg = 0.0, ru_avg = 0.0, rv_avg = 0.0, E_avg = 0.0;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                    r_avg  += w * c->get_U(0, iy, ix, p.N_PTS);
                    ru_avg += w * c->get_U(1, iy, ix, p.N_PTS);
                    rv_avg += w * c->get_U(2, iy, ix, p.N_PTS);
                    E_avg  += w * c->get_U(3, iy, ix, p.N_PTS);
                }
            }
            double rho_avg = std::max(p.POS_LIMITER_EPS, r_avg);
            double u_avg = ru_avg / rho_avg;
            double v_avg = rv_avg / rho_avg;
            double p_avg = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E_avg - 0.5 * rho_avg * (u_avg*u_avg + v_avg*v_avg)));
            double a_avg = std::sqrt(p.GAMMA * p_avg / rho_avg);
            double lambda_loc = std::sqrt(u_avg*u_avg + v_avg*v_avg) + a_avg;
            
            double tau = p.PPR_C_TAU * std::min(c->dx, c->dy) / (lambda_loc + 1e-12);
            double exp_factor = std::exp(-dt_stage / tau);
            
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int k = iy * p.N_PTS + ix;
                    double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                    double u = c->get_U(1, iy, ix, p.N_PTS) / rho;
                    double v = c->get_U(2, iy, ix, p.N_PTS) / rho;
                    double E = c->get_U(3, iy, ix, p.N_PTS);
                    double press = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));
                    double S_eq = rho * press;
                    
                    c->S_field[k] = S_eq + (c->S_field[k] - S_eq) * exp_factor;
                    if (c->S_field[k] < p.POS_LIMITER_EPS) {
                        c->S_field[k] = p.POS_LIMITER_EPS;
                    }
                }
            }
        }
    };

    if (p.ENABLE_IB && p.ib_is_dynamic) {
        update_ib_mask_field(current_time);
    }

    const bool is_parabolic = (p.ENABLE_IGR && p.IGR_TYPE == "PARABOLIC");
    int n_sub = 1;
    if (is_parabolic) {
        if (p.IGR_SUB_ITERS > 0) {
            n_sub = p.IGR_SUB_ITERS;
        } else {
            double alpha_safe = std::max(1e-10, p.ALPHA_SCALE);
            double dt_diff  = 0.5 * p.IGR_TAU_R / (alpha_safe * (1.0 + p.IGR_BR2_ETA) * (2 * p.P_DEG + 1) * (2 * p.P_DEG + 1));
            double dt_relax = 0.5 * p.IGR_TAU_R;
            double dt_limit = std::min(dt_diff, dt_relax);
            double max_dt_active = dt;
            if (p.ENABLE_MULTIRATE) {
                for (Cell* c : cells) {
                    if (c->element_active) max_dt_active = std::max(max_dt_active, c->element_dt);
                }
            }
            n_sub = static_cast<int>(std::ceil(max_dt_active / dt_limit));
            if (n_sub < 1) n_sub = 1;
        }
    }
    const double dt_sub = dt / n_sub;

    current_limiter_stats.num_limited = 0;
    current_limiter_stats.sum_theta = 0.0;
    auto add_stats = [&](const Limiters::LimiterStats &s) {
        current_limiter_stats.num_limited += s.num_limited;
        current_limiter_stats.sum_theta += s.sum_theta;
    };

    auto clamp_sigma_all = [&]() {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            if (p.ENABLE_MULTIRATE && !c->element_active) continue;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int idx = iy * p.N_PTS + ix;
                    if (c->sigma_field[idx] < 0.0) {
                        c->sigma_field[idx] = 0.0;
                        continue;
                    }
                    if (p.USE_PRESSURE_FIELD_CAP) {
                        double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                        double rhou = c->get_U(1, iy, ix, p.N_PTS);
                        double rhov = c->get_U(2, iy, ix, p.N_PTS);
                        double E = c->get_U(3, iy, ix, p.N_PTS);
                        double press = (p.GAMMA - 1.0) *
                                       (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
                        if (press < p.POS_LIMITER_EPS)
                            press = p.POS_LIMITER_EPS;

                        c->sigma_field[idx] = std::min(c->sigma_field[idx], press);
                    }
                }
            }
        }
    };

    auto sub_iterate_sigma_all = [&](double alpha, double beta) {
        if (p.IGR_SUB_ITER_TOL > 0.0) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells.size(); ++i) {
                Cell* c = cells[i];
                if (p.ENABLE_MULTIRATE && !c->element_active) continue;
                double dt_sub_c = c->element_dt / n_sub;
                const size_t N_sig = c->sigma_field.size();
                for (size_t k = 0; k < N_sig; ++k) {
                    c->sigma_field[k] = alpha * c->sigma_old[k] +
                                        beta * (c->sigma_field[k] + dt_sub_c * c->sigma_RHS[k]);
                }
            }
            clamp_sigma_all();

            double diff = 1e20;
            int sub = 1;
            const int max_sub = (p.IGR_SUB_ITERS > 0) ? p.IGR_SUB_ITERS : 1000;

            while (diff > p.IGR_SUB_ITER_TOL && sub < max_sub) {
                compute_igr_parabolic_rhs();

                double total_diff = 0.0;
                long total_pts = 0;

                #pragma omp parallel for reduction(+:total_diff,total_pts)
                for (size_t i = 0; i < cells.size(); ++i) {
                    Cell* c = cells[i];
                    if (p.ENABLE_MULTIRATE && !c->element_active) continue;
                    double dt_sub_c = c->element_dt / n_sub;
                    const size_t N_sig = c->sigma_field.size();
                    total_pts += N_sig;
                    for (size_t k = 0; k < N_sig; ++k) {
                        double old_sig = c->sigma_field[k];
                        double new_sig = old_sig + dt_sub_c * c->sigma_RHS[k];

                        if (new_sig < 0.0) {
                            new_sig = 0.0;
                        } else {
                            if (p.USE_PRESSURE_FIELD_CAP) {
                                int iy = k / p.N_PTS;
                                int ix = k % p.N_PTS;
                                double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
                                double rhou = c->get_U(1, iy, ix, p.N_PTS);
                                double rhov = c->get_U(2, iy, ix, p.N_PTS);
                                double E = c->get_U(3, iy, ix, p.N_PTS);
                                double press = (p.GAMMA - 1.0) * (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
                                if (press < p.POS_LIMITER_EPS) press = p.POS_LIMITER_EPS;
                                if (new_sig > press) new_sig = press;
                            }
                        }

                        total_diff += std::abs(new_sig - old_sig);
                        c->sigma_field[k] = new_sig;
                    }
                }

                diff = total_pts > 0 ? (total_diff / total_pts) : 0.0;
                sub++;
            }
        } else {
            if (n_sub == 1) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < cells.size(); ++i) {
                    Cell* c = cells[i];
                    if (p.ENABLE_MULTIRATE && !c->element_active) continue;
                    const double dt_c = c->element_dt;
                    const size_t N_sig = c->sigma_field.size();
                    for (size_t k = 0; k < N_sig; ++k) {
                        c->sigma_field[k] = alpha * c->sigma_old[k] +
                                            beta * (c->sigma_field[k] + dt_c * c->sigma_RHS[k]);
                    }
                }
                clamp_sigma_all();
            } else {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < cells.size(); ++i) {
                    Cell* c = cells[i];
                    if (p.ENABLE_MULTIRATE && !c->element_active) continue;
                    double dt_sub_c = c->element_dt / n_sub;
                    const size_t N_sig = c->sigma_field.size();
                    for (size_t k = 0; k < N_sig; ++k) {
                        c->sigma_field[k] = alpha * c->sigma_old[k] +
                                            beta * (c->sigma_field[k] + dt_sub_c * c->sigma_RHS[k]);
                    }
                }
                clamp_sigma_all();

                for (int sub = 1; sub < n_sub; ++sub) {
                    compute_igr_parabolic_rhs();
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < cells.size(); ++i) {
                        Cell* c = cells[i];
                        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
                        double dt_sub_c = c->element_dt / n_sub;
                        const size_t N_sig = c->sigma_field.size();
                        for (size_t k = 0; k < N_sig; ++k) {
                            c->sigma_field[k] += dt_sub_c * c->sigma_RHS[k];
                        }
                    }
                    clamp_sigma_all();
                }
            }
        }
    };

    // =====================================================================
    // Stage 1
    // =====================================================================
    compute_rhs();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        const double dt_stage = c->element_dt;
        for (size_t k = 0; k < c->U.size(); ++k) {
            c->U[k] = c->U_old[k] + dt_stage * c->RHS[k];
        }
    }

    relax_phantom_pressure(1.0, 0.0, 1.0);

    if (is_parabolic)
        sub_iterate_sigma_all(0.0, 1.0);

    if (p.ENABLE_IB && p.IB_METHOD == "VPM_ANALYTICAL") {
        apply_ib_analytical(1.0);
    }

    if (p.ENABLE_POS_LIMITER) {
        add_stats(Limiters::apply_positivity_limiter(cells, basis, p));
    }
    if (p.ENABLE_ENTROPY_LIMITER)
        add_stats(Limiters::apply_entropy_limiter(*this));
    check_stability();

    // =====================================================================
    // Stage 2
    // =====================================================================
    if (p.ENABLE_IB && p.ib_is_dynamic) {
        update_ib_mask_field(current_time + dt);
    }
    compute_rhs();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        const double dt_stage = c->element_dt;
        for (size_t k = 0; k < c->U.size(); ++k) {
            c->U[k] = 0.75 * c->U_old[k] + 0.25 * (c->U[k] + dt_stage * c->RHS[k]);
        }
    }

    relax_phantom_pressure(0.25, 0.75, 0.25);

    if (is_parabolic)
        sub_iterate_sigma_all(0.75, 0.25);

    if (p.ENABLE_IB && p.IB_METHOD == "VPM_ANALYTICAL") {
        apply_ib_analytical(0.25);
    }

    if (p.ENABLE_POS_LIMITER) {
        add_stats(Limiters::apply_positivity_limiter(cells, basis, p));
    }
    if (p.ENABLE_ENTROPY_LIMITER)
        add_stats(Limiters::apply_entropy_limiter(*this));
    check_stability();

    // =====================================================================
    // Stage 3
    // =====================================================================
    if (p.ENABLE_IB && p.ib_is_dynamic) {
        update_ib_mask_field(current_time + 0.5 * dt);
    }
    compute_rhs();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        const double dt_stage = c->element_dt;
        for (size_t k = 0; k < c->U.size(); ++k) {
            c->U[k] = (1.0 / 3.0) * c->U_old[k] + (2.0 / 3.0) * (c->U[k] + dt_stage * c->RHS[k]);
        }
    }

    relax_phantom_pressure(2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0);

    if (is_parabolic)
        sub_iterate_sigma_all(1.0 / 3.0, 2.0 / 3.0);

    if (p.ENABLE_IB && p.IB_METHOD == "VPM_ANALYTICAL") {
        apply_ib_analytical(2.0 / 3.0);
    }

    if (p.ENABLE_POS_LIMITER) {
        add_stats(Limiters::apply_positivity_limiter(cells, basis, p));
    }
    if (p.ENABLE_ENTROPY_LIMITER)
        add_stats(Limiters::apply_entropy_limiter(*this));
    check_stability();

    // =====================================================================
    // Multirate finalization and post-sync
    // =====================================================================
    if (p.ENABLE_MULTIRATE) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell* c = cells[i];
            if (c->element_active) {
                c->element_time += c->element_dt;
            }
        }
        
        // Re-apply limiters on final updated state
        if (p.ENABLE_POS_LIMITER) {
            add_stats(Limiters::apply_positivity_limiter(cells, basis, p));
        }
        if (p.ENABLE_ENTROPY_LIMITER)
            add_stats(Limiters::apply_entropy_limiter(*this));
        check_stability();
    }

    current_time += dt;
    if (p.ENABLE_IB && p.ib_is_dynamic) {
        update_ib_mask_field(current_time);
    }
}

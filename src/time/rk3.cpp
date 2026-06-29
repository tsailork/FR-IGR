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
        for (Cell* c : cells) {
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
        for (Cell* c : cells) {
            c->element_active = true;
        }
    }

    // Save previous stage states in cell-local U_old and sigma_old buffers
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        c->U_old = c->U;
        c->sigma_old = c->sigma_field;
    }

    if (p.ENABLE_IB && p.ib_is_dynamic) {
        update_ib_mask_field(current_time);
    }

    const bool is_parabolic = (p.ENABLE_IGR && p.IGR_TYPE == "PARABOLIC");
    const int n_sub = std::max(1, p.IGR_SUB_ITERS);
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
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int idx = iy * p.N_PTS + ix;
                    if (c->sigma_field[idx] < 0.0) {
                        c->sigma_field[idx] = 0.0;
                        continue;
                    }
                    double rho = std::max(1e-14, c->get_U(0, iy, ix, p.N_PTS));
                    double rhou = c->get_U(1, iy, ix, p.N_PTS);
                    double rhov = c->get_U(2, iy, ix, p.N_PTS);
                    double E = c->get_U(3, iy, ix, p.N_PTS);
                    double press = (p.GAMMA - 1.0) *
                                   (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
                    if (press < 1e-14)
                        press = 1e-14;

                    c->sigma_field[idx] = std::min(c->sigma_field[idx], press);
                }
            }
        }
    };

    auto sub_iterate_sigma_all = [&](double alpha, double beta) {
        if (n_sub == 1) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells.size(); ++i) {
                Cell* c = cells[i];
                const size_t N_sig = c->sigma_field.size();
                for (size_t k = 0; k < N_sig; ++k) {
                    c->sigma_field[k] = alpha * c->sigma_old[k] +
                                        beta * (c->sigma_field[k] + dt * c->sigma_RHS[k]);
                }
            }
            clamp_sigma_all();
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells.size(); ++i) {
                Cell* c = cells[i];
                const size_t N_sig = c->sigma_field.size();
                for (size_t k = 0; k < N_sig; ++k) {
                    c->sigma_field[k] = alpha * c->sigma_old[k] +
                                        beta * (c->sigma_field[k] + dt_sub * c->sigma_RHS[k]);
                }
            }
            clamp_sigma_all();

            for (int sub = 1; sub < n_sub; ++sub) {
                compute_igr_parabolic_rhs();
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < cells.size(); ++i) {
                    Cell* c = cells[i];
                    const size_t N_sig = c->sigma_field.size();
                    for (size_t k = 0; k < N_sig; ++k) {
                        c->sigma_field[k] += dt_sub * c->sigma_RHS[k];
                    }
                }
                clamp_sigma_all();
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
        if (p.ENABLE_MULTIRATE) {
            if (c->element_active) {
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] = c->U_old[k] + dt * c->RHS[k];
                }
            } else {
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] = c->U_old[k];
                    c->U_accum[k] += (1.0 / 6.0) * dt * c->RHS[k];
                }
            }
        } else {
            for (size_t k = 0; k < c->U.size(); ++k) {
                c->U[k] = c->U_old[k] + dt * c->RHS[k];
            }
        }
    }

    if (is_parabolic)
        sub_iterate_sigma_all(0.0, 1.0);

    if (p.ENABLE_IB && p.IB_METHOD == "VPM_ANALYTICAL") {
        apply_ib_analytical(dt);
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
        if (p.ENABLE_MULTIRATE) {
            if (c->element_active) {
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] = 0.75 * c->U_old[k] + 0.25 * (c->U[k] + dt * c->RHS[k]);
                }
            } else {
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] = c->U_old[k];
                    c->U_accum[k] += (1.0 / 6.0) * dt * c->RHS[k];
                }
            }
        } else {
            for (size_t k = 0; k < c->U.size(); ++k) {
                c->U[k] = 0.75 * c->U_old[k] + 0.25 * (c->U[k] + dt * c->RHS[k]);
            }
        }
    }

    if (is_parabolic)
        sub_iterate_sigma_all(0.75, 0.25);

    if (p.ENABLE_IB && p.IB_METHOD == "VPM_ANALYTICAL") {
        apply_ib_analytical(0.25 * dt);
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
        if (p.ENABLE_MULTIRATE) {
            if (c->element_active) {
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] = (1.0 / 3.0) * c->U_old[k] + (2.0 / 3.0) * (c->U[k] + dt * c->RHS[k]);
                }
            } else {
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] = c->U_old[k];
                    c->U_accum[k] += (2.0 / 3.0) * dt * c->RHS[k];
                }
            }
        } else {
            for (size_t k = 0; k < c->U.size(); ++k) {
                c->U[k] = (1.0 / 3.0) * c->U_old[k] +
                          (2.0 / 3.0) * (c->U[k] + dt * c->RHS[k]);
            }
        }
    }

    if (is_parabolic)
        sub_iterate_sigma_all(1.0 / 3.0, 2.0 / 3.0);

    if (p.ENABLE_IB && p.IB_METHOD == "VPM_ANALYTICAL") {
        apply_ib_analytical((2.0 / 3.0) * dt);
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
                for (size_t k = 0; k < c->U.size(); ++k) {
                    c->U[k] += c->U_accum[k];
                    c->U_accum[k] = 0.0;
                }
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

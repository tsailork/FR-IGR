/**
 * @file stability.cpp
 * @brief Stability error detection and CFL dynamic stability bounds on decoupled Cells.
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <iomanip>
#include <cstdlib>

void Solver::check_stability() const {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double rho  = c->get_U(0, iy, ix, p.N_PTS);
                double rhou = c->get_U(1, iy, ix, p.N_PTS);
                double rhov = c->get_U(2, iy, ix, p.N_PTS);
                double E    = c->get_U(3, iy, ix, p.N_PTS);
                double press = (p.GAMMA - 1.0) * (E - 0.5*(rhou*rhou + rhov*rhov)/rho);
                if (std::isnan(rho) || std::isnan(press) || rho <= 0.0 || press <= 0.0) {
                    #pragma omp critical
                    {
                        std::cerr << std::scientific << std::setprecision(15)
                                  << "\n[STABILITY ERROR] cell_index=" << i
                                  << " morton_id=" << c->morton_id
                                  << " elem=(" << c->ex << "," << c->ey << ") block=" << c->block_id
                                  << " node=(" << ix << "," << iy << ")"
                                  << "\n  rho  = " << rho
                                  << "\n  rhou = " << rhou
                                  << "\n  rhov = " << rhov
                                  << "\n  E    = " << E
                                  << "\n  p    = " << press << "\n";
                        exit(1);
                    }
                }
            }
        }
    }
}

double Solver::compute_dt() const {
    double min_dt = 1e30;

    #pragma omp parallel for reduction(min:min_dt) schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        double max_lambda = 1e-10;
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double rho = std::max(1e-12, c->get_U(0, iy, ix, p.N_PTS));
                double u   = c->get_U(1, iy, ix, p.N_PTS) / rho;
                double v   = c->get_U(2, iy, ix, p.N_PTS) / rho;
                double press = (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, p.N_PTS) - 0.5 * rho * (u * u + v * v));
                double press_safe = press;
                if (p.ENABLE_PPR) {
                    double theta_cfl = (p.PPR_ADAPTIVE_THETA) ? c->theta_avg : p.PPR_THETA;
                    double p_phan = c->S_field[iy * p.N_PTS + ix] / rho;
                    double p_reg = press + theta_cfl * (press - p_phan);
                    press_safe = std::max(press, p_reg);
                }
                if (press_safe < 1e-12) press_safe = 1e-12;
                double sound_speed = std::sqrt(p.GAMMA * press_safe / rho);
                max_lambda = std::max({max_lambda, std::abs(u) + sound_speed, std::abs(v) + sound_speed});
                if (p.ENABLE_PPR) {
                    double s_wave_x = p.PPR_ADV_MULT * (std::abs(u) + p.PPR_GRAD_ADV_SCALE * sound_speed);
                    double s_wave_y = p.PPR_ADV_MULT * (std::abs(v) + p.PPR_GRAD_ADV_SCALE * sound_speed);
                    max_lambda = std::max({max_lambda, s_wave_x, s_wave_y});
                }
            }
        }

        double h = std::min(c->dx, c->dy);
        double dt_conv = 0.5 * p.CFL * h / (max_lambda * (p.P_DEG + 1) * (p.P_DEG + 1));
        double dt_cell = dt_conv;

        if (p.ENABLE_IGR && p.IGR_TYPE == "PARABOLIC") {
            if (p.IGR_SUB_ITERS > 0) {
                double alpha_safe = std::max(1e-10, p.ALPHA_SCALE);
                double dt_diff  = 0.5 * p.IGR_TAU_R / (alpha_safe * (1.0 + p.IGR_BR2_ETA) * (2 * p.P_DEG + 1) * (2 * p.P_DEG + 1));
                double dt_relax = 0.5 * p.IGR_TAU_R;
                double dt_limit = std::min(dt_diff, dt_relax);
                dt_cell = std::min(dt_conv, p.IGR_SUB_ITERS * dt_limit);
            }
        }

        if (p.ENABLE_NS) {
            double nu = 1.0 / p.RE;
            double h2 = h * h;
            double denom = (2 * p.P_DEG + 1) * (2 * p.P_DEG + 1);
            double dt_visc = 0.25 * p.CFL * h2 / (nu * denom);
            dt_cell = std::min(dt_cell, dt_visc);
        }

        min_dt = std::min(min_dt, dt_cell);
    }
    return min_dt;
}

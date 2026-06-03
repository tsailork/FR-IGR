/**
 * @file ib_vpm.cpp
 * @brief Implements Volume Penalization Method (VPM) routines using cached block masks.
 */

#include "ib.hpp"
#include "../core/solver.hpp"
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::apply_ib_explicit() {
    if (!p.ENABLE_IB) return;

    for (auto &b : blocks) {
        #pragma omp for collapse(2) schedule(static)
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        int idx = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                        double chi = b.ib_mask[idx];
                        if (chi <= 0.0) continue;

                        double rho = b.U(0, ey, ex, iy, ix);
                        double rhou = b.U(1, ey, ex, iy, ix);
                        double rhov = b.U(2, ey, ex, iy, ix);
                        double E = b.U(3, ey, ex, iy, ix);

                        // Solid state targets (static)
                        double u_s = p.IB_VELOCITY_X;
                        double v_s = p.IB_VELOCITY_Y;

                        double target_rhou = rho * u_s;
                        double target_rhov = rho * v_s;
                        double target_E = 0.0;

                        if (p.IB_THERMAL_TYPE == "ADIABATIC") {
                            // Adiabatic VPM: penalize local velocity to solid velocity, leaving internal energy unchanged
                            double u_local = rhou / std::max(1e-12, rho);
                            double v_local = rhov / std::max(1e-12, rho);
                            target_E = E - 0.5 * rho * (u_local * u_local + v_local * v_local)
                                         + 0.5 * rho * (u_s * u_s + v_s * v_s);
                        } else {
                            // Isothermal VPM: penalize temperature to target wall temperature
                            target_E = rho * p.IB_TEMPERATURE / (p.GAMMA - 1.0)
                                     + 0.5 * rho * (u_s * u_s + v_s * v_s);
                        }

                        // S_IB = -chi/eta * (U - U_target)
                        double coeff = -chi / p.IB_PENALIZATION_ETA;
                        b.RHS(1, ey, ex, iy, ix) += coeff * (rhou - target_rhou);
                        b.RHS(2, ey, ex, iy, ix) += coeff * (rhov - target_rhov);
                        b.RHS(3, ey, ex, iy, ix) += coeff * (E - target_E);
                    }
                }
            }
        }
    }
}

void Solver::apply_ib_analytical(double dt_stage) {
    if (!p.ENABLE_IB) return;

    for (auto &b : blocks) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        int idx = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                        double chi = b.ib_mask[idx];
                        if (chi <= 0.0) continue;

                        double rho = b.U(0, ey, ex, iy, ix);
                        double rhou = b.U(1, ey, ex, iy, ix);
                        double rhov = b.U(2, ey, ex, iy, ix);
                        double E = b.U(3, ey, ex, iy, ix);

                        // Solid state targets (static)
                        double u_s = p.IB_VELOCITY_X;
                        double v_s = p.IB_VELOCITY_Y;

                        double target_rhou = rho * u_s;
                        double target_rhov = rho * v_s;
                        double target_E = 0.0;

                        if (p.IB_THERMAL_TYPE == "ADIABATIC") {
                            // Adiabatic VPM: penalize local velocity to solid velocity, leaving internal energy unchanged
                            double u_local = rhou / std::max(1e-12, rho);
                            double v_local = rhov / std::max(1e-12, rho);
                            target_E = E - 0.5 * rho * (u_local * u_local + v_local * v_local)
                                         + 0.5 * rho * (u_s * u_s + v_s * v_s);
                        } else {
                            // Isothermal VPM: penalize temperature to target wall temperature
                            target_E = rho * p.IB_TEMPERATURE / (p.GAMMA - 1.0)
                                     + 0.5 * rho * (u_s * u_s + v_s * v_s);
                        }

                        // e^(-chi * dt_stage / eta)
                        double factor = std::exp(-chi * dt_stage / p.IB_PENALIZATION_ETA);

                        // Apply penalization analytically
                        b.U(1, ey, ex, iy, ix) = target_rhou + factor * (rhou - target_rhou);
                        b.U(2, ey, ex, iy, ix) = target_rhov + factor * (rhov - target_rhov);
                        b.U(3, ey, ex, iy, ix) = target_E + factor * (E - target_E);
                    }
                }
            }
        }
    }
}

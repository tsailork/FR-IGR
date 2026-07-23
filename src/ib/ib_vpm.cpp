/**
 * @file ib_vpm.cpp
 * @brief Implements Volume Penalization Method (VPM) routines on decoupled Cells.
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

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                int idx = iy * p.N_PTS + ix;
                double chi = c->ib_mask[idx];
                chi = std::max(0.0, chi - 0.5);
                if (chi <= 0.0) continue;

                double rho = c->get_U(0, iy, ix, p.N_PTS);
                double rhou = c->get_U(1, iy, ix, p.N_PTS);
                double rhov = c->get_U(2, iy, ix, p.N_PTS);
                double E = c->get_U(3, iy, ix, p.N_PTS);

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
                c->get_RHS(1, iy, ix, p.N_PTS) += coeff * (rhou - target_rhou);
                c->get_RHS(2, iy, ix, p.N_PTS) += coeff * (rhov - target_rhov);
                c->get_RHS(3, iy, ix, p.N_PTS) += coeff * (E - target_E);
            }
        }
    }
}

void Solver::apply_ib_analytical(double dt_stage_ratio) {
    if (!p.ENABLE_IB) return;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        double dt_stage = dt_stage_ratio * c->element_dt;
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                int idx = iy * p.N_PTS + ix;
                double chi = c->ib_mask[idx];
                chi = std::max(0.0, chi - 0.5);
                if (chi <= 0.0) continue;

                double rho = c->get_U(0, iy, ix, p.N_PTS);
                double rhou = c->get_U(1, iy, ix, p.N_PTS);
                double rhov = c->get_U(2, iy, ix, p.N_PTS);
                double E = c->get_U(3, iy, ix, p.N_PTS);

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
                c->get_U(1, iy, ix, p.N_PTS) = target_rhou + factor * (rhou - target_rhou);
                c->get_U(2, iy, ix, p.N_PTS) = target_rhov + factor * (rhov - target_rhov);
                c->get_U(3, iy, ix, p.N_PTS) = target_E + factor * (E - target_E);
            }
        }
    }
}

void SolverDim<3>::apply_ib_explicit() {
    if (!p.ENABLE_IB) return;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        int npts = p.N_PTS;
        int npts3 = npts * npts * npts;

        for (int iz = 0; iz < npts; ++iz) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int idx = iz * npts * npts + iy * npts + ix;
                    double chi = c->ib_mask[idx];
                    chi = std::max(0.0, chi - 0.5);
                    if (chi <= 0.0) continue;

                    double rho  = c->U[0 * npts3 + idx];
                    double rhou = c->U[1 * npts3 + idx];
                    double rhov = c->U[2 * npts3 + idx];
                    double rhow = c->U[3 * npts3 + idx];
                    double E    = c->U[4 * npts3 + idx];

                    double u_s = p.IB_VELOCITY_X;
                    double v_s = p.IB_VELOCITY_Y;
                    double w_s = 0.0;

                    double target_rhou = rho * u_s;
                    double target_rhov = rho * v_s;
                    double target_rhow = rho * w_s;
                    double target_E = 0.0;

                    if (p.IB_THERMAL_TYPE == "ADIABATIC") {
                        double u_local = rhou / std::max(1e-12, rho);
                        double v_local = rhov / std::max(1e-12, rho);
                        double w_local = rhow / std::max(1e-12, rho);
                        target_E = E - 0.5 * rho * (u_local * u_local + v_local * v_local + w_local * w_local)
                                     + 0.5 * rho * (u_s * u_s + v_s * v_s + w_s * w_s);
                    } else {
                        target_E = rho * p.IB_TEMPERATURE / (p.GAMMA - 1.0)
                                 + 0.5 * rho * (u_s * u_s + v_s * v_s + w_s * w_s);
                    }

                    double coeff = -chi / p.IB_PENALIZATION_ETA;
                    c->RHS[1 * npts3 + idx] += coeff * (rhou - target_rhou);
                    c->RHS[2 * npts3 + idx] += coeff * (rhov - target_rhov);
                    c->RHS[3 * npts3 + idx] += coeff * (rhow - target_rhow);
                    c->RHS[4 * npts3 + idx] += coeff * (E - target_E);
                }
            }
        }
    }
}

void SolverDim<3>::apply_ib_analytical(double dt_stage_ratio) {
    if (!p.ENABLE_IB) return;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell3D* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        double dt_stage = dt_stage_ratio * c->element_dt;
        int npts = p.N_PTS;
        int npts3 = npts * npts * npts;

        for (int iz = 0; iz < npts; ++iz) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    int idx = iz * npts * npts + iy * npts + ix;
                    double chi = c->ib_mask[idx];
                    chi = std::max(0.0, chi - 0.5);
                    if (chi <= 0.0) continue;

                    double rho  = c->U[0 * npts3 + idx];
                    double rhou = c->U[1 * npts3 + idx];
                    double rhov = c->U[2 * npts3 + idx];
                    double rhow = c->U[3 * npts3 + idx];
                    double E    = c->U[4 * npts3 + idx];

                    double u_s = p.IB_VELOCITY_X;
                    double v_s = p.IB_VELOCITY_Y;
                    double w_s = 0.0;

                    double target_rhou = rho * u_s;
                    double target_rhov = rho * v_s;
                    double target_rhow = rho * w_s;
                    double target_E = 0.0;

                    if (p.IB_THERMAL_TYPE == "ADIABATIC") {
                        double u_local = rhou / std::max(1e-12, rho);
                        double v_local = rhov / std::max(1e-12, rho);
                        double w_local = rhow / std::max(1e-12, rho);
                        target_E = E - 0.5 * rho * (u_local * u_local + v_local * v_local + w_local * w_local)
                                     + 0.5 * rho * (u_s * u_s + v_s * v_s + w_s * w_s);
                    } else {
                        target_E = rho * p.IB_TEMPERATURE / (p.GAMMA - 1.0)
                                 + 0.5 * rho * (u_s * u_s + v_s * v_s + w_s * w_s);
                    }

                    double alpha = std::exp(- (chi / p.IB_PENALIZATION_ETA) * dt_stage);
                    c->U[1 * npts3 + idx] = target_rhou + alpha * (rhou - target_rhou);
                    c->U[2 * npts3 + idx] = target_rhov + alpha * (rhov - target_rhov);
                    c->U[3 * npts3 + idx] = target_rhow + alpha * (rhow - target_rhow);
                    c->U[4 * npts3 + idx] = target_E    + alpha * (E - target_E);
                }
            }
        }
    }
}

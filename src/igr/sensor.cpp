/// @file sensor.cpp
/// @brief Cao-Schaefer velocity-gradient sensor for IGR.
///
/// Computes  S = ε · 2(u_x² + v_y² + u_x·v_y + v_x·u_y)  at every
/// solution point.  Two modes are supported:
///   LOCAL     — element-interior gradients only.
///   CORRECTED — BR2-style interface-corrected gradients.
///
/// OpenMP: parallelised over elements (ey, ex).

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// Hardcoded thresholds for divergence and sensor magnitude filtering
static constexpr double DIVERGENCE_THRESHOLD = -0.0; // Must be compressive
static constexpr double SENSOR_THRESHOLD = 0.1;      // Threshold for the raw sensor 'val'

void Solver::compute_sensor_source() {
    const double epsilon = p.ALPHA_SCALE * (dx * dy);

    if (p.IGR_GRADIENT_TYPE == "CORRECTED") {
        // ==================================================================
        // CORRECTED gradient (BR2-style interface lifting)
        // ==================================================================
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                double du_dx[MAX_PTS * MAX_PTS], dv_dx[MAX_PTS * MAX_PTS];
                double du_dy[MAX_PTS * MAX_PTS], dv_dy[MAX_PTS * MAX_PTS];

                // ---- X-direction gradients ----
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double UL_face[4] = {}, UR_face[4] = {};
                    for (int k = 0; k < p.N_PTS; ++k)
                        for (int v = 0; v < 4; ++v) {
                            UL_face[v] += U(v, ey, ex, iy, k) * basis.l_L[k];
                            UR_face[v] += U(v, ey, ex, iy, k) * basis.l_R[k];
                        }

                    double U_neigh_L[4], U_neigh_R[4], sd;
                    get_neigh_state_x(ey, ex, iy, false, UL_face, 0.0, U_neigh_L, sd);
                    get_neigh_state_x(ey, ex, iy, true,  UR_face, 0.0, U_neigh_R, sd);

                    double rL  = clamp_positivity(UL_face[0]);
                    double rLn = clamp_positivity(U_neigh_L[0]);
                    double uL_hat = 0.5*(UL_face[1]/rL + U_neigh_L[1]/rLn);
                    double vL_hat = 0.5*(UL_face[2]/rL + U_neigh_L[2]/rLn);

                    double rR  = clamp_positivity(UR_face[0]);
                    double rRn = clamp_positivity(U_neigh_R[0]);
                    double uR_hat = 0.5*(UR_face[1]/rR + U_neigh_R[1]/rRn);
                    double vR_hat = 0.5*(UR_face[2]/rR + U_neigh_R[2]/rRn);

                    double uL_int = UL_face[1]/rL, vL_int = UL_face[2]/rL;
                    double uR_int = UR_face[1]/rR, vR_int = UR_face[2]/rR;

                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double dudx_l = 0, dvdx_l = 0;
                        for (int k = 0; k < p.N_PTS; ++k) {
                            double r_loc = clamp_positivity(U(0, ey, ex, iy, k));
                            dudx_l += basis.D[ix][k] * (U(1, ey, ex, iy, k)/r_loc);
                            dvdx_l += basis.D[ix][k] * (U(2, ey, ex, iy, k)/r_loc);
                        }
                        int idx = iy * p.N_PTS + ix;
                        du_dx[idx] = (dudx_l + (uL_int - uL_hat)*basis.dgl[ix]
                                              + (uR_hat - uR_int)*basis.dgr[ix]) * (2.0/dx);
                        dv_dx[idx] = (dvdx_l + (vL_int - vL_hat)*basis.dgl[ix]
                                              + (vR_hat - vR_int)*basis.dgr[ix]) * (2.0/dx);
                    }
                }

                // ---- Y-direction gradients ----
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double UB_face[4] = {}, UT_face[4] = {};
                    for (int k = 0; k < p.N_PTS; ++k)
                        for (int v = 0; v < 4; ++v) {
                            UB_face[v] += U(v, ey, ex, k, ix) * basis.l_L[k];
                            UT_face[v] += U(v, ey, ex, k, ix) * basis.l_R[k];
                        }

                    double U_neigh_B[4], U_neigh_T[4], sd;
                    get_neigh_state_y(ey, ex, ix, false, UB_face, 0.0, U_neigh_B, sd);
                    get_neigh_state_y(ey, ex, ix, true,  UT_face, 0.0, U_neigh_T, sd);

                    double rB  = clamp_positivity(UB_face[0]);
                    double rBn = clamp_positivity(U_neigh_B[0]);
                    double uB_hat = 0.5*(UB_face[1]/rB + U_neigh_B[1]/rBn);
                    double vB_hat = 0.5*(UB_face[2]/rB + U_neigh_B[2]/rBn);

                    double rT  = clamp_positivity(UT_face[0]);
                    double rTn = clamp_positivity(U_neigh_T[0]);
                    double uT_hat = 0.5*(UT_face[1]/rT + U_neigh_T[1]/rTn);
                    double vT_hat = 0.5*(UT_face[2]/rT + U_neigh_T[2]/rTn);

                    double uB_int = UB_face[1]/rB, vB_int = UB_face[2]/rB;
                    double uT_int = UT_face[1]/rT, vT_int = UT_face[2]/rT;

                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        double dudy_l = 0, dvdy_l = 0;
                        for (int k = 0; k < p.N_PTS; ++k) {
                            double r_loc = clamp_positivity(U(0, ey, ex, k, ix));
                            dudy_l += basis.D[iy][k] * (U(1, ey, ex, k, ix)/r_loc);
                            dvdy_l += basis.D[iy][k] * (U(2, ey, ex, k, ix)/r_loc);
                        }
                        int idx = iy * p.N_PTS + ix;
                        du_dy[idx] = (dudy_l + (uB_int - uB_hat)*basis.dgl[iy]
                                              + (uT_hat - uT_int)*basis.dgr[iy]) * (2.0/dy);
                        dv_dy[idx] = (dvdy_l + (vB_int - vB_hat)*basis.dgl[iy]
                                              + (vT_hat - vT_int)*basis.dgr[iy]) * (2.0/dy);
                    }
                }

                // ---- Assemble sensor ----
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        int idx = iy * p.N_PTS + ix;
                        double val = 2.0 * (du_dx[idx]*du_dx[idx] + dv_dy[idx]*dv_dy[idx]
                                          + du_dx[idx]*dv_dy[idx] + dv_dx[idx]*du_dy[idx]);
                        
                        double div_u = du_dx[idx] + dv_dy[idx];
                        
                        if (div_u < DIVERGENCE_THRESHOLD && val > SENSOR_THRESHOLD) {
                            S_buf[get_flat_idx(ey, ex, iy, ix)] = epsilon * val;
                        } else {
                            S_buf[get_flat_idx(ey, ex, iy, ix)] = 0.0;
                        }
                    }
                }
            }
        }

    } else {
        // ==================================================================
        // LOCAL gradient (element-interior only)
        // ==================================================================
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double du_dx = 0, du_dy = 0, dv_dx = 0, dv_dy = 0;

                        for (int k = 0; k < p.N_PTS; ++k) {
                            double rx = clamp_positivity(U(0, ey, ex, iy, k));
                            du_dx += basis.D[ix][k] * (U(1, ey, ex, iy, k)/rx) * (2.0/dx);
                            dv_dx += basis.D[ix][k] * (U(2, ey, ex, iy, k)/rx) * (2.0/dx);
                        }
                        for (int k = 0; k < p.N_PTS; ++k) {
                            double ry = clamp_positivity(U(0, ey, ex, k, ix));
                            du_dy += basis.D[iy][k] * (U(1, ey, ex, k, ix)/ry) * (2.0/dy);
                            dv_dy += basis.D[iy][k] * (U(2, ey, ex, k, ix)/ry) * (2.0/dy);
                        }

                        double val = 2.0 * (du_dx*du_dx + dv_dy*dv_dy
                                           + du_dx*dv_dy + dv_dx*du_dy);
                        
                        double div_u = du_dx + dv_dy;
                        
                        if (div_u < DIVERGENCE_THRESHOLD && val > SENSOR_THRESHOLD) {
                            S_buf[get_flat_idx(ey, ex, iy, ix)] = epsilon * val;
                        } else {
                            S_buf[get_flat_idx(ey, ex, iy, ix)] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

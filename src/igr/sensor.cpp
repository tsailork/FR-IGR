/**
 * @file sensor.cpp
 * @brief IGR source-term sensor with multiple toggleable approaches on decoupled Cells.
 */

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// Thresholds for divergence and sensor magnitude filtering
static constexpr double DIVERGENCE_THRESHOLD = 1.0e99; // Must be compressive
static constexpr double SENSOR_THRESHOLD = -9.0e99;    // Threshold for raw sensor 'val'

void Solver::compute_sensor_source() {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        const double epsilon = p.ALPHA_SCALE * (c->dx * c->dy);

        double u_loc[MAX_PTS * MAX_PTS];
        double v_loc[MAX_PTS * MAX_PTS];
        for (int k = 0; k < p.N_PTS * p.N_PTS; ++k) {
            int iy = k / p.N_PTS;
            int ix = k % p.N_PTS;
            double rho = std::max(1e-12, c->get_U(0, iy, ix, p.N_PTS));
            u_loc[k] = c->get_U(1, iy, ix, p.N_PTS) / rho;
            v_loc[k] = c->get_U(2, iy, ix, p.N_PTS) / rho;
        }

        if (p.IGR_GRADIENT_TYPE == "CORRECTED") {
            double du_dx[MAX_PTS * MAX_PTS], dv_dx[MAX_PTS * MAX_PTS];
            double du_dy[MAX_PTS * MAX_PTS], dv_dy[MAX_PTS * MAX_PTS];
            double dru_dx[MAX_PTS * MAX_PTS], drv_dy[MAX_PTS * MAX_PTS];

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double UL_face[4] = {}, UR_face[4] = {};
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v) {
                        UL_face[v] += c->get_U(v, iy, k, p.N_PTS) * basis.l_L[k];
                        UR_face[v] += c->get_U(v, iy, k, p.N_PTS) * basis.l_R[k];
                    }
                }

                double U_neigh_L[4], U_neigh_R[4], sd;
                
                // Left neighbor extrapolation
                if (c->neighbors[0]) {
                    Cell* nc = c->neighbors[0];
                    char nface = c->neighbor_faces[0];
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    for (int v = 0; v < 4; ++v) U_neigh_L[v] = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        for (int v = 0; v < 4; ++v)
                            U_neigh_L[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    }
                } else {
                    get_neigh_state_cell(*c, iy, false, UL_face, 0.0, U_neigh_L, sd, 0);
                }

                // Right neighbor extrapolation
                if (c->neighbors[1]) {
                    Cell* nc = c->neighbors[1];
                    char nface = c->neighbor_faces[1];
                    const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                    for (int v = 0; v < 4; ++v) U_neigh_R[v] = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        for (int v = 0; v < 4; ++v)
                            U_neigh_R[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    }
                } else {
                    get_neigh_state_cell(*c, iy, true, UR_face, 0.0, U_neigh_R, sd, 0);
                }

                double rL = std::max(1e-12, UL_face[0]);
                double rLn = std::max(1e-12, U_neigh_L[0]);
                double uL_hat = 0.5 * (UL_face[1] / rL + U_neigh_L[1] / rLn);
                double vL_hat = 0.5 * (UL_face[2] / rL + U_neigh_L[2] / rLn);

                double rR = std::max(1e-12, UR_face[0]);
                double rRn = std::max(1e-12, U_neigh_R[0]);
                double uR_hat = 0.5 * (UR_face[1] / rR + U_neigh_R[1] / rRn);
                double vR_hat = 0.5 * (UR_face[2] / rR + U_neigh_R[2] / rRn);

                double uL_int = UL_face[1] / rL, vL_int = UL_face[2] / rL;
                double uR_int = UR_face[1] / rR, vR_int = UR_face[2] / rR;

                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double dudx_l = 0, dvdx_l = 0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        int kidx = iy * p.N_PTS + k;
                        dudx_l += basis.D[ix][k] * u_loc[kidx];
                        dvdx_l += basis.D[ix][k] * v_loc[kidx];
                    }
                    int idx = iy * p.N_PTS + ix;
                    du_dx[idx] = (dudx_l + (uL_int - uL_hat) * basis.dgl[ix] +
                                  (uR_hat - uR_int) * basis.dgr[ix]) *
                                 (2.0 / c->dx);
                    dv_dx[idx] = (dvdx_l + (vL_int - vL_hat) * basis.dgl[ix] +
                                  (vR_hat - vR_int) * basis.dgr[ix]) *
                                 (2.0 / c->dx);

                    if (p.USE_MOMENTUM_DIV) {
                        double drudx = 0;
                        for (int k = 0; k < p.N_PTS; ++k)
                            drudx += basis.D[ix][k] * c->get_U(1, iy, k, p.N_PTS);
                        dru_dx[idx] = drudx * (2.0 / c->dx);
                    }
                }
            }

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double UB_face[4] = {}, UT_face[4] = {};
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v) {
                        UB_face[v] += c->get_U(v, k, ix, p.N_PTS) * basis.l_L[k];
                        UT_face[v] += c->get_U(v, k, ix, p.N_PTS) * basis.l_R[k];
                    }
                }

                double U_neigh_B[4], U_neigh_T[4], sd;
                
                // Bottom neighbor extrapolation
                if (c->neighbors[2]) {
                    Cell* nc = c->neighbors[2];
                    char nface = c->neighbor_faces[2];
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    for (int v = 0; v < 4; ++v) U_neigh_B[v] = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        for (int v = 0; v < 4; ++v)
                            U_neigh_B[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    }
                } else {
                    get_neigh_state_cell(*c, ix, false, UB_face, 0.0, U_neigh_B, sd, 1);
                }

                // Top neighbor extrapolation
                if (c->neighbors[3]) {
                    Cell* nc = c->neighbors[3];
                    char nface = c->neighbor_faces[3];
                    const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                    for (int v = 0; v < 4; ++v) U_neigh_T[v] = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        for (int v = 0; v < 4; ++v)
                            U_neigh_T[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    }
                } else {
                    get_neigh_state_cell(*c, ix, true, UT_face, 0.0, U_neigh_T, sd, 1);
                }

                double rB = std::max(1e-12, UB_face[0]);
                double rBn = std::max(1e-12, U_neigh_B[0]);
                double uB_hat = 0.5 * (UB_face[1] / rB + U_neigh_B[1] / rBn);
                double vB_hat = 0.5 * (UB_face[2] / rB + U_neigh_B[2] / rBn);

                double rT = std::max(1e-12, UT_face[0]);
                double rTn = std::max(1e-12, U_neigh_T[0]);
                double uT_hat = 0.5 * (UT_face[1] / rT + U_neigh_T[1] / rTn);
                double vT_hat = 0.5 * (UT_face[2] / rT + U_neigh_T[2] / rTn);

                double uB_int = UB_face[1] / rB, vB_int = UB_face[2] / rB;
                double uT_int = UT_face[1] / rT, vT_int = UT_face[2] / rT;

                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double dudy_l = 0, dvdy_l = 0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        int kidx = k * p.N_PTS + ix;
                        dudy_l += basis.D[iy][k] * u_loc[kidx];
                        dvdy_l += basis.D[iy][k] * v_loc[kidx];
                    }
                    int idx = iy * p.N_PTS + ix;
                    du_dy[idx] = (dudy_l + (uB_int - uB_hat) * basis.dgl[iy] +
                                  (uT_hat - uT_int) * basis.dgr[iy]) *
                                 (2.0 / c->dy);
                    dv_dy[idx] = (dvdy_l + (vB_int - vB_hat) * basis.dgl[iy] +
                                  (vT_hat - vT_int) * basis.dgr[iy]) *
                                 (2.0 / c->dy);

                    if (p.USE_MOMENTUM_DIV) {
                        double drvdy = 0;
                        for (int k = 0; k < p.N_PTS; ++k)
                            drvdy += basis.D[iy][k] * c->get_U(2, k, ix, p.N_PTS);
                        drv_dy[idx] = drvdy * (2.0 / c->dy);
                    }
                }
            }

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int idx = iy * p.N_PTS + ix;
                    double val =
                        2.0 * (du_dx[idx] * du_dx[idx] + dv_dy[idx] * dv_dy[idx] +
                               du_dx[idx] * dv_dy[idx] + dv_dx[idx] * du_dy[idx]);
                    double compression = p.USE_MOMENTUM_DIV
                                             ? (dru_dx[idx] + drv_dy[idx])
                                             : (du_dx[idx] + dv_dy[idx]);
                    double ducros = 1.0;
                    if (p.USE_DUCROS_SWITCH) {
                        double div_u = du_dx[idx] + dv_dy[idx];
                        double omega = dv_dx[idx] - du_dy[idx];
                        ducros = (div_u * div_u) / (div_u * div_u + omega * omega + 1e-30);
                    }
                    double source_val = 0.0;
                    if (compression < DIVERGENCE_THRESHOLD && val > SENSOR_THRESHOLD) {
                        source_val = epsilon * val * ducros;
                        if (p.USE_PRESSURE_SOURCE_CAP) {
                            double rho = std::max(1e-12, c->get_U(0, iy, ix, p.N_PTS));
                            double u = u_loc[iy * p.N_PTS + ix];
                            double v = v_loc[iy * p.N_PTS + ix];
                            double E = c->get_U(3, iy, ix, p.N_PTS);
                            double press = std::max(1e-14, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)));
                            source_val = std::min(source_val, press * p.SOURCE_CAP_COEFF);
                        }
                    }
                    c->S_buf[idx] = source_val;
                }
            }
        } else {
            // LOCAL mode
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double du_dx = 0, du_dy = 0, dv_dx = 0, dv_dy = 0;
                    for (int k = 0; k < p.N_PTS; ++k) {
                        int kidx = iy * p.N_PTS + k;
                        du_dx += basis.D[ix][k] * u_loc[kidx] * (2.0 / c->dx);
                        dv_dx += basis.D[ix][k] * v_loc[kidx] * (2.0 / c->dx);
                    }
                    for (int k = 0; k < p.N_PTS; ++k) {
                        int kidx = k * p.N_PTS + ix;
                        du_dy += basis.D[iy][k] * u_loc[kidx] * (2.0 / c->dy);
                        dv_dy += basis.D[iy][k] * v_loc[kidx] * (2.0 / c->dy);
                    }
                    double val = 2.0 * (du_dx * du_dx + dv_dy * dv_dy +
                                        du_dx * dv_dy + dv_dx * du_dy);
                    double compression = du_dx + dv_dy;
                    if (p.USE_MOMENTUM_DIV) {
                        double drudx = 0, drvdy = 0;
                        for (int k = 0; k < p.N_PTS; ++k)
                            drudx += basis.D[ix][k] * c->get_U(1, iy, k, p.N_PTS) * (2.0 / c->dx);
                        for (int k = 0; k < p.N_PTS; ++k)
                            drvdy += basis.D[iy][k] * c->get_U(2, k, ix, p.N_PTS) * (2.0 / c->dy);
                        compression = drudx + drvdy;
                    }
                    double ducros = 1.0;
                    if (p.USE_DUCROS_SWITCH) {
                        double div_u = du_dx + dv_dy;
                        double omega = dv_dx - du_dy;
                        ducros = (div_u * div_u) / (div_u * div_u + omega * omega + 1e-30);
                    }
                    double source_val = 0.0;
                    if (compression < DIVERGENCE_THRESHOLD && val > SENSOR_THRESHOLD) {
                        source_val = epsilon * val * ducros;
                        if (p.USE_PRESSURE_SOURCE_CAP) {
                            double rho = std::max(1e-12, c->get_U(0, iy, ix, p.N_PTS));
                            double u = u_loc[iy * p.N_PTS + ix];
                            double v = v_loc[iy * p.N_PTS + ix];
                            double E = c->get_U(3, iy, ix, p.N_PTS);
                            double press = std::max(1e-14, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)));
                            source_val = std::min(source_val, press * p.SOURCE_CAP_COEFF);
                        }
                    }
                    c->S_buf[iy * p.N_PTS + ix] = source_val;
                }
            }
        }
    }
}

/// @file sensor.cpp
/// @brief IGR source-term sensor with multiple toggleable approaches.
///
/// Base sensor: Cao-Schaefer velocity-gradient norm
///   S = ε · 2(u_x² + v_y² + u_x·v_y + v_x·u_y)
///
/// Toggleable modifications (compile-time switches):
///   USE_DUCROS_SWITCH    — multiply by Ducros ratio to suppress vortical noise
///   USE_PRESSURE_SENSOR  — replace velocity sensor with |∇p|²-based sensor
///   USE_MOMENTUM_DIV     — use ∇·(ρu) instead of ∇·u for compression check
///
/// Modes:
///   LOCAL     — element-interior gradients only.
///   CORRECTED — BR2-style interface-corrected gradients.
///
/// OpenMP: parallelised over elements (ey, ex).

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// ==========================================================================
// Toggleable switches — set to true/false to enable/disable each approach
// ==========================================================================

/// Approach A: Ducros switch.  Multiplies the sensor by
///   F_D = (∇·u)² / ((∇·u)² + |ω|² + ε_machine)
/// where ω = ∂v/∂x − ∂u/∂y.  Suppresses the sensor in regions dominated
/// by vorticity / shear (e.g. aliased contacts) while preserving signal
/// at irrotational shocks.
static constexpr bool USE_DUCROS_SWITCH = false;

/// Approach B: Pressure-gradient sensor.  Replaces the velocity-gradient
/// sensor with  S = ε · |∇p|² / p_ref²  where p_ref = max(p_local, ε).
/// Contacts have continuous pressure → zero signal.
static constexpr bool USE_PRESSURE_SENSOR = false;

/// Approach E: Use momentum divergence ∇·(ρu) for the compression check
/// instead of velocity divergence ∇·u.  Avoids the 1/ρ division that
/// amplifies density aliasing at contacts.
static constexpr bool USE_MOMENTUM_DIV = false;

/// Approach F: Pressure-based Source Cap. Caps the computed IGR source term 
/// (S_buf) to be no larger than C * p_local. This restricts the maximum injection
/// of artificial viscosity at shocks while allowing the Helmholtz smoothing to 
/// naturally diffuse the viscosity ahead of the shock wave.
static constexpr bool USE_PRESSURE_SOURCE_CAP = true;
static constexpr double SOURCE_CAP_COEFF = 1.0;

// Thresholds for divergence and sensor magnitude filtering
static constexpr double DIVERGENCE_THRESHOLD = 1.0e99; // Must be compressive
static constexpr double SENSOR_THRESHOLD =
    -9.0e99; // 0.1; // Threshold for raw sensor 'val'

void Solver::compute_sensor_source() {
  for (auto &b : blocks) {
    const double epsilon = p.ALPHA_SCALE * (b.dx * b.dy);

    if (p.IGR_GRADIENT_TYPE == "CORRECTED") {
#pragma omp for collapse(2) schedule(static)
      for (int ey = 0; ey < b.ny; ++ey) {
        for (int ex = 0; ex < b.nx; ++ex) {

          
          double du_dx[MAX_PTS * MAX_PTS], dv_dx[MAX_PTS * MAX_PTS];
          double du_dy[MAX_PTS * MAX_PTS], dv_dy[MAX_PTS * MAX_PTS];
          double dru_dx[MAX_PTS * MAX_PTS], drv_dy[MAX_PTS * MAX_PTS];

          for (int iy = 0; iy < p.N_PTS; ++iy) {
            double UL_face[4] = {}, UR_face[4] = {};
            for (int k = 0; k < p.N_PTS; ++k)
              for (int v = 0; v < 4; ++v) {
                UL_face[v] += b.U(v, ey, ex, iy, k) * basis.l_L[k];
                UR_face[v] += b.U(v, ey, ex, iy, k) * basis.l_R[k];
              }

            double U_neigh_L[4], U_neigh_R[4], sd;
            get_neigh_state_x(b, ey, ex, iy, false, UL_face, 0.0, U_neigh_L,
                              sd);
            get_neigh_state_x(b, ey, ex, iy, true, UR_face, 0.0, U_neigh_R,
                              sd);

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
                double r_loc = std::max(1e-12, b.U(0, ey, ex, iy, k));
                dudx_l += basis.D[ix][k] * (b.U(1, ey, ex, iy, k) / r_loc);
                dvdx_l += basis.D[ix][k] * (b.U(2, ey, ex, iy, k) / r_loc);
              }
              int idx = iy * p.N_PTS + ix;
              du_dx[idx] = (dudx_l + (uL_int - uL_hat) * basis.dgl[ix] +
                            (uR_hat - uR_int) * basis.dgr[ix]) *
                           (2.0 / b.dx);
              dv_dx[idx] = (dvdx_l + (vL_int - vL_hat) * basis.dgl[ix] +
                            (vR_hat - vR_int) * basis.dgr[ix]) *
                           (2.0 / b.dx);

              if (USE_MOMENTUM_DIV) {
                double drudx = 0;
                for (int k = 0; k < p.N_PTS; ++k)
                  drudx += basis.D[ix][k] * b.U(1, ey, ex, iy, k);
                dru_dx[idx] = drudx * (2.0 / b.dx);
              }
            }
          }

          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double UB_face[4] = {}, UT_face[4] = {};
            for (int k = 0; k < p.N_PTS; ++k)
              for (int v = 0; v < 4; ++v) {
                UB_face[v] += b.U(v, ey, ex, k, ix) * basis.l_L[k];
                UT_face[v] += b.U(v, ey, ex, k, ix) * basis.l_R[k];
              }

            double U_neigh_B[4], U_neigh_T[4], sd;
            get_neigh_state_y(b, ey, ex, ix, false, UB_face, 0.0, U_neigh_B,
                              sd);
            get_neigh_state_y(b, ey, ex, ix, true, UT_face, 0.0, U_neigh_T,
                              sd);

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
                double r_loc = std::max(1e-12, b.U(0, ey, ex, k, ix));
                dudy_l += basis.D[iy][k] * (b.U(1, ey, ex, k, ix) / r_loc);
                dvdy_l += basis.D[iy][k] * (b.U(2, ey, ex, k, ix) / r_loc);
              }
              int idx = iy * p.N_PTS + ix;
              du_dy[idx] = (dudy_l + (uB_int - uB_hat) * basis.dgl[iy] +
                            (uT_hat - uT_int) * basis.dgr[iy]) *
                           (2.0 / b.dy);
              dv_dy[idx] = (dvdy_l + (vB_int - vB_hat) * basis.dgl[iy] +
                            (vT_hat - vT_int) * basis.dgr[iy]) *
                           (2.0 / b.dy);

              if (USE_MOMENTUM_DIV) {
                double drvdy = 0;
                for (int k = 0; k < p.N_PTS; ++k)
                  drvdy += basis.D[iy][k] * b.U(2, ey, ex, k, ix);
                drv_dy[idx] = drvdy * (2.0 / b.dy);
              }
            }
          }

          for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
              int idx = iy * p.N_PTS + ix;
              double val =
                  2.0 * (du_dx[idx] * du_dx[idx] + dv_dy[idx] * dv_dy[idx] +
                         du_dx[idx] * dv_dy[idx] + dv_dx[idx] * du_dy[idx]);
              double compression = USE_MOMENTUM_DIV
                                       ? (dru_dx[idx] + drv_dy[idx])
                                       : (du_dx[idx] + dv_dy[idx]);
              double ducros = 1.0;
              if (USE_DUCROS_SWITCH) {
                double div_u = du_dx[idx] + dv_dy[idx];
                double omega = dv_dx[idx] - du_dy[idx];
                ducros =
                    (div_u * div_u) / (div_u * div_u + omega * omega + 1e-30);
              }
              double source_val = 0.0;
              if (compression < DIVERGENCE_THRESHOLD && val > SENSOR_THRESHOLD) {
                source_val = epsilon * val * ducros;
                if constexpr (USE_PRESSURE_SOURCE_CAP) {
                  double rho = std::max(1e-12, b.U(0, ey, ex, iy, ix));
                  double u = b.U(1, ey, ex, iy, ix) / rho;
                  double v = b.U(2, ey, ex, iy, ix) / rho;
                  double E = b.U(3, ey, ex, iy, ix);
                  double press = std::max(1e-14, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)));
                  source_val = std::min(source_val, press * SOURCE_CAP_COEFF);
                }
              }
              b.S_buf[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)] = source_val;
            }
          }
        }
      }
    } else {
#pragma omp for collapse(2) schedule(static)
      for (int ey = 0; ey < b.ny; ++ey) {
        for (int ex = 0; ex < b.nx; ++ex) {
          for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
              double du_dx = 0, du_dy = 0, dv_dx = 0, dv_dy = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                double rx = std::max(1e-12, b.U(0, ey, ex, iy, k));
                du_dx += basis.D[ix][k] * (b.U(1, ey, ex, iy, k) / rx) *
                         (2.0 / b.dx);
                dv_dx += basis.D[ix][k] * (b.U(2, ey, ex, iy, k) / rx) *
                         (2.0 / b.dx);
              }
              for (int k = 0; k < p.N_PTS; ++k) {
                double ry = std::max(1e-12, b.U(0, ey, ex, k, ix));
                du_dy += basis.D[iy][k] * (b.U(1, ey, ex, k, ix) / ry) *
                         (2.0 / b.dy);
                dv_dy += basis.D[iy][k] * (b.U(2, ey, ex, k, ix) / ry) *
                         (2.0 / b.dy);
              }
              double val = 2.0 * (du_dx * du_dx + dv_dy * dv_dy +
                                  du_dx * dv_dy + dv_dx * du_dy);
              double compression = du_dx + dv_dy;
              if (USE_MOMENTUM_DIV) {
                double drudx = 0, drvdy = 0;
                for (int k = 0; k < p.N_PTS; ++k)
                  drudx +=
                      basis.D[ix][k] * b.U(1, ey, ex, iy, k) * (2.0 / b.dx);
                for (int k = 0; k < p.N_PTS; ++k)
                  drvdy +=
                      basis.D[iy][k] * b.U(2, ey, ex, k, ix) * (2.0 / b.dy);
                compression = drudx + drvdy;
              }
              double ducros = 1.0;
              if (USE_DUCROS_SWITCH) {
                double div_u = du_dx + dv_dy;
                double omega = dv_dx - du_dy;
                ducros =
                    (div_u * div_u) / (div_u * div_u + omega * omega + 1e-30);
              }
              double source_val = 0.0;
              if (compression < DIVERGENCE_THRESHOLD && val > SENSOR_THRESHOLD) {
                source_val = epsilon * val * ducros;
                if constexpr (USE_PRESSURE_SOURCE_CAP) {
                  double rho = std::max(1e-12, b.U(0, ey, ex, iy, ix));
                  double u = b.U(1, ey, ex, iy, ix) / rho;
                  double v = b.U(2, ey, ex, iy, ix) / rho;
                  double E = b.U(3, ey, ex, iy, ix);
                  double press = std::max(1e-14, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)));
                  source_val = std::min(source_val, press * SOURCE_CAP_COEFF);
                }
              }
              b.S_buf[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)] = source_val;
            }
          }
        }
      }
    }
  }
}


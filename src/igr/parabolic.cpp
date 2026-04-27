/// @file parabolic.cpp
/// @brief Parabolic IGR evolution — BR2 gradient + divergence operator.
///
/// Solves  ∂Σ/∂t = (1/τ)(S − Σ) + (ε·ρ/τ) ∇·(ρ⁻¹ ∇Σ)
/// using two FR-style passes:
///   Phase 1: Gradient pass — q = ∇Σ  (with interface corrections).
///   Phase 2: Divergence pass — div(q/ρ)  (with BR2 penalty + Neumann BCs).
///   Phase 3: Assembly — combine relaxation and diffusion into sigma_RHS.
///
/// OpenMP: parallelised over elements (ey, ex) in both phases.

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::compute_igr_parabolic_rhs() {
    std::fill(sigma_RHS.begin(), sigma_RHS.end(), 0.0);
    std::fill(qx_buf.begin(), qx_buf.end(), 0.0);
    std::fill(qy_buf.begin(), qy_buf.end(), 0.0);

    // ==================================================================
    // Phase 1: Gradient pass  q = grad(sigma)
    // ==================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
      for (int ex = 0; ex < p.N_ELEM_X; ++ex) {

        // ---- X-gradient ----
        for (int iy = 0; iy < p.N_PTS; ++iy) {
          double sigL = 0, sigR = 0;
          for (int k = 0; k < p.N_PTS; ++k) {
            double s = sigma_field[get_flat_idx(ey, ex, iy, k)];
            sigL += s * basis.l_L[k];
            sigR += s * basis.l_R[k];
          }

          double dummy_neigh[4], dummy_sig_L, dummy_sig_R;
          double dummy_face[4] = {0,0,0,0};
          get_neigh_state_x(ey, ex, iy, false, dummy_face, sigL, dummy_neigh, dummy_sig_L);
          get_neigh_state_x(ey, ex, iy, true,  dummy_face, sigR, dummy_neigh, dummy_sig_R);

          double sigL_hat = 0.5 * (sigL + dummy_sig_L);
          double sigR_hat = 0.5 * (sigR + dummy_sig_R);

          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double ds_dx_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k)
              ds_dx_loc += basis.D[ix][k] * sigma_field[get_flat_idx(ey, ex, iy, k)];

            double ds_dx = (ds_dx_loc
                            + (sigL - sigL_hat) * basis.dgl[ix]
                            + (sigR_hat - sigR) * basis.dgr[ix]) * (2.0 / dx);
            qx_buf[get_flat_idx(ey, ex, iy, ix)] = ds_dx;

            // Zero-normal-gradient Neumann at walls
            if (ex == 0              && p.BC_L != "PERIODIC") qx_buf[get_flat_idx(ey, ex, iy, ix)] = 0.0;
            if (ex == p.N_ELEM_X - 1 && p.BC_R != "PERIODIC") qx_buf[get_flat_idx(ey, ex, iy, ix)] = 0.0;
          }
        }

        // ---- Y-gradient ----
        for (int ix = 0; ix < p.N_PTS; ++ix) {
          double sigB = 0, sigT = 0;
          for (int k = 0; k < p.N_PTS; ++k) {
            double s = sigma_field[get_flat_idx(ey, ex, k, ix)];
            sigB += s * basis.l_L[k];
            sigT += s * basis.l_R[k];
          }

          double dummy_neigh[4], dummy_sig_B, dummy_sig_T;
          double dummy_face[4] = {0,0,0,0};
          get_neigh_state_y(ey, ex, ix, false, dummy_face, sigB, dummy_neigh, dummy_sig_B);
          get_neigh_state_y(ey, ex, ix, true,  dummy_face, sigT, dummy_neigh, dummy_sig_T);

          double sigB_hat = 0.5 * (sigB + dummy_sig_B);
          double sigT_hat = 0.5 * (sigT + dummy_sig_T);

          for (int iy = 0; iy < p.N_PTS; ++iy) {
            double ds_dy_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k)
              ds_dy_loc += basis.D[iy][k] * sigma_field[get_flat_idx(ey, ex, k, ix)];

            double ds_dy = (ds_dy_loc
                            + (sigB - sigB_hat) * basis.dgl[iy]
                            + (sigT_hat - sigT) * basis.dgr[iy]) * (2.0 / dy);
            qy_buf[get_flat_idx(ey, ex, iy, ix)] = ds_dy;

            if (ey == 0              && p.BC_B != "PERIODIC") qy_buf[get_flat_idx(ey, ex, iy, ix)] = 0.0;
            if (ey == p.N_ELEM_Y - 1 && p.BC_T != "PERIODIC") qy_buf[get_flat_idx(ey, ex, iy, ix)] = 0.0;
          }
        }
      }
    }

    // ==================================================================
    // Phase 2 + 3: Divergence pass + Assembly
    // ==================================================================
    const double epsilon = p.ALPHA_SCALE * (dx * dy);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
      for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
        for (int iy = 0; iy < p.N_PTS; ++iy) {
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double div_q = 0.0;

            // ---- X-contribution ----
            {
              double qxL = 0, qxR = 0, sigL = 0, sigR = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                int idx = get_flat_idx(ey, ex, iy, k);
                qxL += qx_buf[idx] * basis.l_L[k];  qxR += qx_buf[idx] * basis.l_R[k];
                sigL += sigma_field[idx] * basis.l_L[k]; sigR += sigma_field[idx] * basis.l_R[k];
              }

              double qx_neigh_L = 0, qx_neigh_R = 0, sig_neigh_L = 0, sig_neigh_R = 0;
              if (ex > 0) {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(ey, ex-1, iy, k);
                  qx_neigh_L += qx_buf[idx] * basis.l_R[k]; sig_neigh_L += sigma_field[idx] * basis.l_R[k];
                }
              } else if (p.BC_L == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(ey, p.N_ELEM_X-1, iy, k);
                  qx_neigh_L += qx_buf[idx] * basis.l_R[k]; sig_neigh_L += sigma_field[idx] * basis.l_R[k];
                }
              } else { qx_neigh_L = qxL; sig_neigh_L = sigL; }

              if (ex < p.N_ELEM_X - 1) {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(ey, ex+1, iy, k);
                  qx_neigh_R += qx_buf[idx] * basis.l_L[k]; sig_neigh_R += sigma_field[idx] * basis.l_L[k];
                }
              } else if (p.BC_R == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(ey, 0, iy, k);
                  qx_neigh_R += qx_buf[idx] * basis.l_L[k]; sig_neigh_R += sigma_field[idx] * basis.l_L[k];
                }
              } else { qx_neigh_R = qxR; sig_neigh_R = sigR; }

              // Interface density
              double rhoL = 0, rhoR = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                rhoL += U(0, ey, ex, iy, k) * basis.l_L[k];
                rhoR += U(0, ey, ex, iy, k) * basis.l_R[k];
              }
              double rho_neigh_L = rhoL, rho_neigh_R = rhoR;
              {
                double U_n[4], sn;
                double fL[4]={}, fR[4]={};
                for (int k = 0; k < p.N_PTS; ++k) for (int v = 0; v < 4; ++v) {
                  fL[v] += U(v, ey, ex, iy, k)*basis.l_L[k];
                  fR[v] += U(v, ey, ex, iy, k)*basis.l_R[k];
                }
                get_neigh_state_x(ey, ex, iy, false, fL, 0, U_n, sn); rho_neigh_L = U_n[0];
                get_neigh_state_x(ey, ex, iy, true,  fR, 0, U_n, sn); rho_neigh_R = U_n[0];
              }
              double rho_face_L = 0.5*(rhoL + rho_neigh_L);
              double rho_face_R = 0.5*(rhoR + rho_neigh_R);

              double eta_L = p.IGR_BR2_ETA * (p.P_DEG+1)*(p.P_DEG+1) * (epsilon / std::max(1e-10, rho_face_L)) / dx;
              double eta_R = p.IGR_BR2_ETA * (p.P_DEG+1)*(p.P_DEG+1) * (epsilon / std::max(1e-10, rho_face_R)) / dx;

              double F_L = 0.5*((-qxL/rhoL) + (-qx_neigh_L/rho_neigh_L)) - eta_L*(sig_neigh_L - sigL);
              double F_R = 0.5*((qxR/rhoR)  + (qx_neigh_R/rho_neigh_R))  - eta_R*(sig_neigh_R - sigR);

              if (ex == 0              && p.BC_L != "PERIODIC") F_L = 0.0;
              if (ex == p.N_ELEM_X - 1 && p.BC_R != "PERIODIC") F_R = 0.0;

              double dqx_dx_loc = 0;
              for (int k = 0; k < p.N_PTS; ++k)
                dqx_dx_loc += basis.D[ix][k] * (qx_buf[get_flat_idx(ey, ex, iy, k)] / std::max(1e-10, U(0, ey, ex, iy, k)));
              div_q += (dqx_dx_loc + (F_L - (-qxL/rhoL))*basis.dgl[ix] + (F_R - qxR/rhoR)*basis.dgr[ix]) * (2.0/dx);
            }

            // ---- Y-contribution ----
            {
              double qyB = 0, qyT = 0, sigB = 0, sigT = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                int idx = get_flat_idx(ey, ex, k, ix);
                qyB += qy_buf[idx] * basis.l_L[k]; qyT += qy_buf[idx] * basis.l_R[k];
                sigB += sigma_field[idx] * basis.l_L[k]; sigT += sigma_field[idx] * basis.l_R[k];
              }

              double qy_neigh_B = 0, qy_neigh_T = 0, sig_neigh_B = 0, sig_neigh_T = 0;
              if (ey > 0) {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(ey-1, ex, k, ix);
                  qy_neigh_B += qy_buf[idx] * basis.l_R[k]; sig_neigh_B += sigma_field[idx] * basis.l_R[k];
                }
              } else if (p.BC_B == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(p.N_ELEM_Y-1, ex, k, ix);
                  qy_neigh_B += qy_buf[idx] * basis.l_R[k]; sig_neigh_B += sigma_field[idx] * basis.l_R[k];
                }
              } else { qy_neigh_B = qyB; sig_neigh_B = sigB; }

              if (ey < p.N_ELEM_Y - 1) {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(ey+1, ex, k, ix);
                  qy_neigh_T += qy_buf[idx] * basis.l_L[k]; sig_neigh_T += sigma_field[idx] * basis.l_L[k];
                }
              } else if (p.BC_T == "PERIODIC") {
                for (int k = 0; k < p.N_PTS; ++k) {
                  int idx = get_flat_idx(0, ex, k, ix);
                  qy_neigh_T += qy_buf[idx] * basis.l_L[k]; sig_neigh_T += sigma_field[idx] * basis.l_L[k];
                }
              } else { qy_neigh_T = qyT; sig_neigh_T = sigT; }

              double rhoB = 0, rhoT = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                rhoB += U(0, ey, ex, k, ix) * basis.l_L[k];
                rhoT += U(0, ey, ex, k, ix) * basis.l_R[k];
              }
              double rho_neigh_B = rhoB, rho_neigh_T = rhoT;
              {
                double U_n[4], sn;
                double fB[4]={}, fT[4]={};
                for (int k = 0; k < p.N_PTS; ++k) for (int v = 0; v < 4; ++v) {
                  fB[v] += U(v, ey, ex, k, ix)*basis.l_L[k];
                  fT[v] += U(v, ey, ex, k, ix)*basis.l_R[k];
                }
                get_neigh_state_y(ey, ex, ix, false, fB, 0, U_n, sn); rho_neigh_B = U_n[0];
                get_neigh_state_y(ey, ex, ix, true,  fT, 0, U_n, sn); rho_neigh_T = U_n[0];
              }
              double rho_face_B = 0.5*(rhoB + rho_neigh_B);
              double rho_face_T = 0.5*(rhoT + rho_neigh_T);

              double eta_B = p.IGR_BR2_ETA * (p.P_DEG+1)*(p.P_DEG+1) * (epsilon / std::max(1e-10, rho_face_B)) / dy;
              double eta_T = p.IGR_BR2_ETA * (p.P_DEG+1)*(p.P_DEG+1) * (epsilon / std::max(1e-10, rho_face_T)) / dy;

              double F_B = 0.5*((-qyB/rhoB) + (-qy_neigh_B/rho_neigh_B)) - eta_B*(sig_neigh_B - sigB);
              double F_T = 0.5*((qyT/rhoT)  + (qy_neigh_T/rho_neigh_T))  - eta_T*(sig_neigh_T - sigT);

              if (ey == 0              && p.BC_B != "PERIODIC") F_B = 0.0;
              if (ey == p.N_ELEM_Y - 1 && p.BC_T != "PERIODIC") F_T = 0.0;

              double dqy_dy_loc = 0;
              for (int k = 0; k < p.N_PTS; ++k)
                dqy_dy_loc += basis.D[iy][k] * (qy_buf[get_flat_idx(ey, ex, k, ix)] / std::max(1e-10, U(0, ey, ex, k, ix)));
              div_q += (dqy_dy_loc + (F_B - (-qyB/rhoB))*basis.dgl[iy] + (F_T - qyT/rhoT)*basis.dgr[iy]) * (2.0/dy);
            }

            // ---- Phase 3: Assembly ----
            int flat = get_flat_idx(ey, ex, iy, ix);
            double S   = S_buf[flat];
            double sig = sigma_field[flat];
            double rho = std::max(1e-10, U(0, ey, ex, iy, ix));
            sigma_RHS[flat] = (1.0 / p.IGR_TAU_R) * ((S - sig) + epsilon * rho * div_q);
          }
        }
      }
    }
}

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
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

/// Compute the Parabolic IGR right-hand side.
void Solver::compute_igr_parabolic_rhs() {
  // Phase 1: Gradient pass (q = grad(sigma))
  for (auto& b : blocks) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ey = 0; ey < b.ny; ++ey) {
      for (int ex = 0; ex < b.nx; ++ex) {
        // X-gradient
        for (int iy = 0; iy < p.N_PTS; ++iy) {
          double sigL = 0, sigR = 0;
          for (int k = 0; k < p.N_PTS; ++k) {
            double s = b.sigma_field[b.get_flat_idx(ey, ex, iy, k, p.N_PTS)];
            sigL += s * basis.l_L[k];
            sigR += s * basis.l_R[k];
          }

          double dummy_neigh[4], dummy_sig_L, dummy_sig_R;
          double dummy_face[4] = {0,0,0,0};
          get_neigh_state_x(b, ey, ex, iy, false, dummy_face, sigL, dummy_neigh, dummy_sig_L);
          get_neigh_state_x(b, ey, ex, iy, true,  dummy_face, sigR, dummy_neigh, dummy_sig_R);

          double sigL_hat = 0.5 * (sigL + dummy_sig_L);
          double sigR_hat = 0.5 * (sigR + dummy_sig_R);

          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double ds_dx_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k)
              ds_dx_loc += basis.D[ix][k] * b.sigma_field[b.get_flat_idx(ey, ex, iy, k, p.N_PTS)];
            double ds_dx = (ds_dx_loc + (sigL_hat - sigL) * basis.dgl[ix] + (sigR_hat - sigR) * basis.dgr[ix]) * (2.0 / b.dx);
            b.qx_buf[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)] = ds_dx;
          }
        }
        // Y-gradient
        for (int ix = 0; ix < p.N_PTS; ++ix) {
          double sigB = 0, sigT = 0;
          for (int k = 0; k < p.N_PTS; ++k) {
            double s = b.sigma_field[b.get_flat_idx(ey, ex, k, ix, p.N_PTS)];
            sigB += s * basis.l_L[k];
            sigT += s * basis.l_R[k];
          }

          double dummy_neigh[4], dummy_sig_B, dummy_sig_T;
          double dummy_face[4] = {0,0,0,0};
          get_neigh_state_y(b, ey, ex, ix, false, dummy_face, sigB, dummy_neigh, dummy_sig_B);
          get_neigh_state_y(b, ey, ex, ix, true,  dummy_face, sigT, dummy_neigh, dummy_sig_T);

          double sigB_hat = 0.5 * (sigB + dummy_sig_B);
          double sigT_hat = 0.5 * (sigT + dummy_sig_T);

          for (int iy = 0; iy < p.N_PTS; ++iy) {
            double ds_dy_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k)
              ds_dy_loc += basis.D[iy][k] * b.sigma_field[b.get_flat_idx(ey, ex, k, ix, p.N_PTS)];
            double ds_dy = (ds_dy_loc + (sigB_hat - sigB) * basis.dgl[iy] + (sigT_hat - sigT) * basis.dgr[iy]) * (2.0 / b.dy);
            b.qy_buf[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)] = ds_dy;
          }
        }
      }
    }
  }

  // Phase 2 & 3: Fused Divergence Pass + Assembly
  for (auto& b : blocks) {
    const double epsilon = p.ALPHA_SCALE * (b.dx * b.dy);
    const double br2_factor = p.IGR_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ey = 0; ey < b.ny; ++ey) {
      for (int ex = 0; ex < b.nx; ++ex) {
        
        // --- PRE-CALCULATE X-INTERFACE VALUES ---
        double fhatL[MAX_PTS], fhatR[MAX_PTS], fL_int[MAX_PTS], fR_int[MAX_PTS];
        for (int iy = 0; iy < p.N_PTS; ++iy) {
          double fL = 0, fR = 0, sigL = 0, sigR = 0, rhoL = 0, rhoR = 0;
          for (int k = 0; k < p.N_PTS; ++k) {
            int idx = b.get_flat_idx(ey, ex, iy, k, p.N_PTS);
            double r_k = std::max(1e-12, b.U(0, ey, ex, iy, k));
            fL += (b.qx_buf[idx] / r_k) * basis.l_L[k];
            fR += (b.qx_buf[idx] / r_k) * basis.l_R[k];
            sigL += b.sigma_field[idx] * basis.l_L[k];
            sigR += b.sigma_field[idx] * basis.l_R[k];
            rhoL += r_k * basis.l_L[k];
            rhoR += r_k * basis.l_R[k];
          }
          fL_int[iy] = fL; fR_int[iy] = fR;

          double f_nb, sig_nb, rho_nb;
          
          auto get_val = [&](const NeighborInfo& ni_face, bool right_face, double f_self, double sig_self, double rho_self) {
            double f_res, sig_res, rho_res;
            if (ni_face.id != -1) {
              const Block& nb = blocks[ni_face.id];
              int nex = (ni_face.face == 'L') ? 0 : nb.nx - 1;
              const double* w = (ni_face.face == 'L') ? basis.l_L.data() : basis.l_R.data();
              f_res = 0; sig_res = 0; rho_res = 0;
              for (int k=0; k<p.N_PTS; ++k) {
                int idx = nb.get_flat_idx(ey, nex, iy, k, p.N_PTS);
                double r_k = std::max(1e-12, nb.U(0, ey, nex, iy, k));
                f_res += (nb.qx_buf[idx] / r_k) * w[k];
                sig_res += nb.sigma_field[idx] * w[k];
                rho_res += r_k * w[k];
              }

            } else {
              f_res = f_self; sig_res = sig_self; rho_res = rho_self;
            }
            return std::make_tuple(f_res, sig_res, rho_res);
          };

          if (ex == 0) {
            std::tie(f_nb, sig_nb, rho_nb) = get_val(b.ni_l, false, fL, sigL, rhoL);
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoL + rho_nb))) / b.dx;
            fhatL[iy] = 0.5*(f_nb + fL) + eta*(sigL - sig_nb);
            if (b.ni_l.is_wall || b.ni_l.is_noslip_wall || b.ni_l.is_moving_wall || b.ni_l.is_supersonic_inflow || b.ni_l.is_supersonic_outflow || b.ni_l.is_characteristic) fhatL[iy] = 0.0;
          } else {
            f_nb = 0; sig_nb = 0; rho_nb = 0;
            for (int k=0; k<p.N_PTS; ++k) {
              int idx = b.get_flat_idx(ey, ex-1, iy, k, p.N_PTS);
              double r_k = std::max(1e-12, b.U(0, ey, ex-1, iy, k));
              f_nb += (b.qx_buf[idx] / r_k) * basis.l_R[k];
              sig_nb += b.sigma_field[idx] * basis.l_R[k];
              rho_nb += r_k * basis.l_R[k];
            }
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoL + rho_nb))) / b.dx;
            fhatL[iy] = 0.5*(f_nb + fL) + eta*(sigL - sig_nb);
          }

          if (ex == b.nx-1) {
            std::tie(f_nb, sig_nb, rho_nb) = get_val(b.ni_r, true, fR, sigR, rhoR);
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoR + rho_nb))) / b.dx;
            fhatR[iy] = 0.5*(f_nb + fR) + eta*(sig_nb - sigR);
            if (b.ni_r.is_wall || b.ni_r.is_noslip_wall || b.ni_r.is_moving_wall || b.ni_r.is_supersonic_inflow || b.ni_r.is_supersonic_outflow || b.ni_r.is_characteristic) fhatR[iy] = 0.0;
          } else {
            f_nb = 0; sig_nb = 0; rho_nb = 0;
            for (int k=0; k<p.N_PTS; ++k) {
              int idx = b.get_flat_idx(ey, ex+1, iy, k, p.N_PTS);
              double r_k = std::max(1e-12, b.U(0, ey, ex+1, iy, k));
              f_nb += (b.qx_buf[idx] / r_k) * basis.l_L[k];
              sig_nb += b.sigma_field[idx] * basis.l_L[k];
              rho_nb += r_k * basis.l_L[k];
            }
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoR + rho_nb))) / b.dx;
            fhatR[iy] = 0.5*(f_nb + fR) + eta*(sig_nb - sigR);
          }
        }

        // --- PRE-CALCULATE Y-INTERFACE VALUES ---
        double fhatB[MAX_PTS], fhatT[MAX_PTS], fB_int[MAX_PTS], fT_int[MAX_PTS];
        for (int ix = 0; ix < p.N_PTS; ++ix) {
          double fB = 0, fT = 0, sigB = 0, sigT = 0, rhoB = 0, rhoT = 0;
          for (int k = 0; k < p.N_PTS; ++k) {
            int idx = b.get_flat_idx(ey, ex, k, ix, p.N_PTS);
            double r_k = std::max(1e-12, b.U(0, ey, ex, k, ix));
            fB += (b.qy_buf[idx] / r_k) * basis.l_L[k];
            fT += (b.qy_buf[idx] / r_k) * basis.l_R[k];
            sigB += b.sigma_field[idx] * basis.l_L[k];
            sigT += b.sigma_field[idx] * basis.l_R[k];
            rhoB += r_k * basis.l_L[k];
            rhoT += r_k * basis.l_R[k];
          }
          fB_int[ix] = fB; fT_int[ix] = fT;

          double f_nb, sig_nb, rho_nb;
          auto get_val_y = [&](const NeighborInfo& ni_face, bool top_face, double f_self, double sig_self, double rho_self) {
            double f_res, sig_res, rho_res;
            if (ni_face.id != -1) {
              const Block& nb = blocks[ni_face.id];
              int ney = (ni_face.face == 'B') ? 0 : nb.ny - 1;
              const double* w = (ni_face.face == 'B') ? basis.l_L.data() : basis.l_R.data();
              f_res = 0; sig_res = 0; rho_res = 0;
              for (int k=0; k<p.N_PTS; ++k) {
                int idx = nb.get_flat_idx(ney, ex, k, ix, p.N_PTS);
                double r_k = std::max(1e-12, nb.U(0, ney, ex, k, ix));
                f_res += (nb.qy_buf[idx] / r_k) * w[k];
                sig_res += nb.sigma_field[idx] * w[k];
                rho_res += r_k * w[k];
              }

            } else {
              f_res = f_self; sig_res = sig_self; rho_res = rho_self;
            }
            return std::make_tuple(f_res, sig_res, rho_res);
          };

          if (ey == 0) {
            std::tie(f_nb, sig_nb, rho_nb) = get_val_y(b.ni_b, false, fB, sigB, rhoB);
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoB + rho_nb))) / b.dy;
            fhatB[ix] = 0.5*(f_nb + fB) + eta*(sigB - sig_nb);
            if (b.ni_b.is_wall || b.ni_b.is_noslip_wall || b.ni_b.is_moving_wall || b.ni_b.is_supersonic_inflow || b.ni_b.is_supersonic_outflow || b.ni_b.is_characteristic) fhatB[ix] = 0.0;
          } else {
            f_nb = 0; sig_nb = 0; rho_nb = 0;
            for (int k=0; k<p.N_PTS; ++k) {
              int idx = b.get_flat_idx(ey-1, ex, k, ix, p.N_PTS);
              double r_k = std::max(1e-12, b.U(0, ey-1, ex, k, ix));
              f_nb += (b.qy_buf[idx] / r_k) * basis.l_R[k];
              sig_nb += b.sigma_field[idx] * basis.l_R[k];
              rho_nb += r_k * basis.l_R[k];
            }
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoB + rho_nb))) / b.dy;
            fhatB[ix] = 0.5*(f_nb + fB) + eta*(sigB - sig_nb);
          }

          if (ey == b.ny-1) {
            std::tie(f_nb, sig_nb, rho_nb) = get_val_y(b.ni_t, true, fT, sigT, rhoT);
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoT + rho_nb))) / b.dy;
            fhatT[ix] = 0.5*(f_nb + fT) + eta*(sig_nb - sigT);
            if (b.ni_t.is_wall || b.ni_t.is_noslip_wall || b.ni_t.is_moving_wall || b.ni_t.is_supersonic_inflow || b.ni_t.is_supersonic_outflow || b.ni_t.is_characteristic) fhatT[ix] = 0.0;
          } else {
            f_nb = 0; sig_nb = 0; rho_nb = 0;
            for (int k=0; k<p.N_PTS; ++k) {
              int idx = b.get_flat_idx(ey+1, ex, k, ix, p.N_PTS);
              double r_k = std::max(1e-12, b.U(0, ey+1, ex, k, ix));
              f_nb += (b.qy_buf[idx] / r_k) * basis.l_L[k];
              sig_nb += b.sigma_field[idx] * basis.l_L[k];
              rho_nb += r_k * basis.l_L[k];
            }
            double eta = br2_factor * (epsilon / std::max(1e-12, 0.5*(rhoT + rho_nb))) / b.dy;
            fhatT[ix] = 0.5*(f_nb + fT) + eta*(sig_nb - sigT);
          }
        }

        // --- INTERIOR DIVERGENCE + ASSEMBLY ---
        for (int iy = 0; iy < p.N_PTS; ++iy) {
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            double df_dx_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
              int idx = b.get_flat_idx(ey, ex, iy, k, p.N_PTS);
              df_dx_loc += basis.D[ix][k] * (b.qx_buf[idx] / std::max(1e-12, b.U(0, ey, ex, iy, k)));
            }
            double div_x = (df_dx_loc + (fhatL[iy] - fL_int[iy])*basis.dgl[ix] + (fhatR[iy] - fR_int[iy])*basis.dgr[ix]) * (2.0 / b.dx);

            double df_dy_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
              int idx = b.get_flat_idx(ey, ex, k, ix, p.N_PTS);
              df_dy_loc += basis.D[iy][k] * (b.qy_buf[idx] / std::max(1e-12, b.U(0, ey, ex, k, ix)));
            }
            double div_y = (df_dy_loc + (fhatB[ix] - fB_int[ix])*basis.dgl[iy] + (fhatT[ix] - fT_int[ix])*basis.dgr[iy]) * (2.0 / b.dy);

            int flat = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
            double S = b.S_buf[flat];
            double sig = b.sigma_field[flat];
            double rho = std::max(1e-12, b.U(0, ey, ex, iy, ix));
            b.sigma_RHS[flat] = (1.0 / p.IGR_TAU_R) * ((S - sig) + epsilon * rho * (div_x + div_y));
          }
        }
      }
    }
  }
}

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

/// Compute the Parabolic IGR right-hand side.
///
/// @details
/// Data Structures & Indexing:
///   - `sigma_field`: The global scalar entropic pressure field (read-only for gradients).
///   - `qx_buf`, `qy_buf`: Temporary flat arrays storing the intermediate $\nabla \Sigma$ components.
///   - `sigma_RHS`: The output explicit RHS array.
///   - `U(0, ey, ex, iy, ix)`: The global conserved state array (read-only to extract density for BR2 scaling).
///   - `S_buf`: The global scalar sensor source term.
/// Assumptions:
///   - `sigma_field` and `S_buf` are up to date.
///   - OpenMP is thread-safe here because Phase 1 (gradients) writes entirely to `qx_buf`/`qy_buf` using 
///     the flat index `get_flat_idx(ey, ex, iy, ix)`. Phase 2 (divergence + assembly) reads from these 
///     buffers and writes to `sigma_RHS` using the same disjoint indexing per element.
void Solver::compute_igr_parabolic_rhs() {
  std::fill(sigma_RHS.begin(), sigma_RHS.end(), 0.0);
  std::fill(qx_buf.begin(), qx_buf.end(), 0.0);
  std::fill(qy_buf.begin(), qy_buf.end(), 0.0);

  // ==================================================================
  // Phase 1: Gradient pass  q = grad(sigma)
  //
  // Wall Neumann BCs (∂σ/∂n = 0) are handled automatically:
  // get_neigh_state_x/y copies sig_face → sig_neigh for WALL BCs,
  // so sig_hat = sig_int, making the FR correction zero at walls.
  // No explicit zeroing of gradients is needed or correct here.
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
        double dummy_face[4] = {0, 0, 0, 0};
        get_neigh_state_x(ey, ex, iy, false, dummy_face, sigL, dummy_neigh,
                          dummy_sig_L);
        get_neigh_state_x(ey, ex, iy, true,  dummy_face, sigR, dummy_neigh,
                          dummy_sig_R);

        double sigL_hat = 0.5 * (sigL + dummy_sig_L);
        double sigR_hat = 0.5 * (sigR + dummy_sig_R);

        for (int ix = 0; ix < p.N_PTS; ++ix) {
          double ds_dx_loc = 0;
          for (int k = 0; k < p.N_PTS; ++k)
            ds_dx_loc +=
                basis.D[ix][k] * sigma_field[get_flat_idx(ey, ex, iy, k)];

          double ds_dx = (ds_dx_loc + (sigL_hat - sigL) * basis.dgl[ix] +
                          (sigR_hat - sigR) * basis.dgr[ix]) *
                         (2.0 / dx);
          qx_buf[get_flat_idx(ey, ex, iy, ix)] = ds_dx;
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
        double dummy_face[4] = {0, 0, 0, 0};
        get_neigh_state_y(ey, ex, ix, false, dummy_face, sigB, dummy_neigh,
                          dummy_sig_B);
        get_neigh_state_y(ey, ex, ix, true,  dummy_face, sigT, dummy_neigh,
                          dummy_sig_T);

        double sigB_hat = 0.5 * (sigB + dummy_sig_B);
        double sigT_hat = 0.5 * (sigT + dummy_sig_T);

        for (int iy = 0; iy < p.N_PTS; ++iy) {
          double ds_dy_loc = 0;
          for (int k = 0; k < p.N_PTS; ++k)
            ds_dy_loc +=
                basis.D[iy][k] * sigma_field[get_flat_idx(ey, ex, k, ix)];

          double ds_dy = (ds_dy_loc + (sigB_hat - sigB) * basis.dgl[iy] +
                          (sigT_hat - sigT) * basis.dgr[iy]) *
                         (2.0 / dy);
          qy_buf[get_flat_idx(ey, ex, iy, ix)] = ds_dy;
        }
      }
    }
  }

  // ==================================================================
  // Phase 2 + 3: Divergence pass + Assembly
  //
  // Computes  div(q/ρ) = ∂(qx/ρ)/∂x + ∂(qy/ρ)/∂y
  // using FR with SIPG/BR2 interface penalties.
  //
  // Flux convention:
  //   Physical flux: f = q/ρ  (scalar, in the coordinate direction).
  //   Common flux at an interface (normal from e⁻ to e⁺):
  //     f̂ = {f} + η·(σ⁺ − σ⁻)
  //   where {f} = ½(f⁻ + f⁺) is the average and the penalty
  //   ensures coercivity of the bilinear form.
  //
  // Wall Neumann BC:  f̂ = 0 at walls (zero diffusive flux through wall).
  //   The FR correction (f̂ - f_int)*g' then becomes (-f_int)*g',
  //   which properly drives the boundary flux toward zero.
  // ==================================================================
  const double epsilon = p.ALPHA_SCALE * (dx * dy);

  #pragma omp parallel for collapse(2) schedule(static)
  for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
    for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
      for (int iy = 0; iy < p.N_PTS; ++iy) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
          double div_q = 0.0;

          // ---- X-contribution to divergence ----
          {
            // Extrapolate f = qx/ρ and σ to faces of this element
            double fL = 0, fR = 0, sigL = 0, sigR = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
              int idx = get_flat_idx(ey, ex, iy, k);
              double rho_k = clamp_positivity(U(0, ey, ex, iy, k));
              fL += (qx_buf[idx] / rho_k) * basis.l_L[k];
              fR += (qx_buf[idx] / rho_k) * basis.l_R[k];
              sigL += sigma_field[idx] * basis.l_L[k];
              sigR += sigma_field[idx] * basis.l_R[k];
            }

            // Get neighbor face values
            double f_neigh_L = fL, f_neigh_R = fR;
            double sig_neigh_L = sigL, sig_neigh_R = sigR;

            // --- Left neighbor ---
            if (ex > 0 || p.BC_L == "PERIODIC") {
              int nex = (ex > 0) ? ex - 1 : p.N_ELEM_X - 1;
              f_neigh_L = 0;
              sig_neigh_L = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                int idx = get_flat_idx(ey, nex, iy, k);
                double rho_k = clamp_positivity(U(0, ey, nex, iy, k));
                f_neigh_L += (qx_buf[idx] / rho_k) * basis.l_R[k];
                sig_neigh_L += sigma_field[idx] * basis.l_R[k];
              }
            }

            // --- Right neighbor ---
            if (ex < p.N_ELEM_X - 1 || p.BC_R == "PERIODIC") {
              int nex = (ex < p.N_ELEM_X - 1) ? ex + 1 : 0;
              f_neigh_R = 0;
              sig_neigh_R = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                int idx = get_flat_idx(ey, nex, iy, k);
                double rho_k = clamp_positivity(U(0, ey, nex, iy, k));
                f_neigh_R += (qx_buf[idx] / rho_k) * basis.l_L[k];
                sig_neigh_R += sigma_field[idx] * basis.l_L[k];
              }
            }

            // Interface density for penalty coefficient
            double rhoL_int = 0, rhoR_int = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
              rhoL_int += U(0, ey, ex, iy, k) * basis.l_L[k];
              rhoR_int += U(0, ey, ex, iy, k) * basis.l_R[k];
            }
            double U_n[4];
            double sn;
            double faceL[4] = {}, faceR[4] = {};
            for (int k = 0; k < p.N_PTS; ++k)
              for (int v = 0; v < 4; ++v) {
                faceL[v] += U(v, ey, ex, iy, k) * basis.l_L[k];
                faceR[v] += U(v, ey, ex, iy, k) * basis.l_R[k];
              }
            double rho_neigh_L_val = rhoL_int, rho_neigh_R_val = rhoR_int;
            get_neigh_state_x(ey, ex, iy, false, faceL, 0, U_n, sn);
            rho_neigh_L_val = U_n[0];
            get_neigh_state_x(ey, ex, iy, true,  faceR, 0, U_n, sn);
            rho_neigh_R_val = U_n[0];
            double rho_face_L = 0.5 * (rhoL_int + rho_neigh_L_val);
            double rho_face_R = 0.5 * (rhoR_int + rho_neigh_R_val);

            double eta_L = p.IGR_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1) *
                           (epsilon / clamp_positivity(rho_face_L)) / dx;
            double eta_R = p.IGR_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1) *
                           (epsilon / clamp_positivity(rho_face_R)) / dx;

            // Common flux at left face:
            //   f̂_L = ½(f_neigh_L + fL) + η_L·(σL − σ_neigh_L)
            double fhat_L =
                0.5 * (f_neigh_L + fL) + eta_L * (sigL - sig_neigh_L);

            // Common flux at right face:
            //   f̂_R = ½(fR + f_neigh_R) + η_R·(σ_neigh_R − σR)
            double fhat_R =
                0.5 * (fR + f_neigh_R) + eta_R * (sig_neigh_R - sigR);

            // Wall Neumann BC: zero diffusive flux at walls.
            // Set fhat = 0 so FR correction = (0 - fL)*dgl drives
            // boundary flux toward zero.  fL stays as-is (interior value).
            if (ex == 0              && p.BC_L != "PERIODIC") fhat_L = 0.0;
            if (ex == p.N_ELEM_X - 1 && p.BC_R != "PERIODIC") fhat_R = 0.0;

            // Interior derivative of f = qx/ρ
            double df_dx_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k)
              df_dx_loc +=
                  basis.D[ix][k] * (qx_buf[get_flat_idx(ey, ex, iy, k)] /
                                    clamp_positivity(U(0, ey, ex, iy, k)));

            // FR correction: ∂f/∂x ≈ [df_loc + (f̂_L − fL)·g'_L + (f̂_R − fR)·g'_R] · 2/dx
            div_q += (df_dx_loc + (fhat_L - fL) * basis.dgl[ix] +
                      (fhat_R - fR) * basis.dgr[ix]) *
                     (2.0 / dx);
          }

          // ---- Y-contribution to divergence ----
          {
            // Extrapolate f = qy/ρ and σ to faces
            double fB = 0, fT = 0, sigB = 0, sigT = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
              int idx = get_flat_idx(ey, ex, k, ix);
              double rho_k = clamp_positivity(U(0, ey, ex, k, ix));
              fB += (qy_buf[idx] / rho_k) * basis.l_L[k];
              fT += (qy_buf[idx] / rho_k) * basis.l_R[k];
              sigB += sigma_field[idx] * basis.l_L[k];
              sigT += sigma_field[idx] * basis.l_R[k];
            }

            double f_neigh_B = fB, f_neigh_T = fT;
            double sig_neigh_B = sigB, sig_neigh_T = sigT;

            // --- Bottom neighbor ---
            if (ey > 0 || p.BC_B == "PERIODIC") {
              int ney = (ey > 0) ? ey - 1 : p.N_ELEM_Y - 1;
              f_neigh_B = 0;
              sig_neigh_B = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                int idx = get_flat_idx(ney, ex, k, ix);
                double rho_k = clamp_positivity(U(0, ney, ex, k, ix));
                f_neigh_B += (qy_buf[idx] / rho_k) * basis.l_R[k];
                sig_neigh_B += sigma_field[idx] * basis.l_R[k];
              }
            }

            // --- Top neighbor ---
            if (ey < p.N_ELEM_Y - 1 || p.BC_T == "PERIODIC") {
              int ney = (ey < p.N_ELEM_Y - 1) ? ey + 1 : 0;
              f_neigh_T = 0;
              sig_neigh_T = 0;
              for (int k = 0; k < p.N_PTS; ++k) {
                int idx = get_flat_idx(ney, ex, k, ix);
                double rho_k = clamp_positivity(U(0, ney, ex, k, ix));
                f_neigh_T += (qy_buf[idx] / rho_k) * basis.l_L[k];
                sig_neigh_T += sigma_field[idx] * basis.l_L[k];
              }
            }

            // Interface density for penalty
            double rhoB_int = 0, rhoT_int = 0;
            for (int k = 0; k < p.N_PTS; ++k) {
              rhoB_int += U(0, ey, ex, k, ix) * basis.l_L[k];
              rhoT_int += U(0, ey, ex, k, ix) * basis.l_R[k];
            }
            double U_n[4];
            double sn;
            double faceB[4] = {}, faceT[4] = {};
            for (int k = 0; k < p.N_PTS; ++k)
              for (int v = 0; v < 4; ++v) {
                faceB[v] += U(v, ey, ex, k, ix) * basis.l_L[k];
                faceT[v] += U(v, ey, ex, k, ix) * basis.l_R[k];
              }
            double rho_neigh_B_val = rhoB_int, rho_neigh_T_val = rhoT_int;
            get_neigh_state_y(ey, ex, ix, false, faceB, 0, U_n, sn);
            rho_neigh_B_val = U_n[0];
            get_neigh_state_y(ey, ex, ix, true,  faceT, 0, U_n, sn);
            rho_neigh_T_val = U_n[0];
            double rho_face_B = 0.5 * (rhoB_int + rho_neigh_B_val);
            double rho_face_T = 0.5 * (rhoT_int + rho_neigh_T_val);

            double eta_B = p.IGR_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1) *
                           (epsilon / clamp_positivity(rho_face_B)) / dy;
            double eta_T = p.IGR_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1) *
                           (epsilon / clamp_positivity(rho_face_T)) / dy;

            // Common flux at bottom face
            double fhat_B =
                0.5 * (f_neigh_B + fB) + eta_B * (sigB - sig_neigh_B);

            // Common flux at top face
            double fhat_T =
                0.5 * (fT + f_neigh_T) + eta_T * (sig_neigh_T - sigT);

            // Wall Neumann BC: zero diffusive flux at walls
            if (ey == 0              && p.BC_B != "PERIODIC") fhat_B = 0.0;
            if (ey == p.N_ELEM_Y - 1 && p.BC_T != "PERIODIC") fhat_T = 0.0;

            // Interior derivative
            double df_dy_loc = 0;
            for (int k = 0; k < p.N_PTS; ++k)
              df_dy_loc +=
                  basis.D[iy][k] * (qy_buf[get_flat_idx(ey, ex, k, ix)] /
                                    clamp_positivity(U(0, ey, ex, k, ix)));

            div_q += (df_dy_loc + (fhat_B - fB) * basis.dgl[iy] +
                      (fhat_T - fT) * basis.dgr[iy]) *
                     (2.0 / dy);
          }

          // ---- Phase 3: Assembly ----
          int flat = get_flat_idx(ey, ex, iy, ix);
          double S = S_buf[flat];
          double sig = sigma_field[flat];
          double rho = clamp_positivity(U(0, ey, ex, iy, ix));
          sigma_RHS[flat] =
              (1.0 / p.IGR_TAU_R) * ((S - sig) + epsilon * rho * div_q);
        }
      }
    }
  }
}

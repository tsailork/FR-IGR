/// @file gradient.cpp
/// @brief BR2 Phase 1 — compute gradients of conservative variables ∇U.
///
/// Computes dU/dx and dU/dy at every solution point using FR reconstruction
/// with central (average) interface fluxes.  Results are stored in the per-block
/// grad_Ux and grad_Uy buffers, indexed as [var * n_dofs + flat_idx].
///
/// OpenMP: parallelised over ey (X-gradient) and ex (Y-gradient).

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::compute_gradients() {
    for (auto& b : blocks) {
        const int n_dofs = b.nx * b.ny * p.N_PTS * p.N_PTS;

        // =====================================================================
        // X-gradient pass: dU/dx for all 4 conservative variables
        // =====================================================================
        #pragma omp for schedule(static)
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ex = 0; ex < b.nx; ++ex) {

                    // Extrapolate U to left/right faces
                    double UL_face[4] = {}, UR_face[4] = {};
                    for (int k = 0; k < p.N_PTS; ++k)
                        for (int v = 0; v < 4; ++v) {
                            UL_face[v] += b.U(v, ey, ex, iy, k) * basis.l_L[k];
                            UR_face[v] += b.U(v, ey, ex, iy, k) * basis.l_R[k];
                        }

                    // Get neighbour face states
                    double U_neigh_L[4], U_neigh_R[4];
                    double sig_dummy;
                    
                    const ImmersedBoundary::SurrogateFluxPoint* sfp_L = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 0, iy);
                    if (sfp_L) {
                        ImmersedBoundary::compute_sbm_state(*this, sfp_L, U_neigh_L);
                    } else {
                        get_neigh_state_x(b, ey, ex, iy, false,
                                          UL_face, 0.0, U_neigh_L, sig_dummy);
                    }
                    
                    const ImmersedBoundary::SurrogateFluxPoint* sfp_R = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 1, iy);
                    if (sfp_R) {
                        ImmersedBoundary::compute_sbm_state(*this, sfp_R, U_neigh_R);
                    } else {
                        get_neigh_state_x(b, ey, ex, iy, true,
                                          UR_face, 0.0, U_neigh_R, sig_dummy);
                    }

                    // Central interface state: U_hat = 0.5*(U_int + U_neigh)
                    double UL_hat[4], UR_hat[4];
                    for (int v = 0; v < 4; ++v) {
                        UL_hat[v] = 0.5 * (UL_face[v] + U_neigh_L[v]);
                        UR_hat[v] = 0.5 * (UR_face[v] + U_neigh_R[v]);
                    }

                    // FR gradient reconstruction at each solution point
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        for (int v = 0; v < 4; ++v) {
                            double dU_dx_loc = 0.0;
                            for (int k = 0; k < p.N_PTS; ++k)
                                dU_dx_loc += basis.D[ix][k] * b.U(v, ey, ex, iy, k);

                            double dU_dx = (dU_dx_loc
                                + (UL_hat[v] - UL_face[v]) * basis.dgl[ix]
                                + (UR_hat[v] - UR_face[v]) * basis.dgr[ix])
                                * (2.0 / b.dx);

                            int flat = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                            b.grad_Ux[v * n_dofs + flat] = dU_dx;
                        }
                    }
                }
            }
        }

        // =====================================================================
        // Y-gradient pass: dU/dy for all 4 conservative variables
        // =====================================================================
        #pragma omp for schedule(static)
        for (int ex = 0; ex < b.nx; ++ex) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                for (int ey = 0; ey < b.ny; ++ey) {

                    // Extrapolate U to bottom/top faces
                    double UB_face[4] = {}, UT_face[4] = {};
                    for (int k = 0; k < p.N_PTS; ++k)
                        for (int v = 0; v < 4; ++v) {
                            UB_face[v] += b.U(v, ey, ex, k, ix) * basis.l_L[k];
                            UT_face[v] += b.U(v, ey, ex, k, ix) * basis.l_R[k];
                        }

                    // Get neighbour face states
                    double U_neigh_B[4], U_neigh_T[4];
                    double sig_dummy;
                    
                    const ImmersedBoundary::SurrogateFluxPoint* sfp_B = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 2, ix);
                    if (sfp_B) {
                        ImmersedBoundary::compute_sbm_state(*this, sfp_B, U_neigh_B);
                    } else {
                        get_neigh_state_y(b, ey, ex, ix, false,
                                          UB_face, 0.0, U_neigh_B, sig_dummy);
                    }
                    
                    const ImmersedBoundary::SurrogateFluxPoint* sfp_T = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 3, ix);
                    if (sfp_T) {
                        ImmersedBoundary::compute_sbm_state(*this, sfp_T, U_neigh_T);
                    } else {
                        get_neigh_state_y(b, ey, ex, ix, true,
                                          UT_face, 0.0, U_neigh_T, sig_dummy);
                    }

                    // Central interface state
                    double UB_hat[4], UT_hat[4];
                    for (int v = 0; v < 4; ++v) {
                        UB_hat[v] = 0.5 * (UB_face[v] + U_neigh_B[v]);
                        UT_hat[v] = 0.5 * (UT_face[v] + U_neigh_T[v]);
                    }

                    // FR gradient reconstruction at each solution point
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int v = 0; v < 4; ++v) {
                            double dU_dy_loc = 0.0;
                            for (int k = 0; k < p.N_PTS; ++k)
                                dU_dy_loc += basis.D[iy][k] * b.U(v, ey, ex, k, ix);

                            double dU_dy = (dU_dy_loc
                                + (UB_hat[v] - UB_face[v]) * basis.dgl[iy]
                                + (UT_hat[v] - UT_face[v]) * basis.dgr[iy])
                                * (2.0 / b.dy);

                            int flat = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                            b.grad_Uy[v * n_dofs + flat] = dU_dy;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @file sweep_y.cpp
 * @brief Y-direction Flux Reconstruction sweep (tensor-product).
 *
 * Mirrors sweep_x but operates along columns. For each (ex, ix)
 * pair, sweeps over ey computing G-fluxes, face states, Riemann
 * solvers, and FR corrections. Applies Radau correction limits
 * along the Y-axis.
 *
 * OpenMP: parallelised over ex (each column is independent).
 *
 * @see Solver::sweep_x
 */

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Perform the Y-direction Flux Reconstruction sweep.
 *
 * @details
 * Accumulates the flux divergence \f$ \partial G / \partial y \f$ into the RHS vector.
 * Utilizes the FR (Correction Procedure via Reconstruction) method with 
 * tensor-product bases and Radau correction polynomials.
 *
 * Data Structures & Indexing:
 *   - `U(v, ey, ex, iy, ix)`: The global conserved state array. Read-only in this pass.
 *   - `RHS(v, ey, ex, iy, ix)`: The global explicit right-hand side array. This function 
 *     accumulates the flux divergence and interface corrections into RHS via subtraction (`-=`).
 *   - `sigma_field[get_flat_idx(ey, ex, iy, ix)]`: The global scalar entropic pressure field.
 *   - `basis.l_L`, `basis.l_R`: 1D arrays of size N_PTS used to extrapolate solution points to the bottom/top faces.
 *   - `basis.dgl`, `basis.dgr`: 1D arrays of size N_PTS containing the derivatives of the Radau correction polynomials.
 *
 * Assumptions:
 *   - The global arrays are pre-allocated and `RHS` contains the partial accumulation from `sweep_x`.
 *   - The memory access pattern is designed to be OpenMP thread-safe by assigning disjoint columns (`ex`) to different threads.
 *
 * @note Completes the divergence operator application started by Solver::sweep_x.
 */
void Solver::sweep_y() {
    for (auto& b : blocks) {
        #pragma omp for schedule(static)
        for (int ex = 0; ex < b.nx; ++ex) {
            double prev_Flux_T_comm[MAX_PTS][4];
            for (int ey = 0; ey < b.ny; ++ey) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {

                    // --- 1. Pointwise Y-flux ---
                    double G_sol[MAX_PTS][4];
                    for (int iy = 0; iy < p.N_PTS; ++iy)
                        get_flux_pointwise(b, ey, ex, iy, ix,
                                           nullptr, G_sol[iy],
                                           b.sigma_field[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)]);

                    // --- 2. Face-extrapolated states ---
                    double UB_face[4] = {}, UT_face[4] = {};
                    double sig_B_face = 0.0, sig_T_face = 0.0;
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        double s = b.sigma_field[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
                        sig_B_face += s * basis.l_L[iy];
                        sig_T_face += s * basis.l_R[iy];
                    }
                    for (int v = 0; v < 4; ++v) {
                        for (int iy = 0; iy < p.N_PTS; ++iy) {
                            UB_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_L[iy];
                            UT_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_R[iy];
                        }
                    }

                    // --- 3. Common Riemann fluxes ---
                    double Flux_B_comm[4], Flux_T_comm[4];
                    double U_neigh[4];
                    double sig_neigh;

                    const ImmersedBoundary::SurrogateFluxPoint* sfp_B = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 2, ix);
                    if (sfp_B) {
                        double u_sb[4];
                        ImmersedBoundary::compute_sbm_state(*this, sfp_B, u_sb);
                        solve_riemann(u_sb, UB_face, Flux_B_comm, 1, sig_B_face, sig_B_face);
                    } else if (ey > 0) {
                        for (int v = 0; v < 4; ++v)
                            Flux_B_comm[v] = prev_Flux_T_comm[ix][v];
                    } else {
                        get_neigh_state_y(b, ey, ex, ix, false,
                                          UB_face, sig_B_face, U_neigh, sig_neigh);
                        solve_riemann(U_neigh, UB_face, Flux_B_comm, 1,
                                      sig_neigh, sig_B_face);
                    }

                    const ImmersedBoundary::SurrogateFluxPoint* sfp_T = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 3, ix);
                    if (sfp_T) {
                        double u_sb[4];
                        ImmersedBoundary::compute_sbm_state(*this, sfp_T, u_sb);
                        solve_riemann(UT_face, u_sb, Flux_T_comm, 1, sig_T_face, sig_T_face);
                    } else {
                        get_neigh_state_y(b, ey, ex, ix, true,
                                          UT_face, sig_T_face, U_neigh, sig_neigh);
                        solve_riemann(UT_face, U_neigh, Flux_T_comm, 1,
                                      sig_T_face, sig_neigh);
                    }

                    for (int v = 0; v < 4; ++v)
                        prev_Flux_T_comm[ix][v] = Flux_T_comm[v];

                    // --- 4. Interior flux at faces ---
                    double G_B[4] = {}, G_T[4] = {};
                    for (int v = 0; v < 4; ++v) {
                        for (int iy = 0; iy < p.N_PTS; ++iy) {
                            G_B[v] += G_sol[iy][v] * basis.l_L[iy];
                            G_T[v] += G_sol[iy][v] * basis.l_R[iy];
                        }
                    }

                    // --- 5. Accumulate into RHS ---
                    for (int v = 0; v < 4; ++v) {
                        for (int iy = 0; iy < p.N_PTS; ++iy) {
                            double dg = 0.0;
                            for (int k = 0; k < p.N_PTS; ++k)
                                dg += basis.D[iy][k] * G_sol[k][v];
                            b.RHS(v, ey, ex, iy, ix) -=
                                (dg
                                 + (Flux_B_comm[v] - G_B[v]) * basis.dgl[iy]
                                 + (Flux_T_comm[v] - G_T[v]) * basis.dgr[iy])
                                * (2.0 / b.dy);
                        }
                    }
                }
            }
        }
    }
}

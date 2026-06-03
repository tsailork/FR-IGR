/// @file sweep_x.cpp
/// @brief X-direction Flux Reconstruction sweep (tensor-product).
///
/// For each element row (ey, iy), the sweep computes:
///   1. Pointwise fluxes F at every solution point.
///   2. Face-extrapolated states and σ values.
///   3. Common Riemann flux at left/right interfaces.
///   4. FR correction and accumulation into RHS.
///
/// OpenMP: parallelised over ey (each row is independent).

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/// Perform the X-direction Flux Reconstruction sweep.
///
/// @details
/// Data Structures & Indexing:
///   - `U(v, ey, ex, iy, ix)`: The global conserved state array. Read-only in this pass.
///   - `RHS(v, ey, ex, iy, ix)`: The global explicit right-hand side array. This function 
///     accumulates the flux divergence and interface corrections into RHS via subtraction (`-=`).
///   - `sigma_field[get_flat_idx(ey, ex, iy, ix)]`: The global scalar entropic pressure field.
///   - `basis.l_L`, `basis.l_R`: 1D arrays of size N_PTS used to extrapolate solution points to the left/right faces.
///   - `basis.dgl`, `basis.dgr`: 1D arrays of size N_PTS containing the derivatives of the Radau correction polynomials.
/// Assumptions:
///   - The global arrays are pre-allocated and `RHS` is zeroed or holds previous partial accumulations.
///   - The memory access pattern is designed to be OpenMP thread-safe by assigning disjoint rows (`ey`) to different threads.
void Solver::sweep_x() {
    for (auto& b : blocks) {
        #pragma omp for schedule(static)
        for (int ey = 0; ey < b.ny; ++ey) {
            double prev_Flux_R_comm[MAX_PTS][4];
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {

                    // --- 1. Pointwise X-flux at each solution point ---
                    double F_sol[MAX_PTS][4];
                    for (int ix = 0; ix < p.N_PTS; ++ix)
                        get_flux_pointwise(b, ey, ex, iy, ix,
                                           F_sol[ix], nullptr,
                                           b.sigma_field[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)]);

                    // --- 2. Face-extrapolated states ---
                    double UL_face[4] = {}, UR_face[4] = {};
                    double sig_L_face = 0.0, sig_R_face = 0.0;
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double s = b.sigma_field[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
                        sig_L_face += s * basis.l_L[ix];
                        sig_R_face += s * basis.l_R[ix];
                    }
                    for (int v = 0; v < 4; ++v) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            UL_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_L[ix];
                            UR_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_R[ix];
                        }
                    }

                    // --- 3. Common Riemann fluxes ---
                    double Flux_L_comm[4], Flux_R_comm[4];
                    double U_neigh[4];
                    double sig_neigh;

                    if (ex > 0) {
                        for (int v = 0; v < 4; ++v)
                            Flux_L_comm[v] = prev_Flux_R_comm[iy][v];
                    } else {
                        const ImmersedBoundary::SurrogateFluxPoint* sfp_L = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 0, iy);
                        if (sfp_L) {
                            double u_sb[4];
                            ImmersedBoundary::compute_sbm_state(*this, sfp_L, u_sb);
                            solve_riemann(u_sb, UL_face, Flux_L_comm, 0, sig_L_face, sig_L_face);
                        } else {
                            get_neigh_state_x(b, ey, ex, iy, false,
                                              UL_face, sig_L_face, U_neigh, sig_neigh);
                            solve_riemann(U_neigh, UL_face, Flux_L_comm, 0,
                                          sig_neigh, sig_L_face);
                        }
                    }

                    const ImmersedBoundary::SurrogateFluxPoint* sfp_R = ImmersedBoundary::get_sbm_face(b.id, ey, ex, 1, iy);
                    if (sfp_R) {
                        double u_sb[4];
                        ImmersedBoundary::compute_sbm_state(*this, sfp_R, u_sb);
                        solve_riemann(UR_face, u_sb, Flux_R_comm, 0, sig_R_face, sig_R_face);
                    } else {
                        get_neigh_state_x(b, ey, ex, iy, true,
                                          UR_face, sig_R_face, U_neigh, sig_neigh);
                        solve_riemann(UR_face, U_neigh, Flux_R_comm, 0,
                                      sig_R_face, sig_neigh);
                    }

                    for (int v = 0; v < 4; ++v)
                        prev_Flux_R_comm[iy][v] = Flux_R_comm[v];

                    // --- 4. Interior flux at faces (for correction) ---
                    double F_L[4] = {}, F_R[4] = {};
                    for (int v = 0; v < 4; ++v) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            F_L[v] += F_sol[ix][v] * basis.l_L[ix];
                            F_R[v] += F_sol[ix][v] * basis.l_R[ix];
                        }
                    }

                    // --- 5. Accumulate into RHS ---
                    for (int v = 0; v < 4; ++v) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            double df = 0.0;
                            for (int k = 0; k < p.N_PTS; ++k)
                                df += basis.D[ix][k] * F_sol[k][v];
                            b.RHS(v, ey, ex, iy, ix) -=
                                (df
                                 + (Flux_L_comm[v] - F_L[v]) * basis.dgl[ix]
                                 + (Flux_R_comm[v] - F_R[v]) * basis.dgr[ix])
                                * (2.0 / b.dx);
                        }
                    }
                }
            }
        }
    }
}

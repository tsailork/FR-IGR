/// @file sweep_y.cpp
/// @brief Y-direction Flux Reconstruction sweep (tensor-product).
///
/// Mirrors sweep_x but operates along columns.  For each (ex, ix)
/// pair, sweeps over ey computing G-fluxes, face states, Riemann
/// solvers, and FR corrections.
///
/// OpenMP: parallelised over ex (each column is independent).

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/// Perform the Y-direction Flux Reconstruction sweep.
///
/// @details
/// Data Structures & Indexing:
///   - `U(v, ey, ex, iy, ix)`: The global conserved state array. Read-only in this pass.
///   - `RHS(v, ey, ex, iy, ix)`: The global explicit right-hand side array. This function 
///     accumulates the flux divergence and interface corrections into RHS via subtraction (`-=`).
///   - `sigma_field[get_flat_idx(ey, ex, iy, ix)]`: The global scalar entropic pressure field.
///   - `basis.l_L`, `basis.l_R`: 1D arrays of size N_PTS used to extrapolate solution points to the bottom/top faces.
///   - `basis.dgl`, `basis.dgr`: 1D arrays of size N_PTS containing the derivatives of the Radau correction polynomials.
/// Assumptions:
///   - The global arrays are pre-allocated and `RHS` contains the partial accumulation from `sweep_x`.
///   - The memory access pattern is designed to be OpenMP thread-safe by assigning disjoint columns (`ex`) to different threads.
void Solver::sweep_y() {
    #pragma omp parallel for schedule(static)
    for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {

                // --- 1. Pointwise Y-flux ---
                double G_sol[MAX_PTS][4];
                for (int iy = 0; iy < p.N_PTS; ++iy)
                    get_flux_pointwise(ey, ex, iy, ix,
                                       nullptr, G_sol[iy],
                                       sigma_field[get_flat_idx(ey, ex, iy, ix)]);

                // --- 2. Face-extrapolated states ---
                double UB_face[4] = {}, UT_face[4] = {};
                double sig_B_face = 0.0, sig_T_face = 0.0;
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double s = sigma_field[get_flat_idx(ey, ex, iy, ix)];
                    sig_B_face += s * basis.l_L[iy];
                    sig_T_face += s * basis.l_R[iy];
                    for (int v = 0; v < 4; ++v) {
                        UB_face[v] += U(v, ey, ex, iy, ix) * basis.l_L[iy];
                        UT_face[v] += U(v, ey, ex, iy, ix) * basis.l_R[iy];
                    }
                }

                // --- 3. Common Riemann fluxes ---
                double Flux_B_comm[4], Flux_T_comm[4];
                double U_neigh[4];
                double sig_neigh;

                get_neigh_state_y(ey, ex, ix, false,
                                  UB_face, sig_B_face, U_neigh, sig_neigh);
                solve_riemann(U_neigh, UB_face, Flux_B_comm, 1,
                              sig_neigh, sig_B_face);

                get_neigh_state_y(ey, ex, ix, true,
                                  UT_face, sig_T_face, U_neigh, sig_neigh);
                solve_riemann(UT_face, U_neigh, Flux_T_comm, 1,
                              sig_T_face, sig_neigh);

                // --- 4. Interior flux at faces ---
                double G_B[4] = {}, G_T[4] = {};
                for (int iy = 0; iy < p.N_PTS; ++iy)
                    for (int v = 0; v < 4; ++v) {
                        G_B[v] += G_sol[iy][v] * basis.l_L[iy];
                        G_T[v] += G_sol[iy][v] * basis.l_R[iy];
                    }

                // --- 5. Accumulate into RHS ---
                for (int iy = 0; iy < p.N_PTS; ++iy)
                    for (int v = 0; v < 4; ++v) {
                        double dg = 0.0;
                        for (int k = 0; k < p.N_PTS; ++k)
                            dg += basis.D[iy][k] * G_sol[k][v];
                        RHS(v, ey, ex, iy, ix) -=
                            (dg
                             + (Flux_B_comm[v] - G_B[v]) * basis.dgl[iy]
                             + (Flux_T_comm[v] - G_T[v]) * basis.dgr[iy])
                            * (2.0 / dy);
                    }
            }
        }
    }
}

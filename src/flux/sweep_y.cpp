/**
 * @file sweep_y.cpp
 * @brief Y-direction Flux Reconstruction sweep (tensor-product) on decoupled Cells.
 */

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::sweep_y() {
    // =========================================================================
    // Pass 1: Local & Conforming Sweep (highly optimized, vectorizable)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        for (int ix = 0; ix < p.N_PTS; ++ix) {

            // --- 1. Pointwise Y-flux ---
            double G_sol[MAX_PTS][4];
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                get_flux_pointwise_cell(*c, iy, ix,
                                        nullptr, G_sol[iy],
                                        c->sigma_field[iy * p.N_PTS + ix]);
            }

            // --- 2. Face-extrapolated states ---
            double UB_face[4] = {}, UT_face[4] = {};
            double sig_B_face = 0.0, sig_T_face = 0.0;
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double s = c->sigma_field[iy * p.N_PTS + ix];
                sig_B_face += s * basis.l_L[iy];
                sig_T_face += s * basis.l_R[iy];
            }
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    UB_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[iy];
                    UT_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[iy];
                }
            }

            // --- 3. Common Riemann fluxes (Local, Conforming & Boundary) ---
            double Flux_B_local[4] = {}, Flux_T_local[4] = {};
            double U_neigh[4];
            double sig_neigh;

            // Bottom Face (2)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_B = ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 2, ix);
            if (sfp_B) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_B, u_sb);
                double Flux_B_comm[4];
                solve_riemann(u_sb, UB_face, Flux_B_comm, 1, sig_B_face, sig_B_face);
                for (int v = 0; v < 4; ++v) Flux_B_local[v] = Flux_B_comm[v];
            } else if (c->neighbors[2] && c->neighbors[2]->level == c->level) {
                Cell* nc = c->neighbors[2];
                char nface = c->neighbor_faces[2];
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[k * p.N_PTS + ix] * weights[k];
                }
                double Flux_B_comm[4];
                solve_riemann(U_neigh, UB_face, Flux_B_comm, 1, sig_neigh, sig_B_face);
                for (int v = 0; v < 4; ++v) Flux_B_local[v] = Flux_B_comm[v];
            } else if (c->is_boundary[2]) {
                get_neigh_state_cell(*c, ix, false,
                                     UB_face, sig_B_face, U_neigh, sig_neigh, 1);
                double Flux_B_comm[4];
                solve_riemann(U_neigh, UB_face, Flux_B_comm, 1,
                              sig_neigh, sig_B_face);
                for (int v = 0; v < 4; ++v) Flux_B_local[v] = Flux_B_comm[v];
            }

            // Top Face (3)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_T = ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 3, ix);
            if (sfp_T) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_T, u_sb);
                double Flux_T_comm[4];
                solve_riemann(UT_face, u_sb, Flux_T_comm, 1, sig_T_face, sig_T_face);
                for (int v = 0; v < 4; ++v) Flux_T_local[v] = Flux_T_comm[v];
            } else if (c->neighbors[3] && c->neighbors[3]->level == c->level) {
                Cell* nc = c->neighbors[3];
                char nface = c->neighbor_faces[3];
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[k * p.N_PTS + ix] * weights[k];
                }
                double Flux_T_comm[4];
                solve_riemann(UT_face, U_neigh, Flux_T_comm, 1, sig_T_face, sig_neigh);
                for (int v = 0; v < 4; ++v) Flux_T_local[v] = Flux_T_comm[v];
            } else if (c->is_boundary[3]) {
                get_neigh_state_cell(*c, ix, true,
                                     UT_face, sig_T_face, U_neigh, sig_neigh, 1);
                double Flux_T_comm[4];
                solve_riemann(UT_face, U_neigh, Flux_T_comm, 1,
                              sig_T_face, sig_neigh);
                for (int v = 0; v < 4; ++v) Flux_T_local[v] = Flux_T_comm[v];
            }

            // --- 4. Interior flux at faces ---
            double G_B[4] = {}, G_T[4] = {};
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    G_B[v] += G_sol[iy][v] * basis.l_L[iy];
                    G_T[v] += G_sol[iy][v] * basis.l_R[iy];
                }
            }

            // --- 5. Accumulate into RHS (single-write, extremely fast) ---
            for (int v = 0; v < 4; ++v) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double dg = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        dg += basis.D[iy][k] * G_sol[k][v];
                    
                    c->get_RHS(v, iy, ix, p.N_PTS) -= 
                        (dg
                         + (Flux_B_local[v] - G_B[v]) * basis.dgl[iy]
                         + (Flux_T_local[v] - G_T[v]) * basis.dgr[iy])
                        * (2.0 / c->dy);
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Interface Sweep (sparse, only active near hanging nodes)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];

        // Bottom Face (2) non-conforming coarser neighbor
        if (c->neighbors[2] && c->neighbors[2]->level < c->level) {
            Cell* nc = c->neighbors[2];
            char nface = c->neighbor_faces[2];
            int child_idx = c->ex & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double UB_face[4] = {};
                double sig_B_face = 0.0;
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_B_face += s * basis.l_L[iy];
                    for (int v = 0; v < 4; ++v) {
                        UB_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[iy];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                        }
                        sig_coarse_face[kx] += nc->sigma_field[ky * p.N_PTS + kx] * weights[ky];
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[kx][ix] * U_coarse_face[v][kx];
                    }
                    sig_neigh += P[kx][ix] * sig_coarse_face[kx];
                }

                double Flux_B_comm[4];
                solve_riemann(U_neigh, UB_face, Flux_B_comm, 1, sig_neigh, sig_B_face);

                // Update fine cell
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_B_comm[v] * basis.dgl[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_B_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }

        // Top Face (3) non-conforming coarser neighbor
        if (c->neighbors[3] && c->neighbors[3]->level < c->level) {
            Cell* nc = c->neighbors[3];
            char nface = c->neighbor_faces[3];
            int child_idx = c->ex & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'B') ? basis.dgl.data() : basis.dgr.data();

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double UT_face[4] = {};
                double sig_T_face = 0.0;
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_T_face += s * basis.l_R[iy];
                    for (int v = 0; v < 4; ++v) {
                        UT_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[iy];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                        }
                        sig_coarse_face[kx] += nc->sigma_field[ky * p.N_PTS + kx] * weights[ky];
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[kx][ix] * U_coarse_face[v][kx];
                    }
                    sig_neigh += P[kx][ix] * sig_coarse_face[kx];
                }

                double Flux_T_comm[4];
                solve_riemann(UT_face, U_neigh, Flux_T_comm, 1, sig_T_face, sig_neigh);

                // Update fine cell
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_T_comm[v] * basis.dgr[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_T_comm[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }
    }
}

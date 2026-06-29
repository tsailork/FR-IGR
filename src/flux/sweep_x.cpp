/**
 * @file sweep_x.cpp
 * @brief X-direction Flux Reconstruction sweep (tensor-product) on decoupled Cells.
 */

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::sweep_x() {
    // =========================================================================
    // Pass 1: Local & Conforming Sweep (highly optimized, vectorizable)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        for (int iy = 0; iy < p.N_PTS; ++iy) {

            // --- 1. Pointwise X-flux at each solution point ---
            double F_sol[MAX_PTS][4];
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                get_flux_pointwise_cell(*c, iy, ix,
                                        F_sol[ix], nullptr,
                                        c->sigma_field[iy * p.N_PTS + ix]);
            }

            // --- 2. Face-extrapolated states ---
            double UL_face[4] = {}, UR_face[4] = {};
            double sig_L_face = 0.0, sig_R_face = 0.0;
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double s = c->sigma_field[iy * p.N_PTS + ix];
                sig_L_face += s * basis.l_L[ix];
                sig_R_face += s * basis.l_R[ix];
            }
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    UR_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[ix];
                }
            }

            // --- 3. Common Riemann fluxes (Local, Conforming & Boundary) ---
            double Flux_L_local[4] = {}, Flux_R_local[4] = {};
            double U_neigh[4];
            double sig_neigh;

            // Left Face (0)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_L = ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 0, iy);
            if (sfp_L) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_L, u_sb);
                double Flux_L_comm[4];
                solve_riemann(u_sb, UL_face, Flux_L_comm, 0, sig_L_face, sig_L_face);
                for (int v = 0; v < 4; ++v) Flux_L_local[v] = Flux_L_comm[v];
            } else if (c->neighbors[0] && c->neighbors[0]->level == c->level) {
                Cell* nc = c->neighbors[0];
                char nface = c->neighbor_faces[0];
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[iy * p.N_PTS + k] * weights[k];
                }
                double Flux_L_comm[4];
                solve_riemann(U_neigh, UL_face, Flux_L_comm, 0, sig_neigh, sig_L_face);
                for (int v = 0; v < 4; ++v) Flux_L_local[v] = Flux_L_comm[v];
            } else if (c->is_boundary[0]) {
                get_neigh_state_cell(*c, iy, false,
                                     UL_face, sig_L_face, U_neigh, sig_neigh, 0);
                double Flux_L_comm[4];
                solve_riemann(U_neigh, UL_face, Flux_L_comm, 0,
                              sig_neigh, sig_L_face);
                for (int v = 0; v < 4; ++v) Flux_L_local[v] = Flux_L_comm[v];
            }

            // Right Face (1)
            const ImmersedBoundary::SurrogateFluxPoint* sfp_R = ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 1, iy);
            if (sfp_R) {
                double u_sb[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_R, u_sb);
                double Flux_R_comm[4];
                solve_riemann(UR_face, u_sb, Flux_R_comm, 0, sig_R_face, sig_R_face);
                for (int v = 0; v < 4; ++v) Flux_R_local[v] = Flux_R_comm[v];
            } else if (c->neighbors[1] && c->neighbors[1]->level == c->level) {
                Cell* nc = c->neighbors[1];
                char nface = c->neighbor_faces[1];
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                for (int v = 0; v < 4; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                    sig_neigh += nc->sigma_field[iy * p.N_PTS + k] * weights[k];
                }
                double Flux_R_comm[4];
                solve_riemann(UR_face, U_neigh, Flux_R_comm, 0, sig_R_face, sig_neigh);
                for (int v = 0; v < 4; ++v) Flux_R_local[v] = Flux_R_comm[v];
            } else if (c->is_boundary[1]) {
                get_neigh_state_cell(*c, iy, true,
                                     UR_face, sig_R_face, U_neigh, sig_neigh, 0);
                double Flux_R_comm[4];
                solve_riemann(UR_face, U_neigh, Flux_R_comm, 0,
                              sig_R_face, sig_neigh);
                for (int v = 0; v < 4; ++v) Flux_R_local[v] = Flux_R_comm[v];
            }

            // --- 4. Interior flux at faces (for correction) ---
            double F_L[4] = {}, F_R[4] = {};
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    F_L[v] += F_sol[ix][v] * basis.l_L[ix];
                    F_R[v] += F_sol[ix][v] * basis.l_R[ix];
                }
            }

            // --- 5. Accumulate into RHS (single-write, extremely fast) ---
            for (int v = 0; v < 4; ++v) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double df = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        df += basis.D[ix][k] * F_sol[k][v];
                    
                    c->get_RHS(v, iy, ix, p.N_PTS) -= 
                        (df
                         + (Flux_L_local[v] - F_L[v]) * basis.dgl[ix]
                         + (Flux_R_local[v] - F_R[v]) * basis.dgr[ix])
                        * (2.0 / c->dx);
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

        // Left Face (0) non-conforming coarser neighbor
        if (c->neighbors[0] && c->neighbors[0]->level < c->level) {
            Cell* nc = c->neighbors[0];
            char nface = c->neighbor_faces[0];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double UL_face[4] = {};
                double sig_L_face = 0.0;
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_L_face += s * basis.l_L[ix];
                    for (int v = 0; v < 4; ++v) {
                        UL_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_L[ix];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                        }
                        sig_coarse_face[ky] += nc->sigma_field[ky * p.N_PTS + kx] * weights[kx];
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[ky][iy] * U_coarse_face[v][ky];
                    }
                    sig_neigh += P[ky][iy] * sig_coarse_face[ky];
                }

                double Flux_L_comm[4];
                solve_riemann(U_neigh, UL_face, Flux_L_comm, 0, sig_neigh, sig_L_face);

                // Update fine cell
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_L_comm[v] * basis.dgl[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_L_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }

        // Right Face (1) non-conforming coarser neighbor
        if (c->neighbors[1] && c->neighbors[1]->level < c->level) {
            Cell* nc = c->neighbors[1];
            char nface = c->neighbor_faces[1];
            int child_idx = c->ey & 1;
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == 'L') ? basis.dgl.data() : basis.dgr.data();

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                double UR_face[4] = {};
                double sig_R_face = 0.0;
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double s = c->sigma_field[iy * p.N_PTS + ix];
                    sig_R_face += s * basis.l_R[ix];
                    for (int v = 0; v < 4; ++v) {
                        UR_face[v] += c->get_U(v, iy, ix, p.N_PTS) * basis.l_R[ix];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                        }
                        sig_coarse_face[ky] += nc->sigma_field[ky * p.N_PTS + kx] * weights[kx];
                    }
                }

                double U_neigh[4] = {};
                double sig_neigh = 0.0;
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh[v] += P[ky][iy] * U_coarse_face[v][ky];
                    }
                    sig_neigh += P[ky][iy] * sig_coarse_face[ky];
                }

                double Flux_R_comm[4];
                solve_riemann(UR_face, U_neigh, Flux_R_comm, 0, sig_R_face, sig_neigh);

                // Update fine cell
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    for (int v = 0; v < 4; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, p.N_PTS) -= Flux_R_comm[v] * basis.dgr[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, ky, kx, p.N_PTS) -= factor * Flux_R_comm[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }
    }
}

/**
 * @file gradient.cpp
 * @brief BR2 Phase 1 — compute gradients of conservative variables ∇U cell-locally.
 */

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::compute_gradients() {
    // =========================================================================
    // X-gradient pass: dU/dx
    // =========================================================================

    // Pass 1: Local & Conforming Pass (highly optimized, vectorizable)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            double UL_face[4] = {}, UR_face[4] = {};
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v) {
                    UL_face[v] += c->get_U(v, iy, k, p.N_PTS) * basis.l_L[k];
                    UR_face[v] += c->get_U(v, iy, k, p.N_PTS) * basis.l_R[k];
                }
            }

            double jump_L[4] = {}, jump_R[4] = {};

            // Left Face (0) conforming/SFP/boundary
            const ImmersedBoundary::SurrogateFluxPoint* sfp_L = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 0, iy) : nullptr;
            if (sfp_L) {
                double U_neigh_L[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_L, U_neigh_L);
                for (int v = 0; v < 4; ++v) jump_L[v] = 0.5 * (U_neigh_L[v] - UL_face[v]);
            } else if (c->neighbors[0] && c->neighbors[0]->level == c->level) {
                Cell* nc = c->neighbors[0];
                char nface = c->neighbor_faces[0];
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                double U_neigh_L[4] = {};
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh_L[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                }
                for (int v = 0; v < 4; ++v) jump_L[v] = 0.5 * (U_neigh_L[v] - UL_face[v]);
            } else if (c->is_boundary[0]) {
                double U_neigh_L[4], sig_dummy;
                get_neigh_state_cell(*c, iy, false, UL_face, 0.0, U_neigh_L, sig_dummy, 0);
                for (int v = 0; v < 4; ++v) jump_L[v] = 0.5 * (U_neigh_L[v] - UL_face[v]);
            }

            // Right Face (1) conforming/SFP/boundary
            const ImmersedBoundary::SurrogateFluxPoint* sfp_R = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 1, iy) : nullptr;
            if (sfp_R) {
                double U_neigh_R[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_R, U_neigh_R);
                for (int v = 0; v < 4; ++v) jump_R[v] = 0.5 * (U_neigh_R[v] - UR_face[v]);
            } else if (c->neighbors[1] && c->neighbors[1]->level == c->level) {
                Cell* nc = c->neighbors[1];
                char nface = c->neighbor_faces[1];
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                double U_neigh_R[4] = {};
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh_R[v] += nc->get_U(v, iy, k, p.N_PTS) * weights[k];
                }
                for (int v = 0; v < 4; ++v) jump_R[v] = 0.5 * (U_neigh_R[v] - UR_face[v]);
            } else if (c->is_boundary[1]) {
                double U_neigh_R[4], sig_dummy;
                get_neigh_state_cell(*c, iy, true, UR_face, 0.0, U_neigh_R, sig_dummy, 0);
                for (int v = 0; v < 4; ++v) jump_R[v] = 0.5 * (U_neigh_R[v] - UR_face[v]);
            }

            for (int ix = 0; ix < p.N_PTS; ++ix) {
                for (int v = 0; v < 4; ++v) {
                    double dU_dx_loc = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        dU_dx_loc += basis.D[ix][k] * c->get_U(v, iy, k, p.N_PTS);

                    int flat = iy * p.N_PTS + ix;
                    c->grad_Ux[v * p.N_PTS * p.N_PTS + flat] = 
                        (dU_dx_loc
                         + jump_L[v] * basis.dgl[ix]
                         + jump_R[v] * basis.dgr[ix])
                        * (2.0 / c->dx);
                }
            }
        }
    }

    // Pass 2: Non-Conforming Restriction Pass (sparse, only active near hanging nodes)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

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
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v) {
                        UL_face[v] += c->get_U(v, iy, k, p.N_PTS) * basis.l_L[k];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                        }
                    }
                }

                double U_neigh_L[4] = {};
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh_L[v] += P[ky][iy] * U_coarse_face[v][ky];
                    }
                }

                double jump_L[4] = {};
                for (int v = 0; v < 4; ++v) {
                    jump_L[v] = 0.5 * (U_neigh_L[v] - UL_face[v]);
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        int flat = iy * p.N_PTS + ix;
                        #pragma omp atomic
                        c->grad_Ux[v * p.N_PTS * p.N_PTS + flat] += jump_L[v] * basis.dgl[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict jump and subtract from coarse cell nc Right face
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->grad_Ux[v * p.N_PTS * p.N_PTS + ky * p.N_PTS + kx] -= factor * jump_L[v] * dg_nc[kx] * (2.0 / nc->dx);
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
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v) {
                        UR_face[v] += c->get_U(v, iy, k, p.N_PTS) * basis.l_R[k];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'L') ? basis.l_L.data() : basis.l_R.data();
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][ky] += nc->get_U(v, ky, kx, p.N_PTS) * weights[kx];
                        }
                    }
                }

                double U_neigh_R[4] = {};
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh_R[v] += P[ky][iy] * U_coarse_face[v][ky];
                    }
                }

                double jump_R[4] = {};
                for (int v = 0; v < 4; ++v) {
                    jump_R[v] = 0.5 * (U_neigh_R[v] - UR_face[v]);
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        int flat = iy * p.N_PTS + ix;
                        #pragma omp atomic
                        c->grad_Ux[v * p.N_PTS * p.N_PTS + flat] += jump_R[v] * basis.dgr[ix] * (2.0 / c->dx);
                    }
                }

                // Restrict jump and subtract from coarse cell nc Left face
                for (int ky = 0; ky < p.N_PTS; ++ky) {
                    double factor = R[iy][ky];
                    for (int kx = 0; kx < p.N_PTS; ++kx) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->grad_Ux[v * p.N_PTS * p.N_PTS + ky * p.N_PTS + kx] -= factor * jump_R[v] * dg_nc[kx] * (2.0 / nc->dx);
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Y-gradient pass: dU/dy
    // =========================================================================

    // Pass 1: Local & Conforming Pass (highly optimized, vectorizable)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;
        for (int ix = 0; ix < p.N_PTS; ++ix) {
            double UB_face[4] = {}, UT_face[4] = {};
            for (int k = 0; k < p.N_PTS; ++k) {
                for (int v = 0; v < 4; ++v) {
                    UB_face[v] += c->get_U(v, k, ix, p.N_PTS) * basis.l_L[k];
                    UT_face[v] += c->get_U(v, k, ix, p.N_PTS) * basis.l_R[k];
                }
            }

            double jump_B[4] = {}, jump_T[4] = {};

            // Bottom Face (2) conforming/SFP/boundary
            const ImmersedBoundary::SurrogateFluxPoint* sfp_B = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 2, ix) : nullptr;
            if (sfp_B) {
                double U_neigh_B[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_B, U_neigh_B);
                for (int v = 0; v < 4; ++v) jump_B[v] = 0.5 * (U_neigh_B[v] - UB_face[v]);
            } else if (c->neighbors[2] && c->neighbors[2]->level == c->level) {
                Cell* nc = c->neighbors[2];
                char nface = c->neighbor_faces[2];
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                double U_neigh_B[4] = {};
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh_B[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                }
                for (int v = 0; v < 4; ++v) jump_B[v] = 0.5 * (U_neigh_B[v] - UB_face[v]);
            } else if (c->is_boundary[2]) {
                double U_neigh_B[4], sig_dummy;
                get_neigh_state_cell(*c, ix, false, UB_face, 0.0, U_neigh_B, sig_dummy, 1);
                for (int v = 0; v < 4; ++v) jump_B[v] = 0.5 * (U_neigh_B[v] - UB_face[v]);
            }

            // Top Face (3) conforming/SFP/boundary
            const ImmersedBoundary::SurrogateFluxPoint* sfp_T = (p.ENABLE_IB && p.IB_METHOD == "SBM") ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, 3, ix) : nullptr;
            if (sfp_T) {
                double U_neigh_T[4];
                ImmersedBoundary::compute_sbm_state(*this, sfp_T, U_neigh_T);
                for (int v = 0; v < 4; ++v) jump_T[v] = 0.5 * (U_neigh_T[v] - UT_face[v]);
            } else if (c->neighbors[3] && c->neighbors[3]->level == c->level) {
                Cell* nc = c->neighbors[3];
                char nface = c->neighbor_faces[3];
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                double U_neigh_T[4] = {};
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v)
                        U_neigh_T[v] += nc->get_U(v, k, ix, p.N_PTS) * weights[k];
                }
                for (int v = 0; v < 4; ++v) jump_T[v] = 0.5 * (U_neigh_T[v] - UT_face[v]);
            } else if (c->is_boundary[3]) {
                double U_neigh_T[4], sig_dummy;
                get_neigh_state_cell(*c, ix, true, UT_face, 0.0, U_neigh_T, sig_dummy, 1);
                for (int v = 0; v < 4; ++v) jump_T[v] = 0.5 * (U_neigh_T[v] - UT_face[v]);
            }

            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int v = 0; v < 4; ++v) {
                    double dU_dy_loc = 0.0;
                    for (int k = 0; k < p.N_PTS; ++k)
                        dU_dy_loc += basis.D[iy][k] * c->get_U(v, k, ix, p.N_PTS);

                    int flat = iy * p.N_PTS + ix;
                    c->grad_Uy[v * p.N_PTS * p.N_PTS + flat] = 
                        (dU_dy_loc
                         + jump_B[v] * basis.dgl[iy]
                         + jump_T[v] * basis.dgr[iy])
                        * (2.0 / c->dy);
                }
            }
        }
    }

    // Pass 2: Non-Conforming Restriction Pass (sparse, only active near hanging nodes)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

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
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v) {
                        UB_face[v] += c->get_U(v, k, ix, p.N_PTS) * basis.l_L[k];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                        }
                    }
                }

                double U_neigh_B[4] = {};
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh_B[v] += P[kx][ix] * U_coarse_face[v][kx];
                    }
                }

                double jump_B[4] = {};
                for (int v = 0; v < 4; ++v) {
                    jump_B[v] = 0.5 * (U_neigh_B[v] - UB_face[v]);
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        int flat = iy * p.N_PTS + ix;
                        #pragma omp atomic
                        c->grad_Uy[v * p.N_PTS * p.N_PTS + flat] += jump_B[v] * basis.dgl[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict jump and subtract from coarse cell nc Top face
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->grad_Uy[v * p.N_PTS * p.N_PTS + ky * p.N_PTS + kx] -= factor * jump_B[v] * dg_nc[ky] * (2.0 / nc->dy);
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
                for (int k = 0; k < p.N_PTS; ++k) {
                    for (int v = 0; v < 4; ++v) {
                        UT_face[v] += c->get_U(v, k, ix, p.N_PTS) * basis.l_R[k];
                    }
                }

                double U_coarse_face[4][MAX_PTS] = {};
                const double* weights = (nface == 'B') ? basis.l_L.data() : basis.l_R.data();
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            U_coarse_face[v][kx] += nc->get_U(v, ky, kx, p.N_PTS) * weights[ky];
                        }
                    }
                }

                double U_neigh_T[4] = {};
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    for (int v = 0; v < 4; ++v) {
                        U_neigh_T[v] += P[kx][ix] * U_coarse_face[v][kx];
                    }
                }

                double jump_T[4] = {};
                for (int v = 0; v < 4; ++v) {
                    jump_T[v] = 0.5 * (U_neigh_T[v] - UT_face[v]);
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        int flat = iy * p.N_PTS + ix;
                        #pragma omp atomic
                        c->grad_Uy[v * p.N_PTS * p.N_PTS + flat] += jump_T[v] * basis.dgr[iy] * (2.0 / c->dy);
                    }
                }

                // Restrict jump and subtract from coarse cell nc Bottom face
                for (int kx = 0; kx < p.N_PTS; ++kx) {
                    double factor = R[ix][kx];
                    for (int ky = 0; ky < p.N_PTS; ++ky) {
                        for (int v = 0; v < 4; ++v) {
                            #pragma omp atomic
                            nc->grad_Uy[v * p.N_PTS * p.N_PTS + ky * p.N_PTS + kx] -= factor * jump_T[v] * dg_nc[ky] * (2.0 / nc->dy);
                        }
                    }
                }
            }
        }
    }
}

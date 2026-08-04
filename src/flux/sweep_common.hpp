/**
 * @file sweep_common.hpp
 * @brief Unified 2D Flux Reconstruction inviscid sweep, parameterized by direction.
 *
 * @details
 * This header implements a single generic `inviscid_sweep_2d<Dir>()` function
 * that replaces the duplicated `sweep_x()` and `sweep_y()` implementations.
 * Direction-specific quantities (face indices, neighbor face characters, cell
 * metrics, loop orderings) are resolved at compile time via `if constexpr`.
 *
 * Dir = 0 → X-sweep (faces 0,1; metric dx; sweep along ix, line along iy)
 * Dir = 1 → Y-sweep (faces 2,3; metric dy; sweep along iy, line along ix)
 *
 * @note 3D sweeps are NOT handled here. They remain in their dedicated files
 *       due to the additional complexity of 2D mortar projections.
 */

#pragma once

#include "../core/solver.hpp"
#include "../ib/sbm_geometry.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// =========================================================================
// Direction traits — compile-time constants for directional parametrization
// =========================================================================

/// Face index for the "low" face (left=0, bottom=2)
template<int Dir> constexpr int FACE_LO = Dir * 2;
/// Face index for the "high" face (right=1, top=3)
template<int Dir> constexpr int FACE_HI = Dir * 2 + 1;

/// Neighbor face character for "low" side ('L' for X, 'B' for Y)
template<int Dir> constexpr char NFACE_LO = (Dir == 0) ? 'L' : 'B';
/// Neighbor face character for "high" side ('R' for X, 'T' for Y)
template<int Dir> constexpr char NFACE_HI = (Dir == 0) ? 'R' : 'T';

// =========================================================================
// Directional access helpers
// =========================================================================

/// Cell metric in sweep direction
inline double cell_metric(const Cell2D& c, int dir) {
    return (dir == 0) ? c.dx : c.dy;
}

/// Flatten 2D index: always stored as iy * N + ix
inline int flat2d(int iy, int ix, int N) { return iy * N + ix; }

/// Convert (line_idx, sweep_idx) to (iy, ix) depending on direction
/// Dir=0: line=iy, sweep=ix  → (iy, ix) = (line, sweep)
/// Dir=1: line=ix, sweep=iy  → (iy, ix) = (sweep, line)
template<int Dir>
inline void dir_to_ij(int line, int sweep, int& iy, int& ix) {
    if constexpr (Dir == 0) { iy = line; ix = sweep; }
    else                     { iy = sweep; ix = line; }
}

/// PPR velocity component index (1=u for X-sweep, 2=v for Y-sweep)
template<int Dir> constexpr int PPR_VEL_IDX = Dir + 1;

/// Non-conforming child index: which transverse coordinate bit to check
/// Dir=0 (X-sweep): child_idx = c->ey & 1 (transverse = Y)
/// Dir=1 (Y-sweep): child_idx = c->ex & 1 (transverse = X)
template<int Dir>
inline int nc_child_idx(const Cell2D& c) {
    if constexpr (Dir == 0) return c.ey & 1;
    else                     return c.ex & 1;
}

// =========================================================================
// Unified 2D inviscid sweep
// =========================================================================

/**
 * @brief Generic 2D inviscid FR sweep in direction Dir.
 *
 * @tparam Dir Sweep direction: 0 = X, 1 = Y.
 * @param solver Reference to the 2D solver.
 */
template<int Dir>
void inviscid_sweep_2d(Solver& solver) {
    const Parameters& p  = solver.p;
    const Basis&  basis  = solver.basis;
    const int N = p.N_PTS;
    constexpr int NV = 4; // 2D Euler variables

    // =========================================================================
    // Pass 1: Local & Conforming Sweep
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < solver.cells.size(); ++i) {
        Cell2D* c = solver.cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        const double inv_h2 = 2.0 / cell_metric(*c, Dir);

        for (int line = 0; line < N; ++line) {

            // --- 1. Pointwise flux at each solution point along sweep direction ---
            double Flux_sol[MAX_PTS][NV];
            double Flux_sol_S[MAX_PTS] = {};
            for (int sw = 0; sw < N; ++sw) {
                int iy, ix;
                dir_to_ij<Dir>(line, sw, iy, ix);

                // Compute the directional flux component
                if constexpr (Dir == 0) {
                    solver.get_flux_pointwise_cell(*c, iy, ix,
                                                    Flux_sol[sw], nullptr,
                                                    c->sigma_field[flat2d(iy, ix, N)]);
                } else {
                    solver.get_flux_pointwise_cell(*c, iy, ix,
                                                    nullptr, Flux_sol[sw],
                                                    c->sigma_field[flat2d(iy, ix, N)]);
                }

                if (p.ENABLE_PPR) {
                    double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, N));
                    double vel = c->get_U(PPR_VEL_IDX<Dir>, iy, ix, N) / rho;
                    Flux_sol_S[sw] = vel * c->S_field[flat2d(iy, ix, N)];
                }
            }

            // --- 2. Face-extrapolated states ---
            double U_lo[NV] = {}, U_hi[NV] = {};
            double sig_lo = 0.0, sig_hi = 0.0;
            double S_lo = 0.0, S_hi = 0.0;
            #pragma omp simd reduction(+:sig_lo,sig_hi,S_lo,S_hi)
            for (int sw = 0; sw < N; ++sw) {
                int iy, ix;
                dir_to_ij<Dir>(line, sw, iy, ix);
                int flat = flat2d(iy, ix, N);

                double s = c->sigma_field[flat];
                sig_lo += s * basis.l_L[sw];
                sig_hi += s * basis.l_R[sw];
                if (p.ENABLE_PPR) {
                    S_lo += c->S_field[flat] * basis.l_L[sw];
                    S_hi += c->S_field[flat] * basis.l_R[sw];
                }
            }
            for (int v = 0; v < NV; ++v) {
                for (int sw = 0; sw < N; ++sw) {
                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    U_lo[v] += c->get_U(v, iy, ix, N) * basis.l_L[sw];
                    U_hi[v] += c->get_U(v, iy, ix, N) * basis.l_R[sw];
                }
            }

            // --- 3. Common Riemann fluxes ---
            double Flux_lo[NV] = {}, Flux_hi[NV] = {};
            double Flux_S_lo = 0.0, Flux_S_hi = 0.0;
            double U_neigh[NV];
            double sig_neigh;

            // Low face
            const ImmersedBoundary::SurrogateFluxPoint* sfp_lo =
                (p.ENABLE_IB && p.IB_METHOD == "SBM")
                    ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, FACE_LO<Dir>, line)
                    : nullptr;
            if (sfp_lo) {
                double u_sb[NV];
                ImmersedBoundary::compute_sbm_state(solver, sfp_lo, u_sb);
                double S_sb = Solver::compute_wall_phantom_pressure(U_lo, u_sb, S_lo, p.PPR_WALL_BC, 2, p.POS_LIMITER_EPS, p.GAMMA);
                solver.compute_interface_flux(u_sb, U_lo, sig_lo, sig_lo, S_sb, S_lo, c->theta_avg, c->theta_avg, Dir, Flux_lo, Flux_S_lo);
            } else if (c->neighbors[FACE_LO<Dir>] && c->neighbors[FACE_LO<Dir>]->level == c->level) {
                Cell2D* nc = c->neighbors[FACE_LO<Dir>];
                char nface = c->neighbor_faces[FACE_LO<Dir>];
                const double* weights = (nface == NFACE_LO<Dir>) ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int v = 0; v < NV; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < N; ++k) {
                    int niy, nix;
                    dir_to_ij<Dir>(line, k, niy, nix);
                    for (int v = 0; v < NV; ++v)
                        U_neigh[v] += nc->get_U(v, niy, nix, N) * weights[k];
                    sig_neigh += nc->sigma_field[flat2d(niy, nix, N)] * weights[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += nc->S_field[flat2d(niy, nix, N)] * weights[k];
                    }
                }
                solver.compute_interface_flux(U_neigh, U_lo, sig_neigh, sig_lo, S_neigh, S_lo, nc->theta_avg, c->theta_avg, Dir, Flux_lo, Flux_S_lo);
            } else if (c->is_boundary[FACE_LO<Dir>]) {
                double S_neigh = S_lo;
                solver.get_neigh_state_cell(*c, line, false,
                                             U_lo, sig_lo, U_neigh, sig_neigh, Dir, S_lo, &S_neigh);
                solver.compute_interface_flux(U_neigh, U_lo, sig_neigh, sig_lo, S_neigh, S_lo, c->theta_avg, c->theta_avg, Dir, Flux_lo, Flux_S_lo);
            }

            // High face
            const ImmersedBoundary::SurrogateFluxPoint* sfp_hi =
                (p.ENABLE_IB && p.IB_METHOD == "SBM")
                    ? ImmersedBoundary::get_sbm_face(c->block_id, c->ey, c->ex, FACE_HI<Dir>, line)
                    : nullptr;
            if (sfp_hi) {
                double u_sb[NV];
                ImmersedBoundary::compute_sbm_state(solver, sfp_hi, u_sb);
                double S_sb = Solver::compute_wall_phantom_pressure(U_hi, u_sb, S_hi, p.PPR_WALL_BC, 2, p.POS_LIMITER_EPS, p.GAMMA);
                solver.compute_interface_flux(U_hi, u_sb, sig_hi, sig_hi, S_hi, S_sb, c->theta_avg, c->theta_avg, Dir, Flux_hi, Flux_S_hi);
            } else if (c->neighbors[FACE_HI<Dir>] && c->neighbors[FACE_HI<Dir>]->level == c->level) {
                Cell2D* nc = c->neighbors[FACE_HI<Dir>];
                char nface = c->neighbor_faces[FACE_HI<Dir>];
                const double* weights = (nface == NFACE_LO<Dir>) ? basis.l_L.data() : basis.l_R.data();
                sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int v = 0; v < NV; ++v) U_neigh[v] = 0.0;
                for (int k = 0; k < N; ++k) {
                    int niy, nix;
                    dir_to_ij<Dir>(line, k, niy, nix);
                    for (int v = 0; v < NV; ++v)
                        U_neigh[v] += nc->get_U(v, niy, nix, N) * weights[k];
                    sig_neigh += nc->sigma_field[flat2d(niy, nix, N)] * weights[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += nc->S_field[flat2d(niy, nix, N)] * weights[k];
                    }
                }
                solver.compute_interface_flux(U_hi, U_neigh, sig_hi, sig_neigh, S_hi, S_neigh, c->theta_avg, nc->theta_avg, Dir, Flux_hi, Flux_S_hi);
            } else if (c->is_boundary[FACE_HI<Dir>]) {
                double S_neigh = S_hi;
                solver.get_neigh_state_cell(*c, line, true,
                                             U_hi, sig_hi, U_neigh, sig_neigh, Dir, S_hi, &S_neigh);
                solver.compute_interface_flux(U_hi, U_neigh, sig_hi, sig_neigh, S_hi, S_neigh, c->theta_avg, c->theta_avg, Dir, Flux_hi, Flux_S_hi);
            }

            // --- 4. Interior flux at faces (for correction) ---
            double F_lo_int[NV] = {}, F_hi_int[NV] = {};
            double F_S_lo_int = 0.0, F_S_hi_int = 0.0;
            for (int v = 0; v < NV; ++v) {
                for (int sw = 0; sw < N; ++sw) {
                    F_lo_int[v] += Flux_sol[sw][v] * basis.l_L[sw];
                    F_hi_int[v] += Flux_sol[sw][v] * basis.l_R[sw];
                }
            }
            if (p.ENABLE_PPR) {
                for (int sw = 0; sw < N; ++sw) {
                    F_S_lo_int += Flux_sol_S[sw] * basis.l_L[sw];
                    F_S_hi_int += Flux_sol_S[sw] * basis.l_R[sw];
                }
            }

            // --- 5. Accumulate into RHS ---
            for (int v = 0; v < NV; ++v) {
                for (int sw = 0; sw < N; ++sw) {
                    double df = 0.0;
                    for (int k = 0; k < N; ++k)
                        df += basis.D[sw][k] * Flux_sol[k][v];

                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    c->get_RHS(v, iy, ix, N) -=
                        (df
                         + (Flux_lo[v] - F_lo_int[v]) * basis.dgl[sw]
                         + (Flux_hi[v] - F_hi_int[v]) * basis.dgr[sw])
                        * inv_h2;
                }
            }
            if (p.ENABLE_PPR) {
                for (int sw = 0; sw < N; ++sw) {
                    double df_S = 0.0;
                    for (int k = 0; k < N; ++k)
                        df_S += basis.D[sw][k] * Flux_sol_S[k];

                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    c->S_RHS[flat2d(iy, ix, N)] -=
                        (df_S
                         + (Flux_S_lo - F_S_lo_int) * basis.dgl[sw]
                         + (Flux_S_hi - F_S_hi_int) * basis.dgr[sw])
                        * inv_h2;
                }
            }
        }
    }

    // =========================================================================
    // Pass 2: Non-Conforming Interface Sweep (sparse, only near hanging nodes)
    // =========================================================================
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < solver.cells.size(); ++i) {
        Cell2D* c = solver.cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        // Process both low and high faces
        for (int side = 0; side < 2; ++side) {
            const int face_idx = (side == 0) ? FACE_LO<Dir> : FACE_HI<Dir>;
            if (!c->neighbors[face_idx] || c->neighbors[face_idx]->level >= c->level)
                continue;

            Cell2D* nc = c->neighbors[face_idx];
            char nface = c->neighbor_faces[face_idx];
            int child_idx = nc_child_idx<Dir>(*c);
            const auto& P = (child_idx == 0) ? basis.P1 : basis.P2;
            const auto& R = (child_idx == 0) ? basis.R1 : basis.R2;
            const double* dg_nc = (nface == NFACE_LO<Dir>) ? basis.dgl.data() : basis.dgr.data();
            // For the fine cell: low face uses dgl, high face uses dgr
            const double* dg_fine = (side == 0) ? basis.dgl.data() : basis.dgr.data();
            // For extrapolation to this face: low face uses l_L, high face uses l_R
            const double* extrap = (side == 0) ? basis.l_L.data() : basis.l_R.data();

            const double inv_h2_fine = 2.0 / cell_metric(*c, Dir);
            const double inv_h2_coarse = 2.0 / cell_metric(*nc, Dir);

            for (int line = 0; line < N; ++line) {
                // Extrapolate fine cell state to the face
                double U_face[NV] = {};
                double sig_face = 0.0;
                double S_face = 0.0;
                for (int sw = 0; sw < N; ++sw) {
                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    int flat = flat2d(iy, ix, N);

                    sig_face += c->sigma_field[flat] * extrap[sw];
                    if (p.ENABLE_PPR) {
                        S_face += c->S_field[flat] * extrap[sw];
                    }
                    for (int v = 0; v < NV; ++v) {
                        U_face[v] += c->get_U(v, iy, ix, N) * extrap[sw];
                    }
                }

                // Build coarse face state along the transverse line
                double U_coarse_face[NV][MAX_PTS] = {};
                double sig_coarse_face[MAX_PTS] = {};
                double S_coarse_face[MAX_PTS] = {};
                const double* nc_weights = (nface == NFACE_LO<Dir>) ? basis.l_L.data() : basis.l_R.data();
                for (int k_trans = 0; k_trans < N; ++k_trans) {
                    for (int k_sweep = 0; k_sweep < N; ++k_sweep) {
                        int niy, nix;
                        dir_to_ij<Dir>(k_trans, k_sweep, niy, nix);
                        int flat = flat2d(niy, nix, N);

                        for (int v = 0; v < NV; ++v) {
                            U_coarse_face[v][k_trans] += nc->get_U(v, niy, nix, N) * nc_weights[k_sweep];
                        }
                        sig_coarse_face[k_trans] += nc->sigma_field[flat] * nc_weights[k_sweep];
                        if (p.ENABLE_PPR) {
                            S_coarse_face[k_trans] += nc->S_field[flat] * nc_weights[k_sweep];
                        }
                    }
                }

                // Prolongate coarse face state to fine face point
                double U_neigh[NV] = {};
                double sig_neigh = 0.0;
                double S_neigh = 0.0;
                for (int k = 0; k < N; ++k) {
                    for (int v = 0; v < NV; ++v) {
                        U_neigh[v] += P[k][line] * U_coarse_face[v][k];
                    }
                    sig_neigh += P[k][line] * sig_coarse_face[k];
                    if (p.ENABLE_PPR) {
                        S_neigh += P[k][line] * S_coarse_face[k];
                    }
                }

                // Compute interface flux (direction matters for argument ordering)
                double Flux_comm[NV];
                double Flux_S_comm = 0.0;
                if (side == 0) {
                    // Low face: neighbor is on the "outside" (left/bottom)
                    solver.compute_interface_flux(U_neigh, U_face, sig_neigh, sig_face, S_neigh, S_face, nc->theta_avg, c->theta_avg, Dir, Flux_comm, Flux_S_comm);
                } else {
                    // High face: fine cell is on the "inside" (left/bottom relative to interface)
                    solver.compute_interface_flux(U_face, U_neigh, sig_face, sig_neigh, S_face, S_neigh, c->theta_avg, nc->theta_avg, Dir, Flux_comm, Flux_S_comm);
                }

                // Update fine cell RHS
                for (int sw = 0; sw < N; ++sw) {
                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    for (int v = 0; v < NV; ++v) {
                        #pragma omp atomic
                        c->get_RHS(v, iy, ix, N) -= Flux_comm[v] * dg_fine[sw] * inv_h2_fine;
                    }
                    if (p.ENABLE_PPR) {
                        #pragma omp atomic
                        c->S_RHS[flat2d(iy, ix, N)] -= Flux_S_comm * dg_fine[sw] * inv_h2_fine;
                    }
                }

                // Restrict and accumulate to coarse neighbor
                for (int k_trans = 0; k_trans < N; ++k_trans) {
                    double factor = R[line][k_trans];
                    for (int k_sweep = 0; k_sweep < N; ++k_sweep) {
                        int niy, nix;
                        dir_to_ij<Dir>(k_trans, k_sweep, niy, nix);
                        for (int v = 0; v < NV; ++v) {
                            #pragma omp atomic
                            nc->get_RHS(v, niy, nix, N) -= factor * Flux_comm[v] * dg_nc[k_sweep] * inv_h2_coarse;
                        }
                        if (p.ENABLE_PPR) {
                            #pragma omp atomic
                            nc->S_RHS[flat2d(niy, nix, N)] -= factor * Flux_S_comm * dg_nc[k_sweep] * inv_h2_coarse;
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
// Unified 2D Viscous Sweep (BR2 Phase 2)
// =========================================================================

/**
 * @brief Pointwise 2D Navier-Stokes viscous flux calculation.
 */
template<int Dir>
inline void compute_viscous_flux_2d(const double U[4],
                                     const double dUdx[4], const double dUdy[4],
                                     double mu, double kappa, double gamma,
                                     double Flux_v[4], bool enable_suth, double suth_c, double pr)
{
    double rho = std::max(1e-14, U[0]);
    double inv_rho = 1.0 / rho;
    double u = U[1] * inv_rho;
    double v = U[2] * inv_rho;

    double dudx = (dUdx[1] - u * dUdx[0]) * inv_rho;
    double dudy = (dUdy[1] - u * dUdy[0]) * inv_rho;
    double dvdx = (dUdx[2] - v * dUdx[0]) * inv_rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) * inv_rho;

    double p_val = (gamma - 1.0) * (U[3] - 0.5 * rho * (u*u + v*v));
    if (p_val < 1e-14) p_val = 1e-14;
    double T = p_val * inv_rho;

    double mu_local = mu;
    double kappa_local = kappa;
    if (enable_suth) {
        double T_norm = std::max(1e-8, T);
        mu_local = mu * (std::pow(T_norm, 1.5) * (1.0 + suth_c) / (T_norm + suth_c));
        kappa_local = mu_local * gamma / ((gamma - 1.0) * pr);
    }

    if constexpr (Dir == 0) {
        double dpdx = (gamma - 1.0) * (dUdx[3] - 0.5*(u*u+v*v)*dUdx[0] - rho*(u*dudx + v*dvdx));
        double dTdx = (dpdx - T * dUdx[0]) * inv_rho;
        double tau_xx = mu_local * (4.0/3.0 * dudx - 2.0/3.0 * dvdy);
        double tau_xy = mu_local * (dudy + dvdx);
        double qx = -kappa_local * dTdx;

        Flux_v[0] = 0.0;
        Flux_v[1] = tau_xx;
        Flux_v[2] = tau_xy;
        Flux_v[3] = u * tau_xx + v * tau_xy - qx;
    } else {
        double dpdy = (gamma - 1.0) * (dUdy[3] - 0.5*(u*u+v*v)*dUdy[0] - rho*(u*dudy + v*dvdy));
        double dTdy = (dpdy - T * dUdy[0]) * inv_rho;
        double tau_yy = mu_local * (4.0/3.0 * dvdy - 2.0/3.0 * dudx);
        double tau_xy = mu_local * (dudy + dvdx);
        double qy = -kappa_local * dTdy;

        Flux_v[0] = 0.0;
        Flux_v[1] = tau_xy;
        Flux_v[2] = tau_yy;
        Flux_v[3] = u * tau_xy + v * tau_yy - qy;
    }
}

/**
 * @brief Generic 2D viscous FR sweep in direction Dir (BR2 Phase 2).
 */
template<int Dir>
void viscous_sweep_2d(Solver& solver) {
    const Parameters& p = solver.p;
    const Basis& basis  = solver.basis;
    const int N = p.N_PTS;
    constexpr int NV = 4;

    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < solver.cells.size(); ++i) {
        Cell2D* c = solver.cells[i];
        if (p.ENABLE_MULTIRATE && !c->element_active) continue;

        auto get_suth_mu = [&](const double State[4]) {
            double r = std::max(1e-14, State[0]);
            double press = (p.GAMMA - 1.0) * (State[3] - 0.5 * (State[1]*State[1] + State[2]*State[2]) / r);
            if (press < 1e-14) press = 1e-14;
            double Temp = press / r;
            double T_norm = std::max(1e-8, Temp);
            return mu * (std::pow(T_norm, 1.5) * (1.0 + p.SUTH_C) / (T_norm + p.SUTH_C));
        };

        auto extrapolate_neighbor = [&](const Cell2D* nc, char nface, int line, double U_nb[4], double dUdx_nb[4], double dUdy_nb[4]) {
            const double* weights = (nface == NFACE_LO<Dir>) ? basis.l_L.data() : basis.l_R.data();
            for (int v = 0; v < NV; ++v) {
                U_nb[v] = 0.0;
                dUdx_nb[v] = 0.0;
                dUdy_nb[v] = 0.0;
            }
            int nc_offset = nc->cell_index * NV * N * N;
            for (int k = 0; k < N; ++k) {
                int niy, nix;
                dir_to_ij<Dir>(line, k, niy, nix);
                int flat = flat2d(niy, nix, N);
                for (int v = 0; v < NV; ++v) {
                    U_nb[v] += nc->get_U(v, niy, nix, N) * weights[k];
                    dUdx_nb[v] += solver.global_grad_Ux[nc_offset + v * N * N + flat] * weights[k];
                    dUdy_nb[v] += solver.global_grad_Uy[nc_offset + v * N * N + flat] * weights[k];
                }
            }
        };

        int c_offset = c->cell_index * NV * N * N;
        const double h_metric = cell_metric(*c, Dir);
        const double inv_h2 = 2.0 / h_metric;

        for (int line = 0; line < N; ++line) {
            double Flux_v_sol[MAX_PTS][NV];
            for (int sw = 0; sw < N; ++sw) {
                int iy, ix;
                dir_to_ij<Dir>(line, sw, iy, ix);
                int flat = flat2d(iy, ix, N);
                double U_pt[NV], dUdx_pt[NV], dUdy_pt[NV];
                for (int v = 0; v < NV; ++v) {
                    U_pt[v]    = c->get_U(v, iy, ix, N);
                    dUdx_pt[v] = solver.global_grad_Ux[c_offset + v * N * N + flat];
                    dUdy_pt[v] = solver.global_grad_Uy[c_offset + v * N * N + flat];
                }
                compute_viscous_flux_2d<Dir>(U_pt, dUdx_pt, dUdy_pt,
                                             mu, kappa, p.GAMMA, Flux_v_sol[sw],
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
            }

            double Flux_v_lo_int[NV] = {}, Flux_v_hi_int[NV] = {};
            double U_lo_face[NV] = {}, U_hi_face[NV] = {};
            for (int v = 0; v < NV; ++v) {
                for (int sw = 0; sw < N; ++sw) {
                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    Flux_v_lo_int[v] += Flux_v_sol[sw][v] * basis.l_L[sw];
                    Flux_v_hi_int[v] += Flux_v_sol[sw][v] * basis.l_R[sw];
                    U_lo_face[v] += c->get_U(v, iy, ix, N) * basis.l_L[sw];
                    U_hi_face[v] += c->get_U(v, iy, ix, N) * basis.l_R[sw];
                }
            }

            double Flux_v_lo_comm[NV] = {};
            if (c->neighbors[FACE_LO<Dir>] && c->neighbors[FACE_LO<Dir>]->level == c->level) {
                Cell2D* nc = c->neighbors[FACE_LO<Dir>];
                char nface = c->neighbor_faces[FACE_LO<Dir>];
                double U_nb[NV], dUdx_nb[NV], dUdy_nb[NV], Flux_v_nb[NV];
                extrapolate_neighbor(nc, nface, line, U_nb, dUdx_nb, dUdy_nb);
                compute_viscous_flux_2d<Dir>(U_nb, dUdx_nb, dUdy_nb,
                                             mu, kappa, p.GAMMA, Flux_v_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_face = 0.5 * (get_suth_mu(U_lo_face) + get_suth_mu(U_nb));
                }
                double penalty = br2_eta * mu_face / h_metric;
                for (int v = 0; v < NV; ++v) {
                    Flux_v_lo_comm[v] = 0.5 * (Flux_v_nb[v] + Flux_v_lo_int[v])
                                      + penalty * (U_lo_face[v] - U_nb[v]);
                }
            } else if (c->is_boundary[FACE_LO<Dir>]) {
                for (int v = 0; v < NV; ++v) {
                    Flux_v_lo_comm[v] = Flux_v_lo_int[v];
                }
            }

            double Flux_v_hi_comm[NV] = {};
            if (c->neighbors[FACE_HI<Dir>] && c->neighbors[FACE_HI<Dir>]->level == c->level) {
                Cell2D* nc = c->neighbors[FACE_HI<Dir>];
                char nface = c->neighbor_faces[FACE_HI<Dir>];
                double U_nb[NV], dUdx_nb[NV], dUdy_nb[NV], Flux_v_nb[NV];
                extrapolate_neighbor(nc, nface, line, U_nb, dUdx_nb, dUdy_nb);
                compute_viscous_flux_2d<Dir>(U_nb, dUdx_nb, dUdy_nb,
                                             mu, kappa, p.GAMMA, Flux_v_nb,
                                             p.ENABLE_SUTHERLAND, p.SUTH_C, p.PR);
                double mu_face = mu;
                if (p.ENABLE_SUTHERLAND) {
                    mu_face = 0.5 * (get_suth_mu(U_hi_face) + get_suth_mu(U_nb));
                }
                double penalty = br2_eta * mu_face / h_metric;
                for (int v = 0; v < NV; ++v) {
                    Flux_v_hi_comm[v] = 0.5 * (Flux_v_nb[v] + Flux_v_hi_int[v])
                                      + penalty * (U_hi_face[v] - U_nb[v]);
                }
            } else if (c->is_boundary[FACE_HI<Dir>]) {
                for (int v = 0; v < NV; ++v) {
                    Flux_v_hi_comm[v] = Flux_v_hi_int[v];
                }
            }

            for (int v = 0; v < NV; ++v) {
                for (int sw = 0; sw < N; ++sw) {
                    double df_v = 0.0;
                    for (int k = 0; k < N; ++k)
                        df_v += basis.D[sw][k] * Flux_v_sol[k][v];

                    int iy, ix;
                    dir_to_ij<Dir>(line, sw, iy, ix);
                    c->get_RHS(v, iy, ix, N) +=
                        (df_v
                         + (Flux_v_lo_comm[v] - Flux_v_lo_int[v]) * basis.dgl[sw]
                         + (Flux_v_hi_comm[v] - Flux_v_hi_int[v]) * basis.dgr[sw])
                        * inv_h2;
                }
            }
        }
    }
}

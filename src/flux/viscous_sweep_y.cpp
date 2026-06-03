/// @file viscous_sweep_y.cpp
/// @brief Y-direction viscous flux divergence (BR2 Phase 2).
///
/// Mirrors viscous_sweep_x but operates along columns.  Computes the
/// Y-component of the viscous flux divergence using pre-computed gradients.
///
/// OpenMP: parallelised over ex.

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/// Compute the Y-component of the viscous flux at a single solution point.
static void compute_viscous_flux_y(const double U[4],
                                    const double dUdx[4], const double dUdy[4],
                                    double mu, double kappa, double gamma,
                                    double Gv[4])
{
    double rho = std::max(1e-14, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;

    // Velocity gradients from conservative variable gradients
    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;

    // Temperature: T = p/ρ
    double p_val = (gamma - 1.0) * (U[3] - 0.5 * rho * (u*u + v*v));
    if (p_val < 1e-14) p_val = 1e-14;
    double T = p_val / rho;

    // Temperature gradient (Y-component)
    double dpdy = (gamma - 1.0) * (dUdy[3] - 0.5*(u*u+v*v)*dUdy[0]
                                    - rho*(u*dudy + v*dvdy));
    double dTdy = (dpdy - T * dUdy[0]) / rho;

    // Stress tensor components
    double tau_yy = mu * (4.0/3.0 * dvdy - 2.0/3.0 * dudx);
    double tau_xy = mu * (dudy + dvdx);

    // Heat flux
    double qy = -kappa * dTdy;

    // Viscous flux vector Gv = [0, τ_xy, τ_yy, u·τ_xy + v·τ_yy − q_y]
    Gv[0] = 0.0;
    Gv[1] = tau_xy;
    Gv[2] = tau_yy;
    Gv[3] = u * tau_xy + v * tau_yy - qy;
}

void Solver::viscous_sweep_y() {
    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);

    for (auto& b : blocks) {
        const int n_dofs = b.nx * b.ny * p.N_PTS * p.N_PTS;

        #pragma omp for schedule(static)
        for (int ex = 0; ex < b.nx; ++ex) {
            double prev_Gv_T_comm[MAX_PTS][4];
            for (int ey = 0; ey < b.ny; ++ey) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {

                    // --- 1. Pointwise viscous Y-flux at each solution point ---
                    double Gv_sol[MAX_PTS][4];
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        int flat = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                        double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                        for (int v = 0; v < 4; ++v) {
                            U_pt[v]    = b.U(v, ey, ex, iy, ix);
                            dUdx_pt[v] = b.grad_Ux[v * n_dofs + flat];
                            dUdy_pt[v] = b.grad_Uy[v * n_dofs + flat];
                        }
                        compute_viscous_flux_y(U_pt, dUdx_pt, dUdy_pt,
                                                mu, kappa, p.GAMMA, Gv_sol[iy]);
                    }

                    // --- 2. Face-extrapolated viscous fluxes & states ---
                    double Gv_B[4] = {}, Gv_T[4] = {};
                    double UB_face[4] = {}, UT_face[4] = {};
                    for (int v = 0; v < 4; ++v) {
                        for (int iy = 0; iy < p.N_PTS; ++iy) {
                            Gv_B[v] += Gv_sol[iy][v] * basis.l_L[iy];
                            Gv_T[v] += Gv_sol[iy][v] * basis.l_R[iy];
                            UB_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_L[iy];
                            UT_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_R[iy];
                        }
                    }

                    // --- 3. Neighbour viscous fluxes ---
                    // Bottom face
                    double Gv_B_nb[4] = {};
                    double UB_nb[4];
                    bool bnd_B = (ey == 0 && b.ni_b.id == -1);
                    double Gv_B_comm[4], Gv_T_comm[4];
                    double penalty = br2_eta * mu / b.dy;

                    if (ey > 0) {
                        for (int v = 0; v < 4; ++v)
                            Gv_B_comm[v] = prev_Gv_T_comm[ix][v];
                    } else {
                        {
                            double sig_dummy;
                            get_neigh_state_y(b, ey, ex, ix, false,
                                              UB_face, 0.0, UB_nb, sig_dummy);
                            if (ey > 0 || b.ni_b.id != -1) {
                                int ney;
                                const Block* nb_ptr;
                                const double* weights;
                                if (ey > 0) {
                                    nb_ptr = &b;
                                    ney = ey - 1;
                                    weights = basis.l_R.data();
                                } else {
                                    nb_ptr = &blocks[b.ni_b.id];
                                    ney = (b.ni_b.face == 'B') ? 0 : nb_ptr->ny - 1;
                                    weights = (b.ni_b.face == 'B') ? basis.l_L.data() : basis.l_R.data();
                                }
                                const Block& nb = *nb_ptr;
                                int nb_n_dofs = nb.nx * nb.ny * p.N_PTS * p.N_PTS;
                                for (int kk = 0; kk < p.N_PTS; ++kk) {
                                    int nflat = nb.get_flat_idx(ney, ex, kk, ix, p.N_PTS);
                                    double U_nb[4], dUdx_nb[4], dUdy_nb[4];
                                    for (int v = 0; v < 4; ++v) {
                                        U_nb[v]    = nb.U(v, ney, ex, kk, ix);
                                        dUdx_nb[v] = nb.grad_Ux[v * nb_n_dofs + nflat];
                                        dUdy_nb[v] = nb.grad_Uy[v * nb_n_dofs + nflat];
                                    }
                                    double Gv_kk[4];
                                    compute_viscous_flux_y(U_nb, dUdx_nb, dUdy_nb,
                                                            mu, kappa, p.GAMMA, Gv_kk);
                                    for (int v = 0; v < 4; ++v)
                                        Gv_B_nb[v] += Gv_kk[v] * weights[kk];
                                }
                            } else {
                                for (int v = 0; v < 4; ++v)
                                    Gv_B_nb[v] = Gv_B[v];
                            }
                        }
                        for (int v = 0; v < 4; ++v) {
                            if (bnd_B) {
                                Gv_B_comm[v] = Gv_B[v];
                            } else {
                                Gv_B_comm[v] = 0.5 * (Gv_B_nb[v] + Gv_B[v])
                                             + penalty * (UB_face[v] - UB_nb[v]);
                            }
                        }
                    }

                    // Top face
                    double Gv_T_nb[4] = {};
                    double UT_nb[4];
                    {
                        double sig_dummy;
                        get_neigh_state_y(b, ey, ex, ix, true,
                                          UT_face, 0.0, UT_nb, sig_dummy);
                        if (ey < b.ny - 1 || b.ni_t.id != -1) {
                            int ney;
                            const Block* nb_ptr;
                            const double* weights;
                            if (ey < b.ny - 1) {
                                nb_ptr = &b;
                                ney = ey + 1;
                                weights = basis.l_L.data();
                            } else {
                                nb_ptr = &blocks[b.ni_t.id];
                                ney = (b.ni_t.face == 'B') ? 0 : nb_ptr->ny - 1;
                                weights = (b.ni_t.face == 'B') ? basis.l_L.data() : basis.l_R.data();
                            }
                            const Block& nb = *nb_ptr;
                            int nb_n_dofs = nb.nx * nb.ny * p.N_PTS * p.N_PTS;
                            for (int kk = 0; kk < p.N_PTS; ++kk) {
                                int nflat = nb.get_flat_idx(ney, ex, kk, ix, p.N_PTS);
                                double U_nb[4], dUdx_nb[4], dUdy_nb[4];
                                for (int v = 0; v < 4; ++v) {
                                    U_nb[v]    = nb.U(v, ney, ex, kk, ix);
                                    dUdx_nb[v] = nb.grad_Ux[v * nb_n_dofs + nflat];
                                    dUdy_nb[v] = nb.grad_Uy[v * nb_n_dofs + nflat];
                                }
                                double Gv_kk[4];
                                compute_viscous_flux_y(U_nb, dUdx_nb, dUdy_nb,
                                                        mu, kappa, p.GAMMA, Gv_kk);
                                for (int v = 0; v < 4; ++v)
                                    Gv_T_nb[v] += Gv_kk[v] * weights[kk];
                            }
                        } else {
                            for (int v = 0; v < 4; ++v)
                                Gv_T_nb[v] = Gv_T[v];
                        }
                    }

                    bool bnd_T = (ey == b.ny - 1 && b.ni_t.id == -1);

                    for (int v = 0; v < 4; ++v) {
                        if (bnd_T) {
                            Gv_T_comm[v] = Gv_T[v];
                        } else {
                            Gv_T_comm[v] = 0.5 * (Gv_T[v] + Gv_T_nb[v])
                                         + penalty * (UT_nb[v] - UT_face[v]);
                        }
                    }

                    for (int v = 0; v < 4; ++v)
                        prev_Gv_T_comm[ix][v] = Gv_T_comm[v];

                    // --- 5. Accumulate viscous contribution into RHS ---
                    for (int v = 0; v < 4; ++v) {
                        for (int iy = 0; iy < p.N_PTS; ++iy) {
                            double dg = 0.0;
                            for (int kk = 0; kk < p.N_PTS; ++kk)
                                dg += basis.D[iy][kk] * Gv_sol[kk][v];
                            b.RHS(v, ey, ex, iy, ix) +=
                                (dg
                                 + (Gv_B_comm[v] - Gv_B[v]) * basis.dgl[iy]
                                 + (Gv_T_comm[v] - Gv_T[v]) * basis.dgr[iy])
                                * (2.0 / b.dy);
                        }
                    }
                }
            }
        }
    }
}

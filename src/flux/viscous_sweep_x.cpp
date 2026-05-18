/// @file viscous_sweep_x.cpp
/// @brief X-direction viscous flux divergence (BR2 Phase 2).
///
/// Computes the X-component of the viscous flux divergence at every solution
/// point using the pre-computed gradients (grad_Ux, grad_Uy).  The viscous
/// contribution is ADDED to RHS (since viscous fluxes appear on the RHS of
/// the momentum/energy equations with positive sign for dissipation).
///
/// OpenMP: parallelised over ey.

#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/// Compute the X-component of the viscous flux at a single solution point.
///
/// @param U     Conservative state [ρ, ρu, ρv, E].
/// @param dUdx  dU/dx for all 4 conservative variables.
/// @param dUdy  dU/dy for all 4 conservative variables.
/// @param mu    Dynamic viscosity (1/Re).
/// @param kappa Thermal conductivity.
/// @param gamma Ratio of specific heats.
/// @param Fv    Output: X-direction viscous flux [4 components].
static void compute_viscous_flux_x(const double U[4],
                                    const double dUdx[4], const double dUdy[4],
                                    double mu, double kappa, double gamma,
                                    double Fv[4])
{
    double rho = std::max(1e-14, U[0]);
    double u = U[1] / rho;
    double v = U[2] / rho;

    // Velocity gradients from conservative variable gradients:
    //   du/dx = (d(ρu)/dx − u·dρ/dx) / ρ
    double dudx = (dUdx[1] - u * dUdx[0]) / rho;
    double dudy = (dUdy[1] - u * dUdy[0]) / rho;
    double dvdx = (dUdx[2] - v * dUdx[0]) / rho;
    double dvdy = (dUdy[2] - v * dUdy[0]) / rho;

    // Temperature: T = p/ρ, where p = (γ−1)(E − 0.5ρ(u²+v²))
    double p = (gamma - 1.0) * (U[3] - 0.5 * rho * (u*u + v*v));
    if (p < 1e-14) p = 1e-14;
    double T = p / rho;

    // Temperature gradient: dT/dx = (dp/dx − T·dρ/dx) / ρ
    double dpdx = (gamma - 1.0) * (dUdx[3] - 0.5*(u*u+v*v)*dUdx[0]
                                    - rho*(u*dudx + v*dvdx));
    double dTdx = (dpdx - T * dUdx[0]) / rho;

    // Stress tensor components
    double tau_xx = mu * (4.0/3.0 * dudx - 2.0/3.0 * dvdy);
    double tau_xy = mu * (dudy + dvdx);

    // Heat flux
    double qx = -kappa * dTdx;

    // Viscous flux vector Fv = [0, τ_xx, τ_xy, u·τ_xx + v·τ_xy − q_x]
    Fv[0] = 0.0;
    Fv[1] = tau_xx;
    Fv[2] = tau_xy;
    Fv[3] = u * tau_xx + v * tau_xy - qx;
}

void Solver::viscous_sweep_x() {
    const double mu    = 1.0 / p.RE;
    const double kappa = mu * p.GAMMA / ((p.GAMMA - 1.0) * p.PR);
    const double br2_eta = p.NS_BR2_ETA * (p.P_DEG + 1) * (p.P_DEG + 1);

    for (auto& b : blocks) {
        const int n_dofs = b.nx * b.ny * p.N_PTS * p.N_PTS;

        #pragma omp parallel for schedule(static)
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ex = 0; ex < b.nx; ++ex) {

                    // --- 1. Pointwise viscous X-flux at each solution point ---
                    double Fv_sol[MAX_PTS][4];
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        int flat = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                        double U_pt[4], dUdx_pt[4], dUdy_pt[4];
                        for (int v = 0; v < 4; ++v) {
                            U_pt[v]    = b.U(v, ey, ex, iy, ix);
                            dUdx_pt[v] = b.grad_Ux[v * n_dofs + flat];
                            dUdy_pt[v] = b.grad_Uy[v * n_dofs + flat];
                        }
                        compute_viscous_flux_x(U_pt, dUdx_pt, dUdy_pt,
                                                mu, kappa, p.GAMMA, Fv_sol[ix]);
                    }

                    // --- 2. Face-extrapolated viscous fluxes & states ---
                    double Fv_L[4] = {}, Fv_R[4] = {};
                    double UL_face[4] = {}, UR_face[4] = {};
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        for (int v = 0; v < 4; ++v) {
                            Fv_L[v] += Fv_sol[ix][v] * basis.l_L[ix];
                            Fv_R[v] += Fv_sol[ix][v] * basis.l_R[ix];
                            UL_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_L[ix];
                            UR_face[v] += b.U(v, ey, ex, iy, ix) * basis.l_R[ix];
                        }
                    }

                    // --- 3. Neighbour viscous fluxes ---
                    // Left face
                    double Fv_L_nb[4] = {};
                    double UL_nb[4];
                    {
                        double sig_dummy;
                        get_neigh_state_x(b, ey, ex, iy, false,
                                          UL_face, 0.0, UL_nb, sig_dummy);
                        if (ex > 0 || b.ni_l.id != -1) {
                            int nex;
                            const Block* nb_ptr;
                            const double* weights;
                            if (ex > 0) {
                                nb_ptr = &b;
                                nex = ex - 1;
                                weights = basis.l_R.data();
                            } else {
                                nb_ptr = &blocks[b.ni_l.id];
                                nex = (b.ni_l.face == 'L') ? 0 : nb_ptr->nx - 1;
                                weights = (b.ni_l.face == 'L') ? basis.l_L.data() : basis.l_R.data();
                            }
                            const Block& nb = *nb_ptr;
                            int nb_n_dofs = nb.nx * nb.ny * p.N_PTS * p.N_PTS;
                            for (int kk = 0; kk < p.N_PTS; ++kk) {
                                int nflat = nb.get_flat_idx(ey, nex, iy, kk, p.N_PTS);
                                double U_nb[4], dUdx_nb[4], dUdy_nb[4];
                                for (int v = 0; v < 4; ++v) {
                                    U_nb[v]    = nb.U(v, ey, nex, iy, kk);
                                    dUdx_nb[v] = nb.grad_Ux[v * nb_n_dofs + nflat];
                                    dUdy_nb[v] = nb.grad_Uy[v * nb_n_dofs + nflat];
                                }
                                double Fv_kk[4];
                                compute_viscous_flux_x(U_nb, dUdx_nb, dUdy_nb,
                                                        mu, kappa, p.GAMMA, Fv_kk);
                                for (int v = 0; v < 4; ++v)
                                    Fv_L_nb[v] += Fv_kk[v] * weights[kk];
                            }
                        } else {
                            for (int v = 0; v < 4; ++v)
                                Fv_L_nb[v] = Fv_L[v];
                        }
                    }

                    // Right face
                    double Fv_R_nb[4] = {};
                    double UR_nb[4];
                    {
                        double sig_dummy;
                        get_neigh_state_x(b, ey, ex, iy, true,
                                          UR_face, 0.0, UR_nb, sig_dummy);
                        if (ex < b.nx - 1 || b.ni_r.id != -1) {
                            int nex;
                            const Block* nb_ptr;
                            const double* weights;
                            if (ex < b.nx - 1) {
                                nb_ptr = &b;
                                nex = ex + 1;
                                weights = basis.l_L.data();
                            } else {
                                nb_ptr = &blocks[b.ni_r.id];
                                nex = (b.ni_r.face == 'L') ? 0 : nb_ptr->nx - 1;
                                weights = (b.ni_r.face == 'L') ? basis.l_L.data() : basis.l_R.data();
                            }
                            const Block& nb = *nb_ptr;
                            int nb_n_dofs = nb.nx * nb.ny * p.N_PTS * p.N_PTS;
                            for (int kk = 0; kk < p.N_PTS; ++kk) {
                                int nflat = nb.get_flat_idx(ey, nex, iy, kk, p.N_PTS);
                                double U_nb[4], dUdx_nb[4], dUdy_nb[4];
                                for (int v = 0; v < 4; ++v) {
                                    U_nb[v]    = nb.U(v, ey, nex, iy, kk);
                                    dUdx_nb[v] = nb.grad_Ux[v * nb_n_dofs + nflat];
                                    dUdy_nb[v] = nb.grad_Uy[v * nb_n_dofs + nflat];
                                }
                                double Fv_kk[4];
                                compute_viscous_flux_x(U_nb, dUdx_nb, dUdy_nb,
                                                        mu, kappa, p.GAMMA, Fv_kk);
                                for (int v = 0; v < 4; ++v)
                                    Fv_R_nb[v] += Fv_kk[v] * weights[kk];
                            }
                        } else {
                            for (int v = 0; v < 4; ++v)
                                Fv_R_nb[v] = Fv_R[v];
                        }
                    }

                    // --- 4. Common interface viscous flux with BR2 penalty ---
                    // At domain boundaries, no BR2 penalty: wall BC is enforced
                    // through the gradient (Phase 1). At interior faces, use
                    // the standard BR2 penalty for stability.
                    double Fv_L_comm[4], Fv_R_comm[4];
                    double penalty = br2_eta * mu / b.dx;

                    bool bnd_L = (ex == 0 && b.ni_l.id == -1);
                    bool bnd_R = (ex == b.nx - 1 && b.ni_r.id == -1);

                    for (int v = 0; v < 4; ++v) {
                        if (bnd_L) {
                            Fv_L_comm[v] = Fv_L[v];
                        } else {
                            Fv_L_comm[v] = 0.5 * (Fv_L_nb[v] + Fv_L[v])
                                         + penalty * (UL_face[v] - UL_nb[v]);
                        }
                        if (bnd_R) {
                            Fv_R_comm[v] = Fv_R[v];
                        } else {
                            Fv_R_comm[v] = 0.5 * (Fv_R[v] + Fv_R_nb[v])
                                         + penalty * (UR_nb[v] - UR_face[v]);
                        }
                    }

                    // --- 5. Accumulate viscous contribution into RHS ---
                    //   RHS += div(Fv) · (2/dx)
                    for (int ix = 0; ix < p.N_PTS; ++ix)
                        for (int v = 0; v < 4; ++v) {
                            double df = 0.0;
                            for (int kk = 0; kk < p.N_PTS; ++kk)
                                df += basis.D[ix][kk] * Fv_sol[kk][v];
                            b.RHS(v, ey, ex, iy, ix) +=
                                (df
                                 + (Fv_L_comm[v] - Fv_L[v]) * basis.dgl[ix]
                                 + (Fv_R_comm[v] - Fv_R[v]) * basis.dgr[ix])
                                * (2.0 / b.dx);
                        }
                }
            }
        }
    }
}

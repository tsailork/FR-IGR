/**
 * @file esdirk34.cpp
 * @brief 4-Stage 3rd-Order L-Stable ESDIRK34 Implicit Time Integrator & JFNK GMRES Engine.
 */

#include "esdirk34.hpp"
#include "../limiters/limiter_smooth.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Implicit {

constexpr double ESDIRK34Tableau::gamma;
constexpr double ESDIRK34Tableau::c[4];
constexpr double ESDIRK34Tableau::A[4][4];
constexpr double ESDIRK34Tableau::b[4];

// Helper: Vector dot product
static double vec_dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Helper: Vector L2 norm
static double vec_norm(const std::vector<double>& v) {
    return std::sqrt(vec_dot(v, v));
}

// Helper: Vector RMS norm per DOF
static double vec_rms_norm(const std::vector<double>& v) {
    return (v.empty()) ? 0.0 : std::sqrt(vec_dot(v, v) / static_cast<double>(v.size()));
}

// Helper: Givens rotation
static void apply_givens(double& x, double& y, double c, double s) {
    double temp = c * x + s * y;
    y = -s * x + c * y;
    x = temp;
}

static void generate_givens(double dx, double dy, double& c, double& s) {
    if (dy == 0.0) {
        c = 1.0;
        s = 0.0;
    } else if (std::abs(dy) > std::abs(dx)) {
        double temp = dx / dy;
        s = 1.0 / std::sqrt(1.0 + temp * temp);
        c = temp * s;
    } else {
        double temp = dy / dx;
        c = 1.0 / std::sqrt(1.0 + temp * temp);
        s = temp * c;
    }
}

template<typename PrecondType>
bool gmres_solve(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& matvec,
    const std::vector<double>& b,
    std::vector<double>& x,
    const PrecondType& M_op,
    double rtol,
    double atol,
    int max_iters,
    int restart,
    int* gmres_iters_out
) {
    size_t n = b.size();
    x.assign(n, 0.0);

    double b_norm = vec_norm(b);
    if (b_norm == 0.0) return true;

    double tol = std::max(atol, rtol * b_norm);

    std::vector<double> r(b);
    double r_norm = vec_norm(r);
    if (r_norm < tol) return true;

    int m = std::min(restart, max_iters);

    // Krylov subspace basis V (size (m+1) x n)
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(n, 0.0));
    // Hessenberg matrix H (size (m+1) x m)
    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));

    std::vector<double> cs(m, 0.0);
    std::vector<double> sn(m, 0.0);
    std::vector<double> e1(m + 1, 0.0);

    for (size_t i = 0; i < n; ++i) V[0][i] = r[i] / r_norm;
    e1[0] = r_norm;

    std::vector<double> w_prec(n, 0.0);
    std::vector<double> w_matvec(n, 0.0);

    int iter = 0;
    for (; iter < m; ++iter) {
        // Right preconditioning: w = J * (M_op^{-1} * V[iter])
        M_op.apply(V[iter], w_prec);
        matvec(w_prec, w_matvec);

        // Gram-Schmidt orthogonalization
        for (int i = 0; i <= iter; ++i) {
            H[i][iter] = vec_dot(w_matvec, V[i]);
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n; ++k) {
                w_matvec[k] -= H[i][iter] * V[i][k];
            }
        }

        H[iter + 1][iter] = vec_norm(w_matvec);

        if (H[iter + 1][iter] > 1e-14) {
            double inv_h = 1.0 / H[iter + 1][iter];
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n; ++k) {
                V[iter + 1][k] = w_matvec[k] * inv_h;
            }
        }

        // Apply previous Givens rotations to new column of H
        for (int i = 0; i < iter; ++i) {
            apply_givens(H[i][iter], H[i + 1][iter], cs[i], sn[i]);
        }

        // Generate and apply new Givens rotation
        generate_givens(H[iter][iter], H[iter + 1][iter], cs[iter], sn[iter]);
        apply_givens(H[iter][iter], H[iter + 1][iter], cs[iter], sn[iter]);
        apply_givens(e1[iter], e1[iter + 1], cs[iter], sn[iter]);

        if (std::abs(e1[iter + 1]) < tol) {
            iter++;
            break;
        }
    }

    if (gmres_iters_out) *gmres_iters_out += iter;

    int k = iter;
    if (k > m) k = m;

    // Upper triangular solve H * y = e1
    std::vector<double> y(k, 0.0);
    for (int i = k - 1; i >= 0; --i) {
        y[i] = e1[i];
        for (int j = i + 1; j < k; ++j) {
            y[i] -= H[i][j] * y[j];
        }
        if (H[i][i] != 0.0) y[i] /= H[i][i];
    }

    // Compute y_flat = sum_i y[i] * V[i]
    std::vector<double> y_flat(n, 0.0);
    for (int i = 0; i < k; ++i) {
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; ++j) {
            y_flat[j] += y[i] * V[i][j];
        }
    }

    // Un-precondition: x = M_op^{-1} * y_flat
    M_op.apply(y_flat, x);
    return true;
}

// Check realizability: rho > 0 and P > 0 across all interior nodes and boundary face extrapolations
static bool check_realizability_2d(const std::vector<CellDim<2>*>& cells, const Basis& basis, double gamma) {
    bool valid = true;

    #pragma omp parallel for reduction(&&:valid) schedule(static)
    for (size_t c_idx = 0; c_idx < cells.size(); ++c_idx) {
        if (!valid) continue;
        const CellDim<2>* c = cells[c_idx];
        int dofs = c->U.size() / 4;
        for (int idx = 0; idx < dofs; ++idx) {
            double rho = c->U[0 * dofs + idx];
            double rhou = c->U[1 * dofs + idx];
            double rhov = c->U[2 * dofs + idx];
            double E = c->U[3 * dofs + idx];

            if (rho <= 0.0) {
                valid = false;
                break;
            }

            double u = rhou / rho;
            double v = rhov / rho;
            double P = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v));
            if (P <= 0.0) {
                valid = false;
                break;
            }
        }

        if (!valid) continue;

        int npts = static_cast<int>(basis.l_L.size());
        // Check Left and Right face extrapolations
        for (int iy = 0; iy < npts; ++iy) {
            double rho_L = 0.0, rhou_L = 0.0, rhov_L = 0.0, E_L = 0.0;
            double rho_R = 0.0, rhou_R = 0.0, rhov_R = 0.0, E_R = 0.0;
            for (int ix = 0; ix < npts; ++ix) {
                int node = iy * npts + ix;
                double lL = basis.l_L[ix];
                double lR = basis.l_R[ix];
                rho_L  += c->U[0 * npts * npts + node] * lL;
                rhou_L += c->U[1 * npts * npts + node] * lL;
                rhov_L += c->U[2 * npts * npts + node] * lL;
                E_L    += c->U[3 * npts * npts + node] * lL;

                rho_R  += c->U[0 * npts * npts + node] * lR;
                rhou_R += c->U[1 * npts * npts + node] * lR;
                rhov_R += c->U[2 * npts * npts + node] * lR;
                E_R    += c->U[3 * npts * npts + node] * lR;
            }
            if (rho_L <= 0.0 || rho_R <= 0.0) { valid = false; break; }
            double P_L = (gamma - 1.0) * (E_L - 0.5 * (rhou_L * rhou_L + rhov_L * rhov_L) / rho_L);
            double P_R = (gamma - 1.0) * (E_R - 0.5 * (rhou_R * rhou_R + rhov_R * rhov_R) / rho_R);
            if (P_L <= 0.0 || P_R <= 0.0) { valid = false; break; }
        }

        if (!valid) continue;

        // Check Bottom and Top face extrapolations
        for (int ix = 0; ix < npts; ++ix) {
            double rho_B = 0.0, rhou_B = 0.0, rhov_B = 0.0, E_B = 0.0;
            double rho_T = 0.0, rhou_T = 0.0, rhov_T = 0.0, E_T = 0.0;
            for (int iy = 0; iy < npts; ++iy) {
                int node = iy * npts + ix;
                double lL = basis.l_L[iy];
                double lR = basis.l_R[iy];
                rho_B  += c->U[0 * npts * npts + node] * lL;
                rhou_B += c->U[1 * npts * npts + node] * lL;
                rhov_B += c->U[2 * npts * npts + node] * lL;
                E_B    += c->U[3 * npts * npts + node] * lL;

                rho_T  += c->U[0 * npts * npts + node] * lR;
                rhou_T += c->U[1 * npts * npts + node] * lR;
                rhov_T += c->U[2 * npts * npts + node] * lR;
                E_T    += c->U[3 * npts * npts + node] * lR;
            }
            if (rho_B <= 0.0 || rho_T <= 0.0) { valid = false; break; }
            double P_B = (gamma - 1.0) * (E_B - 0.5 * (rhou_B * rhou_B + rhov_B * rhov_B) / rho_B);
            double P_T = (gamma - 1.0) * (E_T - 0.5 * (rhou_T * rhou_T + rhov_T * rhov_T) / rho_T);
            if (P_B <= 0.0 || P_T <= 0.0) { valid = false; break; }
        }
    }
    return valid;
}

// Copy state vectors between cells and flat vector
static void state_to_flat(const std::vector<CellDim<2>*>& cells, std::vector<double>& u_flat) {
    size_t total_dofs = 0;
    for (const auto* c : cells) total_dofs += c->U.size();
    u_flat.resize(total_dofs);

    size_t offset = 0;
    for (const auto* c : cells) {
        std::copy(c->U.begin(), c->U.end(), u_flat.begin() + offset);
        offset += c->U.size();
    }
}

static void flat_to_state(const std::vector<double>& u_flat, std::vector<CellDim<2>*>& cells) {
    size_t offset = 0;
    for (auto* c : cells) {
        std::copy(u_flat.begin() + offset, u_flat.begin() + offset + c->U.size(), c->U.begin());
        offset += c->U.size();
    }
}

template<typename PrecondType>
bool solve_jfnk_stage_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    const std::vector<double>& H_i,
    double gamma_dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    const PrecondType& M_op,
    ImplicitStats& stats
) {
    (void)basis;
    std::vector<double> u_curr;
    state_to_flat(cells, u_curr);
    size_t n_dofs = u_curr.size();

    // Ensure initial predictor state is physically realizable; fallback to H_i if unphysical
    flat_to_state(u_curr, cells);
    if (!check_realizability_2d(cells, basis, params.GAMMA)) {
        u_curr = H_i;
        flat_to_state(u_curr, cells);
    }

    std::vector<double> R_eff(n_dofs, 0.0);
    std::vector<double> G_val(n_dofs, 0.0);
    std::vector<double> G_pert(n_dofs, 0.0);
    std::vector<double> u_pert(n_dofs, 0.0);

    auto compute_stage_G = [&](const std::vector<double>& u_input, std::vector<double>& G_out) {
        flat_to_state(u_input, cells);
        compute_R_effective(cells, R_eff);
        stats.total_res_evals++;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_dofs; ++i) {
            G_out[i] = u_input[i] - H_i[i] - gamma_dt * R_eff[i];
        }
    };

    double newton_tol = params.IMPLICIT_NEWTON_TOL;
    int max_newton = params.IMPLICIT_MAX_NEWTON_ITERS;
    double prev_res_norm = 0.0;

    for (int iter = 0; iter < max_newton; ++iter) {
        stats.total_newton_iters++;
        compute_stage_G(u_curr, G_val);
        double res_norm = vec_rms_norm(G_val);
        stats.final_stage_res = res_norm;

        if (res_norm < newton_tol) {
            flat_to_state(u_curr, cells);
            return true;
        }

        double eta_k = 0.1;
        if (iter > 0 && prev_res_norm > 0.0) {
            double ratio = res_norm / prev_res_norm;
            eta_k = std::min(0.1, std::max(params.IMPLICIT_GMRES_TOL, 0.5 * ratio * ratio));
        } else {
            eta_k = std::min(0.1, std::max(params.IMPLICIT_GMRES_TOL, 0.1));
        }
        prev_res_norm = res_norm;

        // Matrix-free directional derivative J * v with state scaling
        auto jacvec = [&](const std::vector<double>& v, std::vector<double>& Jv_out) {
            double v_norm = vec_norm(v);
            if (v_norm == 0.0) {
                Jv_out.assign(n_dofs, 0.0);
                return;
            }
            double u_norm = vec_norm(u_curr);
            double eps = 1.0e-7 * (1.0 + u_norm) / (v_norm + 1.0e-12);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n_dofs; ++i) {
                u_pert[i] = u_curr[i] + eps * v[i];
            }
            compute_stage_G(u_pert, G_pert);

            double inv_eps = 1.0 / eps;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n_dofs; ++i) {
                Jv_out[i] = (G_pert[i] - G_val[i]) * inv_eps;
            }
        };

        // RHS for GMRES: -G
        std::vector<double> neg_G(n_dofs);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_dofs; ++i) neg_G[i] = -G_val[i];

        std::vector<double> delta_u(n_dofs, 0.0);
        gmres_solve(jacvec, neg_G, delta_u, M_op, eta_k, 1.0e-6, 25, 25, &stats.total_gmres_iters);

        // Realizability backtracking line search
        double alpha = 1.0;
        std::vector<double> u_trial(n_dofs);
        bool step_accepted = false;

        while (alpha > 1.0e-4) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n_dofs; ++i) {
                u_trial[i] = u_curr[i] + alpha * delta_u[i];
            }
            flat_to_state(u_trial, cells);

            if (check_realizability_2d(cells, basis, params.GAMMA)) {
                u_curr = u_trial;
                step_accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!step_accepted) {
            // Restore current state if line search failed
            flat_to_state(u_curr, cells);
            break;
        }
    }

    flat_to_state(u_curr, cells);
    return true;
}

bool solve_direct_block_jacobi_stage_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    const std::vector<double>& H_i,
    double gamma_dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    const BlockJacobiPreconditioner2D& M_op,
    ImplicitStats& stats
) {
    (void)basis;
    std::vector<double> u_curr;
    state_to_flat(cells, u_curr);
    size_t n_dofs = u_curr.size();

    std::vector<double> R_eff(n_dofs, 0.0);
    std::vector<double> G_val(n_dofs, 0.0);
    std::vector<double> neg_G(n_dofs, 0.0);
    std::vector<double> delta_u(n_dofs, 0.0);

    auto compute_stage_G = [&](const std::vector<double>& u_input, std::vector<double>& G_out) {
        flat_to_state(u_input, cells);
        compute_R_effective(cells, R_eff);
        stats.total_res_evals++;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_dofs; ++i) {
            G_out[i] = u_input[i] - H_i[i] - gamma_dt * R_eff[i];
        }
    };

    double newton_tol = params.IMPLICIT_NEWTON_TOL;
    int max_newton = params.IMPLICIT_MAX_NEWTON_ITERS;

    for (int iter = 0; iter < max_newton; ++iter) {
        stats.total_newton_iters++;
        compute_stage_G(u_curr, G_val);
        double res_norm = vec_rms_norm(G_val);
        stats.final_stage_res = res_norm;

        if (res_norm < newton_tol) {
            flat_to_state(u_curr, cells);
            return true;
        }

        // Direct analytical block-Jacobi update: delta_u = - M_op^{-1} * G
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_dofs; ++i) neg_G[i] = -G_val[i];

        M_op.apply(neg_G, delta_u);

        // Realizability positivity backtracking line search (0 extra residual evaluations!)
        double alpha = 1.0;
        std::vector<double> u_trial(n_dofs);
        bool step_accepted = false;

        while (alpha > 1.0e-4) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n_dofs; ++i) {
                u_trial[i] = u_curr[i] + alpha * delta_u[i];
            }
            flat_to_state(u_trial, cells);

            if (check_realizability_2d(cells, basis, params.GAMMA)) {
                u_curr = u_trial;
                step_accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!step_accepted) {
            flat_to_state(u_curr, cells);
            break;
        }
    }

    flat_to_state(u_curr, cells);
    return true;
}

bool solve_direct_ilu_stage_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    const std::vector<double>& H_i,
    double gamma_dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    const BlockILUPreconditioner2D& M_op,
    ImplicitStats& stats
) {
    (void)basis;
    std::vector<double> u_curr;
    state_to_flat(cells, u_curr);
    size_t n_dofs = u_curr.size();

    // Ensure initial predictor state is physically realizable; fallback to H_i if unphysical
    flat_to_state(u_curr, cells);
    if (!check_realizability_2d(cells, basis, params.GAMMA)) {
        u_curr = H_i;
        flat_to_state(u_curr, cells);
    }

    std::vector<double> R_eff(n_dofs, 0.0);
    std::vector<double> G_val(n_dofs, 0.0);
    std::vector<double> neg_G(n_dofs, 0.0);
    std::vector<double> delta_u(n_dofs, 0.0);

    auto compute_stage_G = [&](const std::vector<double>& u_input, std::vector<double>& G_out) {
        flat_to_state(u_input, cells);
        compute_R_effective(cells, R_eff);
        stats.total_res_evals++;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_dofs; ++i) {
            G_out[i] = u_input[i] - H_i[i] - gamma_dt * R_eff[i];
        }
    };

    double newton_tol = params.IMPLICIT_NEWTON_TOL;
    int max_newton = params.IMPLICIT_MAX_NEWTON_ITERS;

    for (int iter = 0; iter < max_newton; ++iter) {
        stats.total_newton_iters++;
        compute_stage_G(u_curr, G_val);
        double res_norm = vec_rms_norm(G_val);
        stats.final_stage_res = res_norm;

        if (res_norm < newton_tol) {
            flat_to_state(u_curr, cells);
            return true;
        }

        // Direct analytical Block-ILU(0) update: delta_u = - (L*U)^{-1} * G
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_dofs; ++i) neg_G[i] = -G_val[i];

        M_op.apply(neg_G, delta_u);

        // Realizability positivity backtracking line search (zero extra residual evaluations!)
        double alpha = 1.0;
        std::vector<double> u_trial(n_dofs);
        bool step_accepted = false;

        while (alpha > 1.0e-12) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n_dofs; ++i) {
                u_trial[i] = u_curr[i] + alpha * delta_u[i];
            }
            flat_to_state(u_trial, cells);

            if (check_realizability_2d(cells, basis, params.GAMMA)) {
                u_curr = u_trial;
                step_accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!step_accepted) {
            flat_to_state(u_curr, cells);
            return false;
        }
    }

    flat_to_state(u_curr, cells);
    return true;
}

void step_esdirk34_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    double dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    BlockJacobiPreconditioner2D& precond,
    BlockILUPreconditioner2D& ilu_precond,
    int& step_counter,
    ImplicitStats& stats_out
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    stats_out = ImplicitStats();
    stats_out.step = step_counter;

    // Build / freeze preconditioner based on IMPLICIT_SOLVER selection
    if (params.IMPLICIT_SOLVER == "ILU0" || params.IMPLICIT_SOLVER == "JFNK_ILU") {
        if (ilu_precond.inv_U_diag.empty() || (step_counter % params.IMPLICIT_PRECOND_FREEZE_STEPS == 0)) {
            ilu_precond.build(cells, basis, params.GAMMA, dt, ESDIRK34Tableau::gamma);
        }
    } else {
        if (precond.inv_M_blocks.empty() || (step_counter % params.IMPLICIT_PRECOND_FREEZE_STEPS == 0)) {
            precond.build(cells, basis, params.GAMMA, dt, ESDIRK34Tableau::gamma);
        }
    }

    std::vector<double> U_n;
    state_to_flat(cells, U_n);
    size_t n_dofs = U_n.size();

    // Stage storage
    std::vector<std::vector<double>> R_stages(ESDIRK34Tableau::s, std::vector<double>(n_dofs, 0.0));
    std::vector<std::vector<double>> U_stages(ESDIRK34Tableau::s, std::vector<double>(n_dofs, 0.0));
    U_stages[0] = U_n;

    // Stage 1: Explicit first stage U^{(1)} = U^n
    flat_to_state(U_n, cells);
    compute_R_effective(cells, R_stages[0]);
    stats_out.total_res_evals++;

    // Stages 2 to 4: Implicit solves
    for (int i = 1; i < ESDIRK34Tableau::s; ++i) {
        double gamma_stage = ESDIRK34Tableau::A[i][i];

        // Explicit stage history H_i = U^n + dt * sum_{j=0}^{i-1} A_{ij} * R^{(j)}
        std::vector<double> H_i(U_n);
        for (int j = 0; j < i; ++j) {
            double a_ij = ESDIRK34Tableau::A[i][j];
            if (a_ij != 0.0) {
                #pragma omp parallel for schedule(static)
                for (size_t k = 0; k < n_dofs; ++k) {
                    H_i[k] += dt * a_ij * R_stages[j][k];
                }
            }
        }

        // High-order stage initial guess predictor U_pred
        std::vector<double> U_pred(n_dofs);
        if (i == 1) { // Stage 2 (c2 = 2*gamma)
            double c2 = ESDIRK34Tableau::c[1];
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n_dofs; ++k) {
                U_pred[k] = U_n[k] + c2 * dt * R_stages[0][k];
            }
        } else if (i == 2) { // Stage 3 (c3 = 0.5)
            double c2 = ESDIRK34Tableau::c[1];
            double c3 = ESDIRK34Tableau::c[2];
            double ratio = c3 / c2;
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n_dofs; ++k) {
                U_pred[k] = U_n[k] + ratio * (U_stages[1][k] - U_n[k]);
            }
        } else { // Stage 4 (c4 = 1.0) - Quadratic Lagrange interpolation
            double c2 = ESDIRK34Tableau::c[1];
            double c3 = ESDIRK34Tableau::c[2];
            double w0 = (1.0 - c2) * (1.0 - c3) / (c2 * c3);
            double w2 = (1.0 - 0.0) * (1.0 - c3) / (c2 * (c2 - c3));
            double w3 = (1.0 - 0.0) * (1.0 - c2) / (c3 * (c3 - c2));
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n_dofs; ++k) {
                U_pred[k] = w0 * U_n[k] + w2 * U_stages[1][k] + w3 * U_stages[2][k];
            }
        }

        double gamma_dt = dt * gamma_stage;
        flat_to_state(U_pred, cells);
        if (params.IMPLICIT_SOLVER == "ILU0") {
            solve_direct_ilu_stage_2d(cells, basis, params, H_i, gamma_dt, compute_R_effective, ilu_precond, stats_out);
        } else if (params.IMPLICIT_SOLVER == "JFNK_ILU") {
            solve_jfnk_stage_2d(cells, basis, params, H_i, gamma_dt, compute_R_effective, ilu_precond, stats_out);
        } else if (params.IMPLICIT_SOLVER == "JFNK") {
            solve_jfnk_stage_2d(cells, basis, params, H_i, gamma_dt, compute_R_effective, precond, stats_out);
        } else {
            solve_direct_block_jacobi_stage_2d(cells, basis, params, H_i, gamma_dt, compute_R_effective, precond, stats_out);
        }

        // Store stage solution
        state_to_flat(cells, U_stages[i]);

        // Store stage residual R^{(i)}
        compute_R_effective(cells, R_stages[i]);
        stats_out.total_res_evals++;
    }

    // End of step: U^{n+1} = U^{(4)} (Stiffly Accurate property)
    // Final smooth limiter pass on new state
    if (params.ENABLE_POS_LIMITER) {
        #pragma omp parallel for schedule(static)
        for (size_t c_idx = 0; c_idx < cells.size(); ++c_idx) {
            Limiters::apply_full_limiter_2d(*cells[c_idx], basis, params.GAMMA);
        }
    }

    step_counter++;

    auto t_end = std::chrono::high_resolution_clock::now();
    stats_out.step_wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

} // namespace Implicit

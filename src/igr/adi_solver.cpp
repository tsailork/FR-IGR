/**
 * @file adi_solver.cpp
 * @brief Thomas algorithm and symmetrized ADI passes for the elliptic IGR solver.
 *
 * Implements the Alternating Direction Implicit (ADI) method to solve the Helmholtz-like
 * entropic pressure smoothing equations:
 * \f[ \left( I - \epsilon \nabla \cdot \left(\frac{1}{\rho} \nabla \right) \right) \Sigma = S \f]
 * Splitting this 2D operator into 1D directional passes allows solving each line as an
 * independent tridiagonal system in \f$O(N)\f$ time using the Thomas algorithm.
 * Double-pass averaging (X->Y and Y->X) is employed to preserve diagonal symmetry.
 */

#include "adi_solver.hpp"
#include "../core/solver.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Solve a tridiagonal system of linear equations using the Thomas algorithm.
 *
 * Solves a system of the form:
 * \f[ a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i \f]
 *
 * @param[in] a Lower diagonal coefficients (size N, index 0 is ignored)
 * @param[in] b Main diagonal coefficients (size N)
 * @param[in] c Upper diagonal coefficients (size N, index N-1 is ignored)
 * @param[in] d Right-hand side values (size N)
 * @param[out] x Solution vector (size N)
 */
void solve_tridiagonal(const std::vector<double>& a,
                       const std::vector<double>& b,
                       const std::vector<double>& c,
                       const std::vector<double>& d,
                       std::vector<double>& x)
{
    int n = static_cast<int>(d.size());
    if (n == 0) return;
    if (n == 1) { x[0] = d[0] / b[0]; return; }

    std::vector<double> cp(n), dp(n);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double den = b[i] - a[i] * cp[i - 1];
        if (std::abs(den) < 1e-15) den = 1e-15;
        cp[i] = c[i] / den;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = dp[i] - cp[i] * x[i + 1];
}

/**
 * @brief Execute a single directional ADI pass on a block.
 *
 * Performs sequential 1D sweeps in either X-then-Y or Y-then-X directions. Imposes Dirichlet
 * interface coupling across block boundaries and zero-gradient Neumann conditions at physical domain edges.
 *
 * @param[in,out] b Computational block to process
 * @param[in] S Source term vector for the ADI smoothing pass
 * @param[out] Out Smoothed output field
 * @param[in] x_first If true, sweep X first then Y; if false, sweep Y first then X
 */
void Solver::solve_adi_pass(Block& b, const std::vector<double>& S,
                            std::vector<double>& Out, bool x_first)
{
    int total_nodes = b.ny * b.nx * p.N_PTS * p.N_PTS;
    std::vector<double> Sigma_Star(total_nodes, 0.0);

    auto solve_direction = [&](bool horizontal,
                               const std::vector<double>& source,
                               std::vector<double>& output,
                               bool is_first_pass)
    {
        int n_outer = horizontal ? b.ny * p.N_PTS : b.nx * p.N_PTS;
        int n_elems = horizontal ? b.nx : b.ny;
        double h_elem = horizontal ? b.dx : b.dy;
        double eps = p.ALPHA_SCALE * (h_elem * h_elem);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_outer; ++i) {
            int n_1d = n_elems * p.N_PTS;
            std::vector<double> A(n_1d, 0.0), B(n_1d), C(n_1d, 0.0), RHS_1d(n_1d);

            auto get_idx = [&](int k) {
                if (horizontal) {
                    int ey = i / p.N_PTS, iy = i % p.N_PTS;
                    return b.get_flat_idx(ey, k / p.N_PTS, iy, k % p.N_PTS, p.N_PTS);
                } else {
                    int ex = i / p.N_PTS, ix = i % p.N_PTS;
                    return b.get_flat_idx(k / p.N_PTS, ex, k % p.N_PTS, ix, p.N_PTS);
                }
            };

            for (int k = 0; k < n_1d; ++k) {
                int e_idx = k / p.N_PTS, p_idx = k % p.N_PTS;
                double z_curr = basis.z[p_idx];

                double x_prev = (p_idx == 0)
                    ? ((e_idx - 1) + 0.5 * (1 + basis.z[p.N_PTS - 1])) * h_elem
                    : (e_idx + 0.5 * (1 + basis.z[p_idx - 1])) * h_elem;
                double x_curr = (e_idx + 0.5 * (1 + z_curr)) * h_elem;
                double x_next = (p_idx == p.N_PTS - 1)
                    ? ((e_idx + 1) + 0.5 * (1 + basis.z[0])) * h_elem
                    : (e_idx + 0.5 * (1 + basis.z[p_idx + 1])) * h_elem;

                double h_L = x_curr - x_prev, h_R = x_next - x_curr;
                double h_avg = 0.5 * (h_L + h_R);

                int flat = get_idx(k);
                double rho_curr = std::max(1e-12, b.U.data[flat * 4 + 0]);
                double rho_prev = (k > 0) ? std::max(1e-12, b.U.data[get_idx(k - 1) * 4 + 0]) : rho_curr;
                double rho_next = (k < n_1d - 1) ? std::max(1e-12, b.U.data[get_idx(k + 1) * 4 + 0]) : rho_curr;

                double K_L = eps / (0.5 * (rho_prev + rho_curr));
                double K_R = eps / (0.5 * (rho_next + rho_curr));

                A[k] = -K_L / (h_L * h_avg);
                C[k] = -K_R / (h_R * h_avg);
                B[k] = 1.0 / rho_curr + (K_L / h_L + K_R / h_R) / h_avg;

                RHS_1d[k] = is_first_pass ? source[flat] : source[flat] / rho_curr;
            }

            // Boundaries (Dirichlet coupling for block interfaces, zero-gradient Neumann for physical boundaries)
            {
                // Left / Bottom boundary (k = 0)
                bool has_neigh_0 = false;
                double sig_neigh_0 = 0.0;
                double dummy_state[4], dummy_face[4] = {0.0, 0.0, 0.0, 0.0};
                if (horizontal) {
                    int ey = i / p.N_PTS, iy = i % p.N_PTS;
                    if (b.ni_l.id != -1) {
                        has_neigh_0 = true;
                        get_neigh_state_x(b, ey, 0, iy, false, dummy_face, 0.0, dummy_state, sig_neigh_0);
                    }
                } else {
                    int ex = i / p.N_PTS, ix = i % p.N_PTS;
                    if (b.ni_b.id != -1) {
                        has_neigh_0 = true;
                        get_neigh_state_y(b, 0, ex, ix, false, dummy_face, 0.0, dummy_state, sig_neigh_0);
                    }
                }

                if (has_neigh_0) {
                    B[0] = 1.0;
                    C[0] = 0.0;
                    A[0] = 0.0;
                    RHS_1d[0] = sig_neigh_0;
                } else {
                    double rho0 = std::max(1e-12, b.U.data[get_idx(0) * 4 + 0]);
                    double rho1 = std::max(1e-12, b.U.data[get_idx(1) * 4 + 0]);
                    double h_R_s = (basis.z[1] - basis.z[0]) * 0.5 * h_elem;
                    double K_R = eps / (0.5 * (rho0 + rho1));
                    double t = K_R / (0.5 * h_R_s * h_R_s);
                    B[0] = 1.0 / rho0 + t;  C[0] = -t;  A[0] = 0.0;
                }

                // Right / Top boundary (k = last)
                int last = n_1d - 1;
                bool has_neigh_last = false;
                double sig_neigh_last = 0.0;
                if (horizontal) {
                    int ey = i / p.N_PTS, iy = i % p.N_PTS;
                    if (b.ni_r.id != -1) {
                        has_neigh_last = true;
                        get_neigh_state_x(b, ey, b.nx - 1, iy, true, dummy_face, 0.0, dummy_state, sig_neigh_last);
                    }
                } else {
                    int ex = i / p.N_PTS, ix = i % p.N_PTS;
                    if (b.ni_t.id != -1) {
                        has_neigh_last = true;
                        get_neigh_state_y(b, b.ny - 1, ex, ix, true, dummy_face, 0.0, dummy_state, sig_neigh_last);
                    }
                }

                if (has_neigh_last) {
                    B[last] = 1.0;
                    A[last] = 0.0;
                    C[last] = 0.0;
                    RHS_1d[last] = sig_neigh_last;
                } else {
                    double rhoN = std::max(1e-12, b.U.data[get_idx(last) * 4 + 0]);
                    double rhoNm1 = std::max(1e-12, b.U.data[get_idx(last - 1) * 4 + 0]);
                    double h_L_s = (basis.z[p.N_PTS - 1] - basis.z[p.N_PTS - 2]) * 0.5 * h_elem;
                    double K_L = eps / (0.5 * (rhoN + rhoNm1));
                    double tL = K_L / (0.5 * h_L_s * h_L_s);
                    B[last] = 1.0 / rhoN + tL;  A[last] = -tL;  C[last] = 0.0;
                }
            }

            std::vector<double> x_sol(n_1d);
            solve_tridiagonal(A, B, C, RHS_1d, x_sol);
            for (int k = 0; k < n_1d; ++k)
                output[get_idx(k)] = std::max(0.0, x_sol[k]);
        }
    };

    if (x_first) {
        solve_direction(true,  S,          Sigma_Star, true);
        solve_direction(false, Sigma_Star, Out,        false);
    } else {
        solve_direction(false, S,          Sigma_Star, true);
        solve_direction(true,  Sigma_Star, Out,        false);
    }
}

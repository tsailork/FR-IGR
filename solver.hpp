#pragma once

#include "parameters.hpp"
#include "state.hpp"
#include "basis.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// ============================================================================
// HELPER: THOMAS ALGORITHM FOR TRIDIAGONAL SYSTEMS
// Solves: A[i]*x[i-1] + B[i]*x[i] + C[i]*x[i+1] = D[i]
// ============================================================================
inline void solve_tridiagonal(const std::vector<double>& a, 
                              const std::vector<double>& b, 
                              const std::vector<double>& c, 
                              const std::vector<double>& d, 
                              std::vector<double>& x) {
    int n = (int)d.size();
    if (n == 0) return;
    if (n == 1) { x[0] = d[0] / b[0]; return; }

    std::vector<double> cp(n);
    std::vector<double> dp(n);

    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double den = b[i] - a[i] * cp[i - 1];
        if (std::abs(den) < 1e-15) den = 1e-15; 
        cp[i] = c[i] / den;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
}

// ============================================================================
// CLASS: SOLVER
// The main engine for the 2D IGR-FR method.
// ============================================================================
class Solver {
public:
    const Parameters& p;
    State U;            // Conservative Variables [rho, rhou, rhov, E]
    State RHS;          // Right Hand Side (update vector)
    Basis basis;        // 1D Basis functions (used for tensor product)
    
    // Geometry
    double dx, dy;
    
    // Entropic Pressure Field (Regularization)
    std::vector<double> sigma_field; 

    // Constructor
    Solver(const Parameters& params) : p(params), U(p), RHS(p), basis(p) {
        dx = (p.X_MAX - p.X_MIN) / p.N_ELEM_X;
        dy = (p.Y_MAX - p.Y_MIN) / p.N_ELEM_Y;
        sigma_field.resize(p.N_ELEM_Y * p.N_ELEM_X * p.N_PTS * p.N_PTS, 0.0);
    }

    // ========================================================================
    // SECTION 1: ENTROPIC PRESSURE (IGR)
    // ========================================================================

    // 1.1 Compute the Cao-Schaefer Sensor Source Term
    std::vector<double> compute_sensor_source() {
        int total_nodes = p.N_ELEM_Y * p.N_ELEM_X * p.N_PTS * p.N_PTS;
        std::vector<double> S(total_nodes, 0.0);
        
        // Consistent Scaling: epsilon = alpha * h^2
        // We use the element size (dx*dy) as the scaling factor for h^2
        double epsilon = p.ALPHA_SCALE * (dx * dy); 

        for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double du_dx = 0.0, du_dy = 0.0;
                        double dv_dx = 0.0, dv_dy = 0.0;

                        // X-Derivatives (Local within element using D matrix)
                        for(int k = 0; k < p.N_PTS; ++k) {
                            double rho_loc = std::max(1e-10, U(0, ey, ex, iy, k));
                            du_dx += basis.D[ix][k] * (U(1, ey, ex, iy, k) / rho_loc) * (2.0 / dx);
                            dv_dx += basis.D[ix][k] * (U(2, ey, ex, iy, k) / rho_loc) * (2.0 / dx);
                        }

                        // Y-Derivatives
                        for(int k = 0; k < p.N_PTS; ++k) {
                            double rho_loc = std::max(1e-10, U(0, ey, ex, k, ix));
                            du_dy += basis.D[iy][k] * (U(1, ey, ex, k, ix) / rho_loc) * (2.0 / dy);
                            dv_dy += basis.D[iy][k] * (U(2, ey, ex, k, ix) / rho_loc) * (2.0 / dy);
                        }

                        double source_val = 2.0*(du_dx*du_dx + dv_dy*dv_dy);
                        source_val += 2.0*(du_dx*dv_dy + dv_dx*du_dy);

                        // Filter and Cap
                        //if (source_val < 0.0) source_val = 0.0;
                        //if (source_val > 1000.0) source_val = 1000.0;

                        S[get_flat_idx(ey, ex, iy, ix)] = epsilon * source_val;
                    }
                }
            }
        }
        return S;
    }

    // Helper: Runs a single ADI pass (X then Y, or Y then X) with Non-Uniform spacing
    void solve_adi_pass(const std::vector<double>& S, std::vector<double>& Out, bool x_first) {
        int total_nodes = p.N_ELEM_Y * p.N_ELEM_X * p.N_PTS * p.N_PTS;
        std::vector<double> Sigma_Star(total_nodes, 0.0);

        // Consistent Scaling: epsilon = alpha * h^2
        double epsilon_x = p.ALPHA_SCALE * (dx * dx); // Use dx^2 for X-pass
        double epsilon_y = p.ALPHA_SCALE * (dy * dy); // Use dy^2 for Y-pass

        if (x_first) {
            // --- PASS 1: X-DIRECTION SOLVE ---
            for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    int n_1d = p.N_ELEM_X * p.N_PTS;
                    std::vector<double> A(n_1d), B(n_1d), C(n_1d), RHS_1d(n_1d);
                    
                    for (int k = 0; k < n_1d; ++k) {
                        int ex = k / p.N_PTS;
                        int ix = k % p.N_PTS;
                        
                        // Node coordinate (physical)
                        double x_curr = (ex + 0.5*(1+basis.z[ix])) * dx;
                        
                        // Previous and Next coordinates
                        double x_prev = (ix == 0) ? ((ex-1) + 0.5*(1+basis.z[p.N_PTS-1])) * dx : (ex + 0.5*(1+basis.z[ix-1])) * dx;
                        double x_next = (ix == p.N_PTS-1) ? ((ex+1) + 0.5*(1+basis.z[0])) * dx : (ex + 0.5*(1+basis.z[ix+1])) * dx;
                        
                        // Spacings
                        double h_L = x_curr - x_prev; // Distance to left node
                        double h_R = x_next - x_curr; // Distance to right node
                        double h_avg = 0.5 * (h_L + h_R); // Control volume width

                        // Densities
                        double rho_curr = std::max(1e-10, U(0, ey, ex, iy, ix));
                        
                        // Diffusion Coeffs at interfaces (Harmonic mean or avg density)
                        // Kappa = epsilon / rho
                        // K_L = epsilon / rho_{i-1/2}
                        double rho_prev = (k > 0) ? std::max(1e-10, U(0, ey, (k-1)/p.N_PTS, iy, (k-1)%p.N_PTS)) : rho_curr;
                        double rho_next = (k < n_1d-1) ? std::max(1e-10, U(0, ey, (k+1)/p.N_PTS, iy, (k+1)%p.N_PTS)) : rho_curr;
                        
                        double K_L = epsilon_x / (0.5*(rho_prev + rho_curr));
                        double K_R = epsilon_x / (0.5*(rho_next + rho_curr));

                        // Finite Difference: 1/h_avg * [ K_R * (u_next - u_curr)/h_R - K_L * (u_curr - u_prev)/h_L ]
                        // A * u_prev + B * u_curr + C * u_next
                        double coeff_A = -K_L / (h_L * h_avg);
                        double coeff_C = -K_R / (h_R * h_avg);
                        double coeff_B = (K_L/h_L + K_R/h_R) / h_avg;

                        RHS_1d[k] = S[get_flat_idx(ey, ex, iy, ix)];
                        
                        if (k > 0) A[k] = coeff_A;
                        if (k < n_1d - 1) C[k] = coeff_C;
                        B[k] = 1.0/rho_curr + coeff_B;
                    }

                    // Neumann BCs: Sigma_{-1} = Sigma_0 => flux_L = 0
                    // Effectively K_L = 0 at left boundary
                    // But we want zero gradient. 
                    // FD at boundary: (1/h_avg) * [ K_R * (u_1 - u_0)/h_R - 0 ] -> No flux out left
                    // So we just zero out the 'A' contribution effectively.
                    // Recalculate boundary B's to be consistent with zero flux
                    
                    // Left Boundary (k=0)
                    {
                        double x0 = (0 + 0.5*(1+basis.z[0])) * dx;
                        double x1 = (0 + 0.5*(1+basis.z[1])) * dx;
                        double h_R = x1 - x0;
                        double h_avg = 0.5 * h_R; // Half-volume
                        double rho0 = std::max(1e-10, U(0, ey, 0, iy, 0));
                        double rho1 = std::max(1e-10, U(0, ey, 0, iy, 1));
                        double K_R = epsilon_x / (0.5*(rho0+rho1));
                        
                        // Eq: (1/rho)*u0 - (1/h_avg) * K_R * (u1 - u0)/h_R = S
                        // (1/rho + K_R/(h_avg*h_R)) * u0 - (K_R/(h_avg*h_R)) * u1 = S
                        double term = K_R / (h_avg * h_R);
                        B[0] = 1.0/rho0 + term;
                        C[0] = -term; 
                        A[0] = 0.0;
                    }

                    // Right Boundary (k=N-1)
                    {
                        int ex = p.N_ELEM_X - 1;
                        int ix = p.N_PTS - 1;
                        double xN = (ex + 0.5*(1+basis.z[ix])) * dx;
                        double xNm1 = (ex + 0.5*(1+basis.z[ix-1])) * dx;
                        double h_L = xN - xNm1;
                        double h_avg = 0.5 * h_L;
                        double rhoN = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double rhoNm1 = std::max(1e-10, U(0, ey, ex, iy, ix-1));
                        double K_L = epsilon_x / (0.5*(rhoN+rhoNm1));

                        // Eq: (1/rho)*uN + (1/h_avg) * K_L * (uN - uNm1)/h_L = S
                        double term = K_L / (h_avg * h_L);
                        B[n_1d-1] = 1.0/rhoN + term;
                        A[n_1d-1] = -term;
                        C[n_1d-1] = 0.0;
                    }

                    std::vector<double> x_sol(n_1d);
                    solve_tridiagonal(A, B, C, RHS_1d, x_sol);
                    for(int k=0; k<n_1d; ++k) Sigma_Star[get_flat_idx(ey, k/p.N_PTS, iy, k%p.N_PTS)] = x_sol[k];
                }
            }

            // --- PASS 2: Y-DIRECTION SOLVE ---
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int n_1d = p.N_ELEM_Y * p.N_PTS;
                    std::vector<double> A(n_1d), B(n_1d), C(n_1d), RHS_1d(n_1d);

                    for (int k = 0; k < n_1d; ++k) {
                        int ey = k / p.N_PTS;
                        int iy = k % p.N_PTS;
                        
                        double y_curr = (ey + 0.5*(1+basis.z[iy])) * dy;
                        double y_prev = (iy == 0) ? ((ey-1) + 0.5*(1+basis.z[p.N_PTS-1])) * dy : (ey + 0.5*(1+basis.z[iy-1])) * dy;
                        double y_next = (iy == p.N_PTS-1) ? ((ey+1) + 0.5*(1+basis.z[0])) * dy : (ey + 0.5*(1+basis.z[iy+1])) * dy;
                        
                        double h_L = y_curr - y_prev;
                        double h_R = y_next - y_curr;
                        double h_avg = 0.5 * (h_L + h_R);

                        double rho_curr = std::max(1e-10, U(0, ey, ex, iy, ix));
                        
                        double rho_prev = (k > 0) ? std::max(1e-10, U(0, (k-1)/p.N_PTS, ex, (k-1)%p.N_PTS, ix)) : rho_curr;
                        double rho_next = (k < n_1d-1) ? std::max(1e-10, U(0, (k+1)/p.N_PTS, ex, (k+1)%p.N_PTS, ix)) : rho_curr;
                        
                        double K_L = epsilon_y / (0.5*(rho_prev + rho_curr));
                        double K_R = epsilon_y / (0.5*(rho_next + rho_curr));

                        double coeff_A = -K_L / (h_L * h_avg);
                        double coeff_C = -K_R / (h_R * h_avg);
                        double coeff_B = (K_L/h_L + K_R/h_R) / h_avg;

                        RHS_1d[k] = Sigma_Star[get_flat_idx(ey, ex, iy, ix)] / rho_curr;
                        
                        if (k > 0) A[k] = coeff_A;
                        if (k < n_1d - 1) C[k] = coeff_C;
                        B[k] = 1.0/rho_curr + coeff_B;
                    }

                    // Bottom BC (k=0)
                    {
                        double y0 = (0 + 0.5*(1+basis.z[0])) * dy;
                        double y1 = (0 + 0.5*(1+basis.z[1])) * dy;
                        double h_R = y1 - y0;
                        double h_avg = 0.5 * h_R;
                        double rho0 = std::max(1e-10, U(0, 0, ex, 0, ix));
                        double rho1 = std::max(1e-10, U(0, 0, ex, 1, ix));
                        double K_R = epsilon_y / (0.5*(rho0+rho1));
                        
                        double term = K_R / (h_avg * h_R);
                        B[0] = 1.0/rho0 + term;
                        C[0] = -term; 
                        A[0] = 0.0;
                    }

                    // Top BC (k=N-1)
                    {
                        int ey = p.N_ELEM_Y - 1;
                        int iy = p.N_PTS - 1;
                        double yN = (ey + 0.5*(1+basis.z[iy])) * dy;
                        double yNm1 = (ey + 0.5*(1+basis.z[iy-1])) * dy;
                        double h_L = yN - yNm1;
                        double h_avg = 0.5 * h_L;
                        double rhoN = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double rhoNm1 = std::max(1e-10, U(0, ey, ex, iy-1, ix));
                        double K_L = epsilon_y / (0.5*(rhoN+rhoNm1));

                        double term = K_L / (h_avg * h_L);
                        B[n_1d-1] = 1.0/rhoN + term;
                        A[n_1d-1] = -term;
                        C[n_1d-1] = 0.0;
                    }

                    std::vector<double> x_sol(n_1d);
                    solve_tridiagonal(A, B, C, RHS_1d, x_sol);
                    for(int k=0; k<n_1d; ++k) Out[get_flat_idx(k/p.N_PTS, ex, k%p.N_PTS, ix)] = std::max(0.0, x_sol[k]);
                }
            }
        } else {
            // Y-FIRST ORDER
            
            // --- PASS 1: Y-DIRECTION SOLVE ---
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    int n_1d = p.N_ELEM_Y * p.N_PTS;
                    std::vector<double> A(n_1d), B(n_1d), C(n_1d), RHS_1d(n_1d);

                    for (int k = 0; k < n_1d; ++k) {
                        int ey = k / p.N_PTS;
                        int iy = k % p.N_PTS;
                        
                        double y_curr = (ey + 0.5*(1+basis.z[iy])) * dy;
                        double y_prev = (iy == 0) ? ((ey-1) + 0.5*(1+basis.z[p.N_PTS-1])) * dy : (ey + 0.5*(1+basis.z[iy-1])) * dy;
                        double y_next = (iy == p.N_PTS-1) ? ((ey+1) + 0.5*(1+basis.z[0])) * dy : (ey + 0.5*(1+basis.z[iy+1])) * dy;
                        
                        double h_L = y_curr - y_prev;
                        double h_R = y_next - y_curr;
                        double h_avg = 0.5 * (h_L + h_R);

                        double rho_curr = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double rho_prev = (k > 0) ? std::max(1e-10, U(0, (k-1)/p.N_PTS, ex, (k-1)%p.N_PTS, ix)) : rho_curr;
                        double rho_next = (k < n_1d-1) ? std::max(1e-10, U(0, (k+1)/p.N_PTS, ex, (k+1)%p.N_PTS, ix)) : rho_curr;
                        
                        double K_L = epsilon_y / (0.5*(rho_prev + rho_curr));
                        double K_R = epsilon_y / (0.5*(rho_next + rho_curr));

                        double coeff_A = -K_L / (h_L * h_avg);
                        double coeff_C = -K_R / (h_R * h_avg);
                        double coeff_B = (K_L/h_L + K_R/h_R) / h_avg;

                        RHS_1d[k] = S[get_flat_idx(ey, ex, iy, ix)];
                        if (k > 0) A[k] = coeff_A;
                        if (k < n_1d - 1) C[k] = coeff_C;
                        B[k] = 1.0/rho_curr + coeff_B;
                    }

                    // Bottom BC
                    {
                        double y0 = (0 + 0.5*(1+basis.z[0])) * dy;
                        double y1 = (0 + 0.5*(1+basis.z[1])) * dy;
                        double h_R = y1 - y0;
                        double h_avg = 0.5 * h_R;
                        double rho0 = std::max(1e-10, U(0, 0, ex, 0, ix));
                        double rho1 = std::max(1e-10, U(0, 0, ex, 1, ix));
                        double K_R = epsilon_y / (0.5*(rho0+rho1));
                        double term = K_R / (h_avg * h_R);
                        B[0] = 1.0/rho0 + term;
                        C[0] = -term; A[0] = 0.0;
                    }
                    // Top BC
                    {
                        int ey = p.N_ELEM_Y - 1; int iy = p.N_PTS - 1;
                        double yN = (ey + 0.5*(1+basis.z[iy])) * dy;
                        double yNm1 = (ey + 0.5*(1+basis.z[iy-1])) * dy;
                        double h_L = yN - yNm1;
                        double h_avg = 0.5 * h_L;
                        double rhoN = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double rhoNm1 = std::max(1e-10, U(0, ey, ex, iy-1, ix));
                        double K_L = epsilon_y / (0.5*(rhoN+rhoNm1));
                        double term = K_L / (h_avg * h_L);
                        B[n_1d-1] = 1.0/rhoN + term;
                        A[n_1d-1] = -term; C[n_1d-1] = 0.0;
                    }

                    std::vector<double> x_sol(n_1d);
                    solve_tridiagonal(A, B, C, RHS_1d, x_sol);
                    for(int k=0; k<n_1d; ++k) Sigma_Star[get_flat_idx(k/p.N_PTS, ex, k%p.N_PTS, ix)] = x_sol[k];
                }
            }

            // --- PASS 2: X-DIRECTION SOLVE ---
            for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    int n_1d = p.N_ELEM_X * p.N_PTS;
                    std::vector<double> A(n_1d), B(n_1d), C(n_1d), RHS_1d(n_1d);
                    
                    for (int k = 0; k < n_1d; ++k) {
                        int ex = k / p.N_PTS;
                        int ix = k % p.N_PTS;
                        
                        double x_curr = (ex + 0.5*(1+basis.z[ix])) * dx;
                        double x_prev = (ix == 0) ? ((ex-1) + 0.5*(1+basis.z[p.N_PTS-1])) * dx : (ex + 0.5*(1+basis.z[ix-1])) * dx;
                        double x_next = (ix == p.N_PTS-1) ? ((ex+1) + 0.5*(1+basis.z[0])) * dx : (ex + 0.5*(1+basis.z[ix+1])) * dx;
                        
                        double h_L = x_curr - x_prev;
                        double h_R = x_next - x_curr;
                        double h_avg = 0.5 * (h_L + h_R);

                        double rho_curr = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double rho_prev = (k > 0) ? std::max(1e-10, U(0, ey, (k-1)/p.N_PTS, iy, (k-1)%p.N_PTS)) : rho_curr;
                        double rho_next = (k < n_1d-1) ? std::max(1e-10, U(0, ey, (k+1)/p.N_PTS, iy, (k+1)%p.N_PTS)) : rho_curr;
                        
                        double K_L = epsilon_x / (0.5*(rho_prev + rho_curr));
                        double K_R = epsilon_x / (0.5*(rho_next + rho_curr));

                        double coeff_A = -K_L / (h_L * h_avg);
                        double coeff_C = -K_R / (h_R * h_avg);
                        double coeff_B = (K_L/h_L + K_R/h_R) / h_avg;

                        RHS_1d[k] = Sigma_Star[get_flat_idx(ey, ex, iy, ix)] / rho_curr;
                        if (k > 0) A[k] = coeff_A;
                        if (k < n_1d - 1) C[k] = coeff_C;
                        B[k] = 1.0/rho_curr + coeff_B;
                    }

                    // Left BC
                    {
                        double x0 = (0 + 0.5*(1+basis.z[0])) * dx;
                        double x1 = (0 + 0.5*(1+basis.z[1])) * dx;
                        double h_R = x1 - x0;
                        double h_avg = 0.5 * h_R;
                        double rho0 = std::max(1e-10, U(0, ey, 0, iy, 0));
                        double rho1 = std::max(1e-10, U(0, ey, 0, iy, 1));
                        double K_R = epsilon_x / (0.5*(rho0+rho1));
                        double term = K_R / (h_avg * h_R);
                        B[0] = 1.0/rho0 + term;
                        C[0] = -term; A[0] = 0.0;
                    }
                    // Right BC
                    {
                        int ex = p.N_ELEM_X - 1; int ix = p.N_PTS - 1;
                        double xN = (ex + 0.5*(1+basis.z[ix])) * dx;
                        double xNm1 = (ex + 0.5*(1+basis.z[ix-1])) * dx;
                        double h_L = xN - xNm1;
                        double h_avg = 0.5 * h_L;
                        double rhoN = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double rhoNm1 = std::max(1e-10, U(0, ey, ex, iy, ix-1));
                        double K_L = epsilon_x / (0.5*(rhoN+rhoNm1));
                        double term = K_L / (h_avg * h_L);
                        B[n_1d-1] = 1.0/rhoN + term;
                        A[n_1d-1] = -term; C[n_1d-1] = 0.0;
                    }

                    std::vector<double> x_sol(n_1d);
                    solve_tridiagonal(A, B, C, RHS_1d, x_sol);
                    for(int k=0; k<n_1d; ++k) Out[get_flat_idx(ey, k/p.N_PTS, iy, k%p.N_PTS)] = std::max(0.0, x_sol[k]);
                }
            }
        }
    }

    // 1.2 The Main ADI Solver (Symmetrized)
    void compute_entropic_pressure() {
        if (!p.ENABLE_IGR) return;

        std::vector<double> S = compute_sensor_source();
        
        // Calculate XY pass
        std::vector<double> Sigma_XY(S.size());
        solve_adi_pass(S, Sigma_XY, true);

        // Calculate YX pass
        std::vector<double> Sigma_YX(S.size());
        solve_adi_pass(S, Sigma_YX, false);

        // Average for Symmetry
        for(size_t i=0; i<sigma_field.size(); ++i) {
            sigma_field[i] = 0.5 * (Sigma_XY[i] + Sigma_YX[i]);
        }
    }

    // ========================================================================
    // SECTION 2: FLUX RECONSTRUCTION PHYSICS
    // ========================================================================

    void get_flux_pointwise(int ey, int ex, int iy, int ix, 
                            double* F, double* G, double sigma) {
        double rho = std::max(1e-10, U(0, ey, ex, iy, ix));
        double u   = U(1, ey, ex, iy, ix) / rho;
        double v   = U(2, ey, ex, iy, ix) / rho;
        double E   = U(3, ey, ex, iy, ix);
        double press = std::max(1e-10, (p.GAMMA - 1.0) * (E - 0.5 * rho * (u*u + v*v)));

        if (F) {
            F[0] = rho * u;
            F[1] = rho * u * u + press + sigma;
            F[2] = rho * u * v;
            F[3] = (E + press) * u;
        }

        if (G) {
            G[0] = rho * v;
            G[1] = rho * v * u;
            G[2] = rho * v * v + press + sigma;
            G[3] = (E + press) * v;
        }
    }

    void solve_riemann(double* UL, double* UR, double* F_comm, int dir, double sigl, double sigr) {
        double rhoL=std::max(1e-10, UL[0]), uL=UL[1]/rhoL, vL=UL[2]/rhoL, pL=std::max(1e-10, (p.GAMMA-1)*(UL[3]-0.5*rhoL*(uL*uL+vL*vL)));
        double rhoR=std::max(1e-10, UR[0]), uR=UR[1]/rhoR, vR=UR[2]/rhoR, pR=std::max(1e-10, (p.GAMMA-1)*(UR[3]-0.5*rhoR*(uR*uR+vR*vR)));

        double vnL = (dir == 0) ? uL : vL;
        double vnR = (dir == 0) ? uR : vR;

        double cL = std::sqrt(p.GAMMA * pL / rhoL);
        double cR = std::sqrt(p.GAMMA * pR / rhoR);
        double max_wave = std::max(std::abs(vnL) + cL, std::abs(vnR) + cR);

        double FL[4], FR[4];
        if(dir==0) {
            FL[0]=rhoL*uL; FL[1]=rhoL*uL*uL+pL+sigl; FL[2]=rhoL*uL*vL; FL[3]=(UL[3]+pL)*uL;
            FR[0]=rhoR*uR; FR[1]=rhoR*uR*uR+pR+sigr; FR[2]=rhoR*uR*vR; FR[3]=(UR[3]+pR)*uR;
        } else {
            FL[0]=rhoL*vL; FL[1]=rhoL*vL*uL; FL[2]=rhoL*vL*vL+pL+sigl; FL[3]=(UL[3]+pL)*vL;
            FR[0]=rhoR*vR; FR[1]=rhoR*vR*uR; FR[2]=rhoR*vR*vR+pR+sigr; FR[3]=(UR[3]+pR)*vR;
        }

        for(int v=0; v<4; ++v) F_comm[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * max_wave * (UR[v] - UL[v]);
    }

    void compute_rhs() {
        std::fill(RHS.data.begin(), RHS.data.end(), 0.0);
        compute_entropic_pressure();

        // X-SWEEP
        for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
            for (int iy = 0; iy < p.N_PTS; ++iy) { 
                for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                    std::vector<std::vector<double>> F_sol(p.N_PTS, std::vector<double>(4));
                    for(int ix=0; ix<p.N_PTS; ++ix) get_flux_pointwise(ey, ex, iy, ix, F_sol[ix].data(), nullptr, sigma_field[get_flat_idx(ey, ex, iy, ix)]);

                    double UL_face[4]={0}, UR_face[4]={0}, sig_L_face=0, sig_R_face=0;
                    for(int ix=0; ix<p.N_PTS; ++ix) {
                        double s = sigma_field[get_flat_idx(ey, ex, iy, ix)];
                        sig_L_face += s * basis.l_L[ix]; sig_R_face += s * basis.l_R[ix];
                        for(int v=0; v<4; ++v) {
                            UL_face[v] += U(v, ey, ex, iy, ix) * basis.l_L[ix];
                            UR_face[v] += U(v, ey, ex, iy, ix) * basis.l_R[ix];
                        }
                    }

                    double Flux_L_comm[4], Flux_R_comm[4], sig_neigh=0, U_neigh[4];
                    // Left Interface
                    if(ex == 0) { for(int v=0; v<4; ++v) U_neigh[v]=UL_face[v]; sig_neigh=sig_L_face; }
                    else {
                        sig_neigh=0; for(int v=0; v<4; ++v) { U_neigh[v]=0; for(int k=0; k<p.N_PTS; ++k) {
                            U_neigh[v] += U(v, ey, ex-1, iy, k) * basis.l_R[k];
                            if(v==0) sig_neigh += sigma_field[get_flat_idx(ey, ex-1, iy, k)] * basis.l_R[k];
                        } }
                    }
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 0, sig_neigh, sig_L_face);
                    // Right Interface
                    if(ex == p.N_ELEM_X-1) { for(int v=0; v<4; ++v) U_neigh[v]=UR_face[v]; sig_neigh=sig_R_face; }
                    else {
                        sig_neigh=0; for(int v=0; v<4; ++v) { U_neigh[v]=0; for(int k=0; k<p.N_PTS; ++k) {
                            U_neigh[v] += U(v, ey, ex+1, iy, k) * basis.l_L[k];
                            if(v==0) sig_neigh += sigma_field[get_flat_idx(ey, ex+1, iy, k)] * basis.l_L[k];
                        } }
                    }
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 0, sig_R_face, sig_neigh);

                    double F_L[4]={0}, F_R[4]={0};
                    for(int ix=0; ix<p.N_PTS; ++ix) for(int v=0; v<4; ++v) { F_L[v]+=F_sol[ix][v]*basis.l_L[ix]; F_R[v]+=F_sol[ix][v]*basis.l_R[ix]; }

                    for(int ix=0; ix<p.N_PTS; ++ix) for(int v=0; v<4; ++v) {
                        double df = 0; for(int k=0; k<p.N_PTS; ++k) df += basis.D[ix][k] * F_sol[k][v];
                        RHS(v, ey, ex, iy, ix) -= (df + (Flux_L_comm[v]-F_L[v])*basis.dgl[ix] + (Flux_R_comm[v]-F_R[v])*basis.dgr[ix]) * (2.0/dx);
                    }
                }
            }
        }

        // Y-SWEEP
        for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
            for (int ix = 0; ix < p.N_PTS; ++ix) { 
                for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
                    std::vector<std::vector<double>> G_sol(p.N_PTS, std::vector<double>(4));
                    for(int iy=0; iy<p.N_PTS; ++iy) get_flux_pointwise(ey, ex, iy, ix, nullptr, G_sol[iy].data(), sigma_field[get_flat_idx(ey, ex, iy, ix)]);

                    double UL_face[4]={0}, UR_face[4]={0}, sig_L_face=0, sig_R_face=0;
                    for(int iy=0; iy<p.N_PTS; ++iy) {
                        double s = sigma_field[get_flat_idx(ey, ex, iy, ix)];
                        sig_L_face += s * basis.l_L[iy]; sig_R_face += s * basis.l_R[iy];
                        for(int v=0; v<4; ++v) {
                            UL_face[v] += U(v, ey, ex, iy, ix) * basis.l_L[iy];
                            UR_face[v] += U(v, ey, ex, iy, ix) * basis.l_R[iy];
                        }
                    }

                    double Flux_L_comm[4], Flux_R_comm[4], sig_neigh=0, U_neigh[4];
                    // Bottom
                    if(ey == 0) { for(int v=0; v<4; ++v) U_neigh[v]=UL_face[v]; sig_neigh=sig_L_face; }
                    else {
                        sig_neigh=0; for(int v=0; v<4; ++v) { U_neigh[v]=0; for(int k=0; k<p.N_PTS; ++k) {
                            U_neigh[v] += U(v, ey-1, ex, k, ix) * basis.l_R[k];
                            if(v==0) sig_neigh += sigma_field[get_flat_idx(ey-1, ex, k, ix)] * basis.l_R[k];
                        } }
                    }
                    solve_riemann(U_neigh, UL_face, Flux_L_comm, 1, sig_neigh, sig_L_face);
                    // Top
                    if(ey == p.N_ELEM_Y-1) { for(int v=0; v<4; ++v) U_neigh[v]=UR_face[v]; sig_neigh=sig_R_face; }
                    else {
                        sig_neigh=0; for(int v=0; v<4; ++v) { U_neigh[v]=0; for(int k=0; k<p.N_PTS; ++k) {
                            U_neigh[v] += U(v, ey+1, ex, k, ix) * basis.l_L[k];
                            if(v==0) sig_neigh += sigma_field[get_flat_idx(ey+1, ex, k, ix)] * basis.l_L[k];
                        } }
                    }
                    solve_riemann(UR_face, U_neigh, Flux_R_comm, 1, sig_R_face, sig_neigh);

                    double G_L[4]={0}, G_R[4]={0};
                    for(int iy=0; iy<p.N_PTS; ++iy) for(int v=0; v<4; ++v) { G_L[v]+=G_sol[iy][v]*basis.l_L[iy]; G_R[v]+=G_sol[iy][v]*basis.l_R[iy]; }

                    for(int iy=0; iy<p.N_PTS; ++iy) for(int v=0; v<4; ++v) {
                        double dg = 0; for(int k=0; k<p.N_PTS; ++k) dg += basis.D[iy][k] * G_sol[k][v];
                        RHS(v, ey, ex, iy, ix) -= (dg + (Flux_L_comm[v]-G_L[v])*basis.dgl[iy] + (Flux_R_comm[v]-G_R[v])*basis.dgr[iy]) * (2.0/dy);
                    }
                }
            }
        }
    }

    void check_stability() {
        for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double rho = U(0, ey, ex, iy, ix);
                        double rhou = U(1, ey, ex, iy, ix), rhov = U(2, ey, ex, iy, ix), E = U(3, ey, ex, iy, ix);
                        double press = (p.GAMMA - 1.0) * (E - 0.5 * (rhou*rhou + rhov*rhov) / rho);
                        if (std::isnan(rho) || std::isnan(press) || rho <= 0.0 || press <= 0.0) {
                            std::cerr << "\n[STABILITY ERROR] at (" << ex << "," << ey << ") node (" << ix << "," << iy << ") rho=" << rho << " p=" << press << "\n";
                            exit(1);
                        }
                    }
                }
            }
        }
    }

    double compute_dt() {
        double max_lambda = 0.0;
        for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
            for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double rho = std::max(1e-10, U(0, ey, ex, iy, ix));
                        double u = U(1, ey, ex, iy, ix) / rho;
                        double v = U(2, ey, ex, iy, ix) / rho;
                        double press = std::max(1e-10, (p.GAMMA - 1.0) * (U(3, ey, ex, iy, ix) - 0.5 * rho * (u*u + v*v)));
                        double c = std::sqrt(p.GAMMA * press / rho);
                        max_lambda = std::max({max_lambda, std::abs(u) + c, std::abs(v) + c});
                    }
                }
            }
        }
        double h = std::min(dx, dy);
        //return 0.5 * p.CFL * h / (max_lambda * (2 * p.P_DEG + 1));
        return 0.5 * p.CFL * h / (max_lambda * ((p.P_DEG+1) * (p.P_DEG+1)));
    }

    void step_rk3(double dt) {
        State U_old = U, U_star = U;
        compute_rhs(); for(size_t i=0; i<U.data.size(); ++i) U.data[i] = U_old.data[i] + dt * RHS.data[i];
        check_stability();
        compute_rhs(); for(size_t i=0; i<U.data.size(); ++i) U_star.data[i] = 0.75*U_old.data[i] + 0.25*(U.data[i] + dt*RHS.data[i]);
        U = U_star; check_stability();
        compute_rhs(); for(size_t i=0; i<U.data.size(); ++i) U.data[i] = (1.0/3.0)*U_old.data[i] + (2.0/3.0)*(U.data[i] + dt*RHS.data[i]);
        check_stability();
    }

private:
    inline int get_flat_idx(int ey, int ex, int iy, int ix) {
        return ey * (p.N_ELEM_X * p.N_PTS * p.N_PTS) + ex * (p.N_PTS * p.N_PTS) + iy * p.N_PTS + ix;
    }
};

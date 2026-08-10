/**
 * @file limiter_smooth.cpp
 * @brief Implementation of C1-Smooth LMEP Modal Filter and Zhang-Shu Positivity Limiter.
 */

#include "limiter_smooth.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace Limiters {

void apply_smooth_lmep_filter_2d(CellDim<2>& cell, const Basis& basis, double gamma, double ks, double eta, double beta, int s_power) {
    const int npts = basis.z.size();
    const int p = npts - 1;
    if (p <= 0) return;

    const int dofs = npts * npts;
    if (dofs > 64) return;

    double s_nodes[64];
    double w_2d[64];

    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            int idx = iy * npts + ix;
            double rho = std::max(cell.U[0 * dofs + idx], 1e-12);
            double rhou = cell.U[1 * dofs + idx];
            double rhov = cell.U[2 * dofs + idx];
            double E = cell.U[3 * dofs + idx];

            double u = rhou / rho;
            double v = rhov / rho;
            double internal_energy = E - 0.5 * rho * (u * u + v * v);
            double P = std::max((gamma - 1.0) * internal_energy, 1e-12);

            s_nodes[idx] = std::log(P) - gamma * std::log(rho);
            w_2d[idx] = basis.w[iy] * basis.w[ix];
        }
    }

    // Compute smooth minimum over solution nodes
    double min_s = s_nodes[0];
    double sum_w = 0.0;
    for (int i = 0; i < dofs; ++i) {
        min_s = std::min(min_s, s_nodes[i]);
        sum_w += w_2d[i];
    }
    double exp_sum = 0.0;
    for (int i = 0; i < dofs; ++i) {
        exp_sum += (w_2d[i] / sum_w) * std::exp(-beta * (s_nodes[i] - min_s));
    }
    double s_min_elem = min_s - (1.0 / beta) * std::log(std::max(1e-15, exp_sum));

    // Compute smooth entropy violation metric
    double E_elem = 0.0;
    for (int idx = 0; idx < dofs; ++idx) {
        double violation = softplus(s_min_elem - s_nodes[idx], eta);
        E_elem += violation * violation;
    }
    E_elem /= dofs;

    double sigma_elem = 1.0 - std::exp(-ks * E_elem);
    if (sigma_elem < 1e-10) return; // Early exit on smooth flow

    double U_hat[64];
    double U_filt[64];

    // Apply modal decay to polynomial modes m_y, m_x >= 1
    for (int v = 0; v < 4; ++v) {
        // Nodal -> Modal transform: U_hat = V_inv_y * U * V_inv_x^T
        for (int my = 0; my < npts; ++my) {
            for (int mx = 0; mx < npts; ++mx) {
                double val = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    for (int ix = 0; ix < npts; ++ix) {
                        val += basis.V_inv(my, iy) * basis.V_inv(mx, ix) * cell.U[v * dofs + iy * npts + ix];
                    }
                }
                U_hat[my * npts + mx] = val;
            }
        }

        // Apply decay function Phi(m)
        for (int my = 0; my < npts; ++my) {
            for (int mx = 0; mx < npts; ++mx) {
                int max_m = std::max(my, mx);
                double ratio = static_cast<double>(max_m) / std::max(p, 1);
                double Phi = 1.0 - sigma_elem * std::pow(ratio, 2 * s_power);
                U_hat[my * npts + mx] *= Phi;
            }
        }

        // Modal -> Nodal transform: U_filt = V_y * U_hat * V_x^T
        double orig_sum = 0.0;
        double filt_sum = 0.0;
        double w_total = 0.0;

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                int idx = iy * npts + ix;
                double val = 0.0;
                for (int my = 0; my < npts; ++my) {
                    for (int mx = 0; mx < npts; ++mx) {
                        val += basis.V(iy, my) * basis.V(ix, mx) * U_hat[my * npts + mx];
                    }
                }
                U_filt[idx] = val;
                double w = w_2d[idx];
                orig_sum += w * cell.U[v * dofs + idx];
                filt_sum += w * val;
                w_total += w;
            }
        }

        double orig_avg = orig_sum / w_total;
        double filt_avg = filt_sum / w_total;
        double diff = orig_avg - filt_avg;

        for (int idx = 0; idx < dofs; ++idx) {
            cell.U[v * dofs + idx] = U_filt[idx] + diff;
        }
    }
}

void apply_zhang_shu_limiter_2d(CellDim<2>& cell, const Basis& basis, double gamma, double eps) {
    const int npts = basis.z.size();
    const int dofs = npts * npts;

    // 1. Compute element cell average
    double U_avg[4] = {0.0, 0.0, 0.0, 0.0};
    double w_sum = 0.0;
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            int idx = iy * npts + ix;
            double w = basis.w[iy] * basis.w[ix];
            w_sum += w;
            for (int v = 0; v < 4; ++v) {
                U_avg[v] += w * cell.U[v * dofs + idx];
            }
        }
    }
    for (int v = 0; v < 4; ++v) U_avg[v] /= w_sum;

    // Check minimum density across interior solution nodes AND interface face points
    double rho_min = U_avg[0];

    // Check interior nodes
    for (int idx = 0; idx < dofs; ++idx) {
        rho_min = std::min(rho_min, cell.U[0 * dofs + idx]);
    }

    // Check Left/Right face points
    for (int iy = 0; iy < npts; ++iy) {
        double rho_L = 0.0, rho_R = 0.0;
        for (int ix = 0; ix < npts; ++ix) {
            int idx = iy * npts + ix;
            rho_L += basis.l_L[ix] * cell.U[0 * dofs + idx];
            rho_R += basis.l_R[ix] * cell.U[0 * dofs + idx];
        }
        rho_min = std::min(rho_min, std::min(rho_L, rho_R));
    }

    // Check Bottom/Top face points
    for (int ix = 0; ix < npts; ++ix) {
        double rho_B = 0.0, rho_T = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            int idx = iy * npts + ix;
            rho_B += basis.l_L[iy] * cell.U[0 * dofs + idx];
            rho_T += basis.l_R[iy] * cell.U[0 * dofs + idx];
        }
        rho_min = std::min(rho_min, std::min(rho_B, rho_T));
    }

    double theta_1 = 1.0;
    if (rho_min < eps) {
        double denom = U_avg[0] - rho_min;
        theta_1 = (denom > 1e-14) ? (U_avg[0] - eps) / denom : 0.0;
    }
    theta_1 = std::clamp(theta_1, 0.0, 1.0);

    if (theta_1 < 1.0) {
        for (int v = 0; v < 4; ++v) {
            for (int idx = 0; idx < dofs; ++idx) {
                cell.U[v * dofs + idx] = U_avg[v] + theta_1 * (cell.U[v * dofs + idx] - U_avg[v]);
            }
        }
    }

    // 2. Check minimum pressure
    auto compute_pressure = [gamma](double rho, double rhou, double rhov, double E) {
        double r = std::max(rho, 1e-12);
        double u = rhou / r;
        double v = rhov / r;
        return (gamma - 1.0) * (E - 0.5 * r * (u * u + v * v));
    };

    double P_avg = compute_pressure(U_avg[0], U_avg[1], U_avg[2], U_avg[3]);
    double P_min = P_avg;

    // Check interior nodes
    for (int idx = 0; idx < dofs; ++idx) {
        double P_node = compute_pressure(cell.U[0 * dofs + idx], cell.U[1 * dofs + idx],
                                         cell.U[2 * dofs + idx], cell.U[3 * dofs + idx]);
        P_min = std::min(P_min, P_node);
    }

    // Check Left/Right face points
    for (int iy = 0; iy < npts; ++iy) {
        double UL[4] = {0.0}, UR[4] = {0.0};
        for (int ix = 0; ix < npts; ++ix) {
            int idx = iy * npts + ix;
            for (int v = 0; v < 4; ++v) {
                UL[v] += basis.l_L[ix] * cell.U[v * dofs + idx];
                UR[v] += basis.l_R[ix] * cell.U[v * dofs + idx];
            }
        }
        P_min = std::min(P_min, compute_pressure(UL[0], UL[1], UL[2], UL[3]));
        P_min = std::min(P_min, compute_pressure(UR[0], UR[1], UR[2], UR[3]));
    }

    // Check Bottom/Top face points
    for (int ix = 0; ix < npts; ++ix) {
        double UB[4] = {0.0}, UT[4] = {0.0};
        for (int iy = 0; iy < npts; ++iy) {
            int idx = iy * npts + ix;
            for (int v = 0; v < 4; ++v) {
                UB[v] += basis.l_L[iy] * cell.U[v * dofs + idx];
                UT[v] += basis.l_R[iy] * cell.U[v * dofs + idx];
            }
        }
        P_min = std::min(P_min, compute_pressure(UB[0], UB[1], UB[2], UB[3]));
        P_min = std::min(P_min, compute_pressure(UT[0], UT[1], UT[2], UT[3]));
    }

    double theta_2 = 1.0;
    if (P_min < eps) {
        double denom = P_avg - P_min;
        theta_2 = (denom > 1e-14) ? (P_avg - eps) / denom : 0.0;
    }
    theta_2 = std::clamp(theta_2, 0.0, 1.0);

    if (theta_2 < 1.0) {
        for (int v = 0; v < 4; ++v) {
            for (int idx = 0; idx < dofs; ++idx) {
                cell.U[v * dofs + idx] = U_avg[v] + theta_2 * (cell.U[v * dofs + idx] - U_avg[v]);
            }
        }
    }
}

void apply_full_limiter_2d(CellDim<2>& cell, const Basis& basis, double gamma, double eps) {
    apply_smooth_lmep_filter_2d(cell, basis, gamma);
    apply_zhang_shu_limiter_2d(cell, basis, gamma, eps);
}

} // namespace Limiters

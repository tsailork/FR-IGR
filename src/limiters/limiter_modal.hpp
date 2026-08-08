/**
 * @file limiter_modal.hpp
 * @brief Implementation of Dzanic / Persson-Peraire Smooth Legendre Modal Filtering Limiter.
 */

#pragma once
#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"
#include "limiter_common.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace Limiters {

/**
 * @brief Evaluate Legendre polynomial P_k(x) for k in [0, 7] on x in [-1, 1].
 */
inline double legendre_P(int k, double x) {
    if (k == 0) return 1.0;
    if (k == 1) return x;
    double p0 = 1.0, p1 = x, p2 = 0.0;
    for (int i = 2; i <= k; ++i) {
        p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / i;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

/**
 * @brief Apply Dzanic smooth scale-aware Legendre modal filter for positivity (with PPR support).
 */
inline bool apply_modal_positivity(Cell& c, const Basis& basis, const Parameters& p) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS) return false;
    const double eps = p.POS_LIMITER_EPS;
    const int num_vars = p.ENABLE_PPR ? 5 : 4;

    // Extrapolate face values for boundary checking
    double face_pts[MAX_FACE_PTS][4];
    double face_S[MAX_FACE_PTS];
    int n_face = p.ENABLE_PPR ? extrapolate_face_values_ppr(c, basis, face_pts, face_S, npts)
                              : extrapolate_face_values(c, basis, face_pts, npts);

    bool positivity_violated = false;
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            if (p.ENABLE_IB && !c.ib_mask.empty() && c.ib_mask[iy * npts + ix] >= 0.5) continue;
            double r  = c.get_U(0, iy, ix, npts);
            double ru = c.get_U(1, iy, ix, npts);
            double rv = c.get_U(2, iy, ix, npts);
            double E  = c.get_U(3, iy, ix, npts);
            double S  = p.ENABLE_PPR ? c.S_field[iy * npts + ix] : 0.0;
            if (!check_positivity_valid(r, ru, rv, E, S, c.theta_avg, p.GAMMA, eps, p.ENABLE_PPR)) {
                positivity_violated = true;
                break;
            }
        }
        if (positivity_violated) break;
    }
    if (!positivity_violated) {
        for (int f = 0; f < n_face; ++f) {
            double r  = face_pts[f][0];
            double ru = face_pts[f][1];
            double rv = face_pts[f][2];
            double E  = face_pts[f][3];
            double S  = p.ENABLE_PPR ? face_S[f] : 0.0;
            if (!check_positivity_valid(r, ru, rv, E, S, c.theta_avg, p.GAMMA, eps, p.ENABLE_PPR)) {
                positivity_violated = true;
                break;
            }
        }
    }
    if (!positivity_violated) return false;

    // Pre-compute 1D transformation matrices on GL nodes
    double T_modal[MAX_LIM_PTS][MAX_LIM_PTS];
    double T_nodal[MAX_LIM_PTS][MAX_LIM_PTS];
    for (int k = 0; k < npts; ++k) {
        double norm = (2.0 * k + 1.0) / 2.0;
        for (int i = 0; i < npts; ++i) {
            T_modal[k][i] = norm * basis.w[i] * legendre_P(k, basis.z[i]);
            T_nodal[i][k] = legendre_P(k, basis.z[i]);
        }
    }

    // Compute modal representation U_modal[var][ky][kx]
    double U_modal[5][MAX_LIM_PTS][MAX_LIM_PTS];
    for (int v = 0; v < num_vars; ++v) {
        double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int iy = 0; iy < npts; ++iy) {
            for (int kx = 0; kx < npts; ++kx) {
                tmp[iy][kx] = 0.0;
                for (int ix = 0; ix < npts; ++ix) {
                    double val = (v < 4) ? c.get_U(v, iy, ix, npts) : c.S_field[iy * npts + ix];
                    tmp[iy][kx] += T_modal[kx][ix] * val;
                }
            }
        }
        for (int ky = 0; ky < npts; ++ky) {
            for (int kx = 0; kx < npts; ++kx) {
                U_modal[v][ky][kx] = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    U_modal[v][ky][kx] += T_modal[ky][iy] * tmp[iy][kx];
                }
            }
        }
    }

    const int P = npts - 1;

    // Ensure cell average itself is strictly positive
    double r_avg = U_modal[0][0][0];
    double ru_avg = U_modal[1][0][0];
    double rv_avg = U_modal[2][0][0];
    double E_avg = U_modal[3][0][0];
    if (r_avg < eps) {
        r_avg = eps;
        U_modal[0][0][0] = r_avg;
    }
    double ke_avg = 0.5 * (ru_avg * ru_avg + rv_avg * rv_avg) / r_avg;
    double p_avg = (p.GAMMA - 1.0) * (E_avg - ke_avg);
    if (p_avg < eps) {
        E_avg = eps / (p.GAMMA - 1.0) + ke_avg;
        U_modal[3][0][0] = E_avg;
    }

    // Reconstruct nodal solution with scale-aware modal damping factor theta in [0, 1]
    auto reconstruct_with_theta = [&](double theta, double U_out[5][MAX_LIM_PTS][MAX_LIM_PTS]) {
        for (int v = 0; v < num_vars; ++v) {
            double U_filt_modal[MAX_LIM_PTS][MAX_LIM_PTS];
            for (int ky = 0; ky < npts; ++ky) {
                for (int kx = 0; kx < npts; ++kx) {
                    if (kx == 0 && ky == 0) {
                        U_filt_modal[ky][kx] = U_modal[v][ky][kx];
                    } else {
                        double eta_x = (P > 0) ? (double)kx / P : 0.0;
                        double eta_y = (P > 0) ? (double)ky / P : 0.0;
                        double power = std::pow(eta_x, 2.0) + std::pow(eta_y, 2.0);
                        double sigma = (theta <= 0.0) ? 0.0 : std::pow(theta, power);
                        U_filt_modal[ky][kx] = sigma * U_modal[v][ky][kx];
                    }
                }
            }
            double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
            for (int iy = 0; iy < npts; ++iy) {
                for (int kx = 0; kx < npts; ++kx) {
                    tmp[iy][kx] = 0.0;
                    for (int ky = 0; ky < npts; ++ky) {
                        tmp[iy][kx] += T_nodal[iy][ky] * U_filt_modal[ky][kx];
                    }
                }
            }
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    U_out[v][iy][ix] = 0.0;
                    for (int kx = 0; kx < npts; ++kx) {
                        U_out[v][iy][ix] += T_nodal[ix][kx] * tmp[iy][kx];
                    }
                }
            }
        }
    };

    double lo = 0.0, hi = 1.0;
    double U_candidate[5][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_faces[MAX_FACE_PTS][4];
    double candidate_S[MAX_FACE_PTS];

    for (int iter = 0; iter < 15; ++iter) {
        double mid = 0.5 * (lo + hi);
        reconstruct_with_theta(mid, U_candidate);
        int n_cand_faces = p.ENABLE_PPR ? extrapolate_face_values_ppr_array(U_candidate, basis, candidate_faces, candidate_S, npts)
                                        : extrapolate_face_values_array(U_candidate, basis, candidate_faces, npts);

        bool valid = true;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double r  = U_candidate[0][iy][ix];
                double ru = U_candidate[1][iy][ix];
                double rv = U_candidate[2][iy][ix];
                double E  = U_candidate[3][iy][ix];
                double S  = p.ENABLE_PPR ? U_candidate[4][iy][ix] : 0.0;
                if (!check_positivity_valid(r, ru, rv, E, S, c.theta_avg, p.GAMMA, eps, p.ENABLE_PPR)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }

        if (valid) {
            for (int f = 0; f < n_cand_faces; ++f) {
                double r  = candidate_faces[f][0];
                double ru = candidate_faces[f][1];
                double rv = candidate_faces[f][2];
                double E  = candidate_faces[f][3];
                double S  = p.ENABLE_PPR ? candidate_S[f] : 0.0;
                if (!check_positivity_valid(r, ru, rv, E, S, c.theta_avg, p.GAMMA, eps, p.ENABLE_PPR)) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    reconstruct_with_theta(lo, U_candidate);

    // Apply exact L2 cell average conservation projection to eliminate floating-point drift
    double orig_r_avg, orig_ru_avg, orig_rv_avg, orig_E_avg;
    Limiters::compute_cell_average(c, basis, orig_r_avg, orig_ru_avg, orig_rv_avg, orig_E_avg, npts);
    double S_avg = 0.0;
    if (p.ENABLE_PPR) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double w = 0.25 * basis.w[iy] * basis.w[ix];
                S_avg += w * c.S_field[iy * npts + ix];
            }
        }
    }
    double U_avg[5] = { r_avg, ru_avg, rv_avg, E_avg, S_avg };

    for (int v = 0; v < num_vars; ++v) {
        double calc_avg = 0.0;
        double sum_w = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double w = 0.25 * basis.w[iy] * basis.w[ix];
                sum_w += w;
                calc_avg += w * U_candidate[v][iy][ix];
            }
        }
        double delta = calc_avg - U_avg[v];
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double val = U_candidate[v][iy][ix] - delta / sum_w;
                if (v < 4) c.get_U(v, iy, ix, npts) = val;
                else c.S_field[iy * npts + ix] = val;
            }
        }
    }

    return true;
}

/**
 * @brief Apply Dzanic smooth scale-aware Legendre modal filter to cell conservative variables for entropy minimum preservation.
 */
inline bool apply_modal_entropy(Cell& c, double s_floor, const Basis& basis, const Parameters& p) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS) return false;

    double face_pts[MAX_FACE_PTS][4];
    int n_face = extrapolate_face_values(c, basis, face_pts, npts);

    bool entropy_violated = false;
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            double s = Limiters::specific_entropy(
                c.get_U(0, iy, ix, npts), c.get_U(1, iy, ix, npts),
                c.get_U(2, iy, ix, npts), c.get_U(3, iy, ix, npts), p.GAMMA);
            if (s < s_floor) {
                entropy_violated = true;
                break;
            }
        }
        if (entropy_violated) break;
    }
    if (!entropy_violated) {
        for (int f = 0; f < n_face; ++f) {
            double s = Limiters::specific_entropy(
                face_pts[f][0], face_pts[f][1], face_pts[f][2], face_pts[f][3], p.GAMMA);
            if (s < s_floor) {
                entropy_violated = true;
                break;
            }
        }
    }
    if (!entropy_violated) return false;

    double T_modal[MAX_LIM_PTS][MAX_LIM_PTS], T_nodal[MAX_LIM_PTS][MAX_LIM_PTS];
    for (int k = 0; k < npts; ++k) {
        double norm = (2.0 * k + 1.0) / 2.0;
        for (int i = 0; i < npts; ++i) {
            T_modal[k][i] = norm * basis.w[i] * legendre_P(k, basis.z[i]);
            T_nodal[i][k] = legendre_P(k, basis.z[i]);
        }
    }

    double U_modal[4][MAX_LIM_PTS][MAX_LIM_PTS];
    for (int v = 0; v < 4; ++v) {
        double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int iy = 0; iy < npts; ++iy) {
            for (int kx = 0; kx < npts; ++kx) {
                tmp[iy][kx] = 0.0;
                for (int ix = 0; ix < npts; ++ix) {
                    tmp[iy][kx] += T_modal[kx][ix] * c.get_U(v, iy, ix, npts);
                }
            }
        }
        for (int ky = 0; ky < npts; ++ky) {
            for (int kx = 0; kx < npts; ++kx) {
                U_modal[v][ky][kx] = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    U_modal[v][ky][kx] += T_modal[ky][iy] * tmp[iy][kx];
                }
            }
        }
    }

    const int P = npts - 1;

    auto reconstruct_with_theta = [&](double theta, double U_out[4][MAX_LIM_PTS][MAX_LIM_PTS]) {
        for (int v = 0; v < 4; ++v) {
            double U_filt_modal[MAX_LIM_PTS][MAX_LIM_PTS];
            for (int ky = 0; ky < npts; ++ky) {
                for (int kx = 0; kx < npts; ++kx) {
                    if (kx == 0 && ky == 0) {
                        U_filt_modal[ky][kx] = U_modal[v][ky][kx];
                    } else {
                        double eta_x = (P > 0) ? (double)kx / P : 0.0;
                        double eta_y = (P > 0) ? (double)ky / P : 0.0;
                        double power = std::pow(eta_x, 2.0) + std::pow(eta_y, 2.0);
                        double sigma = (theta <= 0.0) ? 0.0 : std::pow(theta, power);
                        U_filt_modal[ky][kx] = sigma * U_modal[v][ky][kx];
                    }
                }
            }
            double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
            for (int iy = 0; iy < npts; ++iy) {
                for (int kx = 0; kx < npts; ++kx) {
                    tmp[iy][kx] = 0.0;
                    for (int ky = 0; ky < npts; ++ky) {
                        tmp[iy][kx] += T_nodal[iy][ky] * U_filt_modal[ky][kx];
                    }
                }
            }
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    U_out[v][iy][ix] = 0.0;
                    for (int kx = 0; kx < npts; ++kx) {
                        U_out[v][iy][ix] += T_nodal[ix][kx] * tmp[iy][kx];
                    }
                }
            }
        }
    };

    double lo = 0.0, hi = 1.0;
    double U_candidate[4][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_faces[MAX_FACE_PTS][4];

    for (int iter = 0; iter < 15; ++iter) {
        double mid = 0.5 * (lo + hi);
        reconstruct_with_theta(mid, U_candidate);
        int n_cand_faces = extrapolate_face_values_array(U_candidate, basis, candidate_faces, npts);

        bool valid = true;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double s = Limiters::specific_entropy(
                    U_candidate[0][iy][ix], U_candidate[1][iy][ix],
                    U_candidate[2][iy][ix], U_candidate[3][iy][ix], p.GAMMA);
                if (s < s_floor) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }

        if (valid) {
            for (int f = 0; f < n_cand_faces; ++f) {
                double s = Limiters::specific_entropy(
                    candidate_faces[f][0], candidate_faces[f][1],
                    candidate_faces[f][2], candidate_faces[f][3], p.GAMMA);
                if (s < s_floor) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    reconstruct_with_theta(lo, U_candidate);

    for (int v = 0; v < 4; ++v) {
        double r_avg, ru_avg, rv_avg, E_avg;
        Limiters::compute_cell_average(c, basis, r_avg, ru_avg, rv_avg, E_avg, npts);
        double U_avg[4] = { r_avg, ru_avg, rv_avg, E_avg };

        double calc_avg = 0.0;
        double sum_w = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double w = 0.25 * basis.w[iy] * basis.w[ix];
                sum_w += w;
                calc_avg += w * U_candidate[v][iy][ix];
            }
        }
        double delta = calc_avg - U_avg[v];
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                c.get_U(v, iy, ix, npts) = U_candidate[v][iy][ix] - delta / sum_w;
            }
        }
    }

    return true;
}

/**
 * @brief Apply direct Legendre modal damping to cell conservative variables for IB boundary stabilization.
 * Preserves exact cell averages (zero-order modes) while damping high-order aliasing modes.
 */
inline bool apply_ib_modal_filter(Cell& c, const Basis& basis, const Parameters& p, double damping_factor = 0.5) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS || npts <= 1) return false;

    double T_modal[MAX_LIM_PTS][MAX_LIM_PTS], T_nodal[MAX_LIM_PTS][MAX_LIM_PTS];
    for (int k = 0; k < npts; ++k) {
        double norm = (2.0 * k + 1.0) / 2.0;
        for (int i = 0; i < npts; ++i) {
            T_modal[k][i] = norm * basis.w[i] * legendre_P(k, basis.z[i]);
            T_nodal[i][k] = legendre_P(k, basis.z[i]);
        }
    }

    double U_modal[4][MAX_LIM_PTS][MAX_LIM_PTS];
    for (int v = 0; v < 4; ++v) {
        double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int iy = 0; iy < npts; ++iy) {
            for (int kx = 0; kx < npts; ++kx) {
                tmp[iy][kx] = 0.0;
                for (int ix = 0; ix < npts; ++ix) {
                    tmp[iy][kx] += T_modal[kx][ix] * c.get_U(v, iy, ix, npts);
                }
            }
        }
        for (int ky = 0; ky < npts; ++ky) {
            for (int kx = 0; kx < npts; ++kx) {
                U_modal[v][ky][kx] = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    U_modal[v][ky][kx] += T_modal[ky][iy] * tmp[iy][kx];
                }
            }
        }
    }

    const int P = npts - 1;
    for (int v = 0; v < 4; ++v) {
        double U_filt_modal[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int ky = 0; ky < npts; ++ky) {
            for (int kx = 0; kx < npts; ++kx) {
                if (kx == 0 && ky == 0) {
                    U_filt_modal[ky][kx] = U_modal[v][ky][kx];
                } else {
                    double eta_x = (P > 0) ? (double)kx / P : 0.0;
                    double eta_y = (P > 0) ? (double)ky / P : 0.0;
                    double power = std::pow(eta_x, 2.0) + std::pow(eta_y, 2.0);
                    double sigma = std::pow(damping_factor, power);
                    U_filt_modal[ky][kx] = sigma * U_modal[v][ky][kx];
                }
            }
        }
        double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int iy = 0; iy < npts; ++iy) {
            for (int kx = 0; kx < npts; ++kx) {
                tmp[iy][kx] = 0.0;
                for (int ky = 0; ky < npts; ++ky) {
                    tmp[iy][kx] += T_nodal[iy][ky] * U_filt_modal[ky][kx];
                }
            }
        }
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double val = 0.0;
                for (int kx = 0; kx < npts; ++kx) {
                    val += T_nodal[ix][kx] * tmp[iy][kx];
                }
                c.get_U(v, iy, ix, npts) = val;
            }
        }
    }

    return true;
}

} // namespace Limiters

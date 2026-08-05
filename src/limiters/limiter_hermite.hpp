/**
 * @file limiter_hermite.hpp
 * @brief Implementation of Hermite Patch Subgrid Limiter with Neighbor Face Value Coupling and PPR Support.
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

inline double hermite_H_L(double z) {
    return 0.25 * (1.0 - z) * (1.0 - z) * (2.0 + z) - 0.5;
}

inline double hermite_H_R(double z) {
    return 0.25 * (1.0 + z) * (1.0 + z) * (2.0 - z) - 0.5;
}

/**
 * @brief Apply Hermite patch positivity limiter with face value coupling and PPR support.
 */
inline bool apply_hermite_positivity(Cell& c, const Basis& basis, const Parameters& p) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS) return false;
    const double eps = p.POS_LIMITER_EPS;
    const int num_vars = p.ENABLE_PPR ? 5 : 4;

    // Extrapolate face values for boundary checking
    double face_pts[MAX_FACE_PTS][4];
    double face_S[MAX_FACE_PTS];
    int n_face = p.ENABLE_PPR ? extrapolate_face_values_ppr(c, basis, face_pts, face_S, npts)
                              : extrapolate_face_values(c, basis, face_pts, npts);

    bool violated = false;
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            double r  = c.get_U(0, iy, ix, npts);
            double ru = c.get_U(1, iy, ix, npts);
            double rv = c.get_U(2, iy, ix, npts);
            double E  = c.get_U(3, iy, ix, npts);
            double S  = p.ENABLE_PPR ? c.S_field[iy * npts + ix] : 0.0;
            if (!check_positivity_valid(r, ru, rv, E, S, c.theta_avg, p.GAMMA, eps, p.ENABLE_PPR)) {
                violated = true;
                break;
            }
        }
        if (violated) break;
    }
    if (!violated) {
        for (int f = 0; f < n_face; ++f) {
            double r  = face_pts[f][0];
            double ru = face_pts[f][1];
            double rv = face_pts[f][2];
            double E  = face_pts[f][3];
            double S  = p.ENABLE_PPR ? face_S[f] : 0.0;
            if (!check_positivity_valid(r, ru, rv, E, S, c.theta_avg, p.GAMMA, eps, p.ENABLE_PPR)) {
                violated = true;
                break;
            }
        }
    }
    if (!violated) return false;

    // Compute cell averages U_avg
    double r_avg, ru_avg, rv_avg, E_avg;
    compute_cell_average(c, basis, r_avg, ru_avg, rv_avg, E_avg, npts);
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

    // Determine target neighbor face values
    double U_L[5], U_R[5], U_B[5], U_T[5];
    for (int v = 0; v < num_vars; ++v) {
        U_L[v] = U_avg[v];
        U_R[v] = U_avg[v];
        U_B[v] = U_avg[v];
        U_T[v] = U_avg[v];
    }

    if (c.neighbors[0]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[0]), basis, r2, ru2, rv2, E2, npts);
        U_L[0] = 0.5 * (U_avg[0] + r2);
        U_L[1] = 0.5 * (U_avg[1] + ru2);
        U_L[2] = 0.5 * (U_avg[2] + rv2);
        U_L[3] = 0.5 * (U_avg[3] + E2);
    }
    if (c.neighbors[1]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[1]), basis, r2, ru2, rv2, E2, npts);
        U_R[0] = 0.5 * (U_avg[0] + r2);
        U_R[1] = 0.5 * (U_avg[1] + ru2);
        U_R[2] = 0.5 * (U_avg[2] + rv2);
        U_R[3] = 0.5 * (U_avg[3] + E2);
    }
    if (c.neighbors[2]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[2]), basis, r2, ru2, rv2, E2, npts);
        U_B[0] = 0.5 * (U_avg[0] + r2);
        U_B[1] = 0.5 * (U_avg[1] + ru2);
        U_B[2] = 0.5 * (U_avg[2] + rv2);
        U_B[3] = 0.5 * (U_avg[3] + E2);
    }
    if (c.neighbors[3]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[3]), basis, r2, ru2, rv2, E2, npts);
        U_T[0] = 0.5 * (U_avg[0] + r2);
        U_T[1] = 0.5 * (U_avg[1] + ru2);
        U_T[2] = 0.5 * (U_avg[2] + rv2);
        U_T[3] = 0.5 * (U_avg[3] + E2);
    }

    double U_patch[5][MAX_LIM_PTS][MAX_LIM_PTS];

    for (int v = 0; v < num_vars; ++v) {
        for (int iy = 0; iy < npts; ++iy) {
            double y = basis.z[iy];
            double hB = hermite_H_L(y);
            double hT = hermite_H_R(y);
            for (int ix = 0; ix < npts; ++ix) {
                double x = basis.z[ix];
                double hL = hermite_H_L(x);
                double hR = hermite_H_R(x);

                U_patch[v][iy][ix] = U_avg[v]
                                   + hL * (U_L[v] - U_avg[v])
                                   + hR * (U_R[v] - U_avg[v])
                                   + hB * (U_B[v] - U_avg[v])
                                   + hT * (U_T[v] - U_avg[v]);
            }
        }
        // Enforce L2 cell average conservation on U_patch
        double calc_avg = 0.0, sum_w = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double w = 0.25 * basis.w[iy] * basis.w[ix];
                sum_w += w;
                calc_avg += w * U_patch[v][iy][ix];
            }
        }
        double delta = calc_avg - U_avg[v];
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                U_patch[v][iy][ix] -= delta / sum_w;
            }
        }
    }

    double lo = 0.0, hi = 1.0;
    double U_cand[5][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_faces[MAX_FACE_PTS][4];
    double candidate_S[MAX_FACE_PTS];

    // Bisection search on theta in [0, 1] towards exact P0 cell average limit (theta = 0)
    for (int iter = 0; iter < 15; ++iter) {
        double mid = 0.5 * (lo + hi);
        for (int v = 0; v < num_vars; ++v) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    U_cand[v][iy][ix] = mid * U_patch[v][iy][ix] + (1.0 - mid) * U_avg[v];
                }
            }
        }

        int n_cand_faces = p.ENABLE_PPR ? extrapolate_face_values_ppr_array(U_cand, basis, candidate_faces, candidate_S, npts)
                                        : extrapolate_face_values_array(U_cand, basis, candidate_faces, npts);

        bool valid = true;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double r  = U_cand[0][iy][ix];
                double ru = U_cand[1][iy][ix];
                double rv = U_cand[2][iy][ix];
                double E  = U_cand[3][iy][ix];
                double S  = p.ENABLE_PPR ? U_cand[4][iy][ix] : 0.0;
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

    for (int v = 0; v < num_vars; ++v) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double val = lo * U_patch[v][iy][ix] + (1.0 - lo) * U_avg[v];
                if (v < 4) c.get_U(v, iy, ix, npts) = val;
                else c.S_field[iy * npts + ix] = val;
            }
        }
    }

    return true;
}

/**
 * @brief Apply Hermite patch limiter for entropy minimum preservation.
 */
inline bool apply_hermite_entropy(Cell& c, double s_floor, const Basis& basis, const Parameters& p) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS) return false;

    double face_pts[MAX_FACE_PTS][4];
    int n_face = extrapolate_face_values(c, basis, face_pts, npts);

    bool violated = false;
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            double s = specific_entropy(
                c.get_U(0, iy, ix, npts), c.get_U(1, iy, ix, npts),
                c.get_U(2, iy, ix, npts), c.get_U(3, iy, ix, npts), p.GAMMA);
            if (s < s_floor) {
                violated = true;
                break;
            }
        }
        if (violated) break;
    }
    if (!violated) {
        for (int f = 0; f < n_face; ++f) {
            double s = specific_entropy(
                face_pts[f][0], face_pts[f][1], face_pts[f][2], face_pts[f][3], p.GAMMA);
            if (s < s_floor) {
                violated = true;
                break;
            }
        }
    }
    if (!violated) return false;

    double r_avg, ru_avg, rv_avg, E_avg;
    compute_cell_average(c, basis, r_avg, ru_avg, rv_avg, E_avg, npts);
    double U_avg[4] = { r_avg, ru_avg, rv_avg, E_avg };

    double U_L[4] = { r_avg, ru_avg, rv_avg, E_avg };
    double U_R[4] = { r_avg, ru_avg, rv_avg, E_avg };
    double U_B[4] = { r_avg, ru_avg, rv_avg, E_avg };
    double U_T[4] = { r_avg, ru_avg, rv_avg, E_avg };

    if (c.neighbors[0]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[0]), basis, r2, ru2, rv2, E2, npts);
        U_L[0] = 0.5 * (r_avg + r2); U_L[1] = 0.5 * (ru_avg + ru2); U_L[2] = 0.5 * (rv_avg + rv2); U_L[3] = 0.5 * (E_avg + E2);
    }
    if (c.neighbors[1]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[1]), basis, r2, ru2, rv2, E2, npts);
        U_R[0] = 0.5 * (r_avg + r2); U_R[1] = 0.5 * (ru_avg + ru2); U_R[2] = 0.5 * (rv_avg + rv2); U_R[3] = 0.5 * (E_avg + E2);
    }
    if (c.neighbors[2]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[2]), basis, r2, ru2, rv2, E2, npts);
        U_B[0] = 0.5 * (r_avg + r2); U_B[1] = 0.5 * (ru_avg + ru2); U_B[2] = 0.5 * (rv_avg + rv2); U_B[3] = 0.5 * (E_avg + E2);
    }
    if (c.neighbors[3]) {
        double r2, ru2, rv2, E2;
        compute_cell_average(*(c.neighbors[3]), basis, r2, ru2, rv2, E2, npts);
        U_T[0] = 0.5 * (r_avg + r2); U_T[1] = 0.5 * (ru_avg + ru2); U_T[2] = 0.5 * (rv_avg + rv2); U_T[3] = 0.5 * (E_avg + E2);
    }

    double U_patch[4][MAX_LIM_PTS][MAX_LIM_PTS];

    for (int v = 0; v < 4; ++v) {
        for (int iy = 0; iy < npts; ++iy) {
            double y = basis.z[iy];
            double hB = hermite_H_L(y);
            double hT = hermite_H_R(y);
            for (int ix = 0; ix < npts; ++ix) {
                double x = basis.z[ix];
                double hL = hermite_H_L(x);
                double hR = hermite_H_R(x);

                U_patch[v][iy][ix] = U_avg[v]
                                   + hL * (U_L[v] - U_avg[v])
                                   + hR * (U_R[v] - U_avg[v])
                                   + hB * (U_B[v] - U_avg[v])
                                   + hT * (U_T[v] - U_avg[v]);
            }
        }
        double calc_avg = 0.0, sum_w = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double w = 0.25 * basis.w[iy] * basis.w[ix];
                sum_w += w;
                calc_avg += w * U_patch[v][iy][ix];
            }
        }
        double delta = calc_avg - U_avg[v];
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                U_patch[v][iy][ix] -= delta / sum_w;
            }
        }
    }

    double lo = 0.0, hi = 1.0;
    double U_cand[4][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_faces[MAX_FACE_PTS][4];

    for (int iter = 0; iter < 15; ++iter) {
        double mid = 0.5 * (lo + hi);
        for (int v = 0; v < 4; ++v) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    U_cand[v][iy][ix] = mid * U_patch[v][iy][ix] + (1.0 - mid) * U_avg[v];
                }
            }
        }

        int n_cand_faces = extrapolate_face_values_array(U_cand, basis, candidate_faces, npts);

        bool valid = true;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double s = specific_entropy(
                    U_cand[0][iy][ix], U_cand[1][iy][ix],
                    U_cand[2][iy][ix], U_cand[3][iy][ix], p.GAMMA);
                if (s < s_floor) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }

        if (valid) {
            for (int f = 0; f < n_cand_faces; ++f) {
                double s = specific_entropy(
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

    for (int v = 0; v < 4; ++v) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                c.get_U(v, iy, ix, npts) = lo * U_patch[v][iy][ix] + (1.0 - lo) * U_avg[v];
            }
        }
    }

    return true;
}

} // namespace Limiters

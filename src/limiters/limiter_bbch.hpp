/**
 * @file limiter_bbch.hpp
 * @brief Implementation of Bernstein-Bézier Convex Hull (BBCH) Limiter with Exact Cell Average Conservation and PPR Support.
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
 * @brief Helper function to compute n choose k.
 */
inline double n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    double res = 1.0;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - k + i) / i;
    }
    return res;
}

/**
 * @brief Evaluate 1D Bernstein polynomial B_{i, P}(t) for t in [0, 1].
 */
inline double bernstein_B(int i, int P, double t) {
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return n_choose_k(P, i) * std::pow(t, i) * std::pow(1.0 - t, P - i);
}

/**
 * @brief Invert a square matrix A of size N x N in-place.
 */
inline bool invert_matrix(int N, double A[][8], double Ainv[][8]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Ainv[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    double M[8][8];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M[i][j] = A[i][j];

    for (int i = 0; i < N; ++i) {
        double pivot = M[i][i];
        int pivot_row = i;
        for (int k = i + 1; k < N; ++k) {
            if (std::abs(M[k][i]) > std::abs(pivot)) {
                pivot = M[k][i];
                pivot_row = k;
            }
        }
        if (std::abs(pivot) < 1e-12) return false;
        if (pivot_row != i) {
            for (int j = 0; j < N; ++j) {
                std::swap(M[i][j], M[pivot_row][j]);
                std::swap(Ainv[i][j], Ainv[pivot_row][j]);
            }
        }
        pivot = M[i][i];
        for (int j = 0; j < N; ++j) {
            M[i][j] /= pivot;
            Ainv[i][j] /= pivot;
        }
        for (int k = 0; k < N; ++k) {
            if (k != i) {
                double factor = M[k][i];
                for (int j = 0; j < N; ++j) {
                    M[k][j] -= factor * M[i][j];
                    Ainv[k][j] -= factor * Ainv[i][j];
                }
            }
        }
    }
    return true;
}

/**
 * @brief Apply Bernstein-Bézier Convex Hull (BBCH) limiter for positivity with exact average conservation and PPR support.
 */
inline bool apply_bbch_positivity(Cell& c, const Basis& basis, const Parameters& p) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS) return false;
    const double eps = p.POS_LIMITER_EPS;
    const int P = npts - 1;
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

    // Build M_BB_to_nodal matrix: [j][i] = B_{i, P}( (x_j + 1)/2 )
    double M_BB2Nodal[MAX_LIM_PTS][MAX_LIM_PTS];
    for (int j = 0; j < npts; ++j) {
        double t = 0.5 * (basis.z[j] + 1.0);
        for (int i = 0; i < npts; ++i) {
            M_BB2Nodal[j][i] = bernstein_B(i, P, t);
        }
    }

    double M_Nodal2BB[MAX_LIM_PTS][MAX_LIM_PTS];
    if (!invert_matrix(npts, M_BB2Nodal, M_Nodal2BB)) return false;

    // Compute cell averages U_avg
    double r_avg, ru_avg, rv_avg, E_avg;
    Limiters::compute_cell_average(c, basis, r_avg, ru_avg, rv_avg, E_avg, npts);
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

    // Compute BB weights for integration: w_BB[i][j]
    double w_BB[MAX_LIM_PTS][MAX_LIM_PTS];
    double sum_w_sq = 0.0;
    for (int i = 0; i < npts; ++i) {
        for (int j = 0; j < npts; ++j) {
            w_BB[i][j] = 0.0;
            for (int my = 0; my < npts; ++my) {
                for (int mx = 0; mx < npts; ++mx) {
                    double weight = 0.25 * basis.w[my] * basis.w[mx];
                    w_BB[i][j] += weight * M_BB2Nodal[my][j] * M_BB2Nodal[mx][i];
                }
            }
            sum_w_sq += w_BB[i][j] * w_BB[i][j];
        }
    }

    // Transform nodal U to BB control points C[var][j][i]
    double C[5][MAX_LIM_PTS][MAX_LIM_PTS];
    for (int v = 0; v < num_vars; ++v) {
        double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int iy = 0; iy < npts; ++iy) {
            for (int i = 0; i < npts; ++i) {
                tmp[iy][i] = 0.0;
                for (int ix = 0; ix < npts; ++ix) {
                    double val = (v < 4) ? c.get_U(v, iy, ix, npts) : c.S_field[iy * npts + ix];
                    tmp[iy][i] += M_Nodal2BB[i][ix] * val;
                }
            }
        }
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                C[v][j][i] = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    C[v][j][i] += M_Nodal2BB[j][iy] * tmp[iy][i];
                }
            }
        }
    }

    double lo = 0.0, hi = 1.0;
    double C_candidate[5][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_nodal[5][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_faces[MAX_FACE_PTS][4];
    double candidate_S[MAX_FACE_PTS];

    for (int iter = 0; iter < 15; ++iter) {
        double mid = 0.5 * (lo + hi);
        for (int v = 0; v < num_vars; ++v) {
            for (int j = 0; j < npts; ++j) {
                for (int i = 0; i < npts; ++i) {
                    C_candidate[v][j][i] = mid * C[v][j][i] + (1.0 - mid) * U_avg[v];
                }
            }

            double calc_avg = 0.0;
            for (int j = 0; j < npts; ++j) {
                for (int i = 0; i < npts; ++i) {
                    calc_avg += w_BB[i][j] * C_candidate[v][j][i];
                }
            }
            double delta = calc_avg - U_avg[v];
            for (int j = 0; j < npts; ++j) {
                for (int i = 0; i < npts; ++i) {
                    C_candidate[v][j][i] -= (delta / sum_w_sq) * w_BB[i][j];
                }
            }
        }

        // Convert candidate C back to nodal and check positivity
        for (int v = 0; v < num_vars; ++v) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    candidate_nodal[v][iy][ix] = 0.0;
                    for (int j = 0; j < npts; ++j) {
                        for (int i = 0; i < npts; ++i) {
                            candidate_nodal[v][iy][ix] += C_candidate[v][j][i] * M_BB2Nodal[iy][j] * M_BB2Nodal[ix][i];
                        }
                    }
                }
            }
        }

        int n_cand_faces = p.ENABLE_PPR ? extrapolate_face_values_ppr_array(candidate_nodal, basis, candidate_faces, candidate_S, npts)
                                        : extrapolate_face_values_array(candidate_nodal, basis, candidate_faces, npts);

        bool valid = true;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double r  = candidate_nodal[0][iy][ix];
                double ru = candidate_nodal[1][iy][ix];
                double rv = candidate_nodal[2][iy][ix];
                double E  = candidate_nodal[3][iy][ix];
                double S  = p.ENABLE_PPR ? candidate_nodal[4][iy][ix] : 0.0;
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
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                C_candidate[v][j][i] = lo * C[v][j][i] + (1.0 - lo) * U_avg[v];
            }
        }
        double calc_avg = 0.0;
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                calc_avg += w_BB[i][j] * C_candidate[v][j][i];
            }
        }
        double delta = calc_avg - U_avg[v];
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                C_candidate[v][j][i] -= (delta / sum_w_sq) * w_BB[i][j];
            }
        }

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double val = 0.0;
                for (int j = 0; j < npts; ++j) {
                    for (int i = 0; i < npts; ++i) {
                        val += C_candidate[v][j][i] * M_BB2Nodal[iy][j] * M_BB2Nodal[ix][i];
                    }
                }
                if (v < 4) c.get_U(v, iy, ix, npts) = val;
                else c.S_field[iy * npts + ix] = val;
            }
        }
    }

    return true;
}

/**
 * @brief Apply BBCH limiter for entropy minimum preservation.
 */
inline bool apply_bbch_entropy(Cell& c, double s_floor, const Basis& basis, const Parameters& p) {
    const int npts = p.N_PTS;
    if (npts > MAX_LIM_PTS) return false;
    const int P = npts - 1;

    double face_pts[MAX_FACE_PTS][4];
    int n_face = extrapolate_face_values(c, basis, face_pts, npts);

    bool violated = false;
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            double s = Limiters::specific_entropy(
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
            double s = Limiters::specific_entropy(
                face_pts[f][0], face_pts[f][1], face_pts[f][2], face_pts[f][3], p.GAMMA);
            if (s < s_floor) {
                violated = true;
                break;
            }
        }
    }
    if (!violated) return false;

    double M_BB2Nodal[MAX_LIM_PTS][MAX_LIM_PTS];
    for (int j = 0; j < npts; ++j) {
        double t = 0.5 * (basis.z[j] + 1.0);
        for (int i = 0; i < npts; ++i) {
            M_BB2Nodal[j][i] = bernstein_B(i, P, t);
        }
    }

    double M_Nodal2BB[MAX_LIM_PTS][MAX_LIM_PTS];
    if (!invert_matrix(npts, M_BB2Nodal, M_Nodal2BB)) return false;

    double r_avg, ru_avg, rv_avg, E_avg;
    Limiters::compute_cell_average(c, basis, r_avg, ru_avg, rv_avg, E_avg, npts);
    double U_avg[4] = { r_avg, ru_avg, rv_avg, E_avg };

    double w_BB[MAX_LIM_PTS][MAX_LIM_PTS];
    double sum_w_sq = 0.0;
    for (int i = 0; i < npts; ++i) {
        for (int j = 0; j < npts; ++j) {
            w_BB[i][j] = 0.0;
            for (int my = 0; my < npts; ++my) {
                for (int mx = 0; mx < npts; ++mx) {
                    double weight = 0.25 * basis.w[my] * basis.w[mx];
                    w_BB[i][j] += weight * M_BB2Nodal[my][j] * M_BB2Nodal[mx][i];
                }
            }
            sum_w_sq += w_BB[i][j] * w_BB[i][j];
        }
    }

    double C[4][MAX_LIM_PTS][MAX_LIM_PTS];
    for (int v = 0; v < 4; ++v) {
        double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
        for (int iy = 0; iy < npts; ++iy) {
            for (int i = 0; i < npts; ++i) {
                tmp[iy][i] = 0.0;
                for (int ix = 0; ix < npts; ++ix) {
                    tmp[iy][i] += M_Nodal2BB[i][ix] * c.get_U(v, iy, ix, npts);
                }
            }
        }
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                C[v][j][i] = 0.0;
                for (int iy = 0; iy < npts; ++iy) {
                    C[v][j][i] += M_Nodal2BB[j][iy] * tmp[iy][i];
                }
            }
        }
    }

    double lo = 0.0, hi = 1.0;
    double C_candidate[4][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_nodal[4][MAX_LIM_PTS][MAX_LIM_PTS];
    double candidate_faces[MAX_FACE_PTS][4];

    for (int iter = 0; iter < 15; ++iter) {
        double mid = 0.5 * (lo + hi);
        for (int v = 0; v < 4; ++v) {
            for (int j = 0; j < npts; ++j) {
                for (int i = 0; i < npts; ++i) {
                    C_candidate[v][j][i] = mid * C[v][j][i] + (1.0 - mid) * U_avg[v];
                }
            }

            double calc_avg = 0.0;
            for (int j = 0; j < npts; ++j) {
                for (int i = 0; i < npts; ++i) {
                    calc_avg += w_BB[i][j] * C_candidate[v][j][i];
                }
            }
            double delta = calc_avg - U_avg[v];
            for (int j = 0; j < npts; ++j) {
                for (int i = 0; i < npts; ++i) {
                    C_candidate[v][j][i] -= (delta / sum_w_sq) * w_BB[i][j];
                }
            }
        }

        for (int v = 0; v < 4; ++v) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    candidate_nodal[v][iy][ix] = 0.0;
                    for (int j = 0; j < npts; ++j) {
                        for (int i = 0; i < npts; ++i) {
                            candidate_nodal[v][iy][ix] += C_candidate[v][j][i] * M_BB2Nodal[iy][j] * M_BB2Nodal[ix][i];
                        }
                    }
                }
            }
        }

        int n_cand_faces = extrapolate_face_values_array(candidate_nodal, basis, candidate_faces, npts);

        bool valid = true;
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double s = Limiters::specific_entropy(
                    candidate_nodal[0][iy][ix], candidate_nodal[1][iy][ix],
                    candidate_nodal[2][iy][ix], candidate_nodal[3][iy][ix], p.GAMMA);
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

    for (int v = 0; v < 4; ++v) {
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                C_candidate[v][j][i] = lo * C[v][j][i] + (1.0 - lo) * U_avg[v];
            }
        }
        double calc_avg = 0.0;
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                calc_avg += w_BB[i][j] * C_candidate[v][j][i];
            }
        }
        double delta = calc_avg - U_avg[v];
        for (int j = 0; j < npts; ++j) {
            for (int i = 0; i < npts; ++i) {
                C_candidate[v][j][i] -= (delta / sum_w_sq) * w_BB[i][j];
            }
        }

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double val = 0.0;
                for (int j = 0; j < npts; ++j) {
                    for (int i = 0; i < npts; ++i) {
                        val += C_candidate[v][j][i] * M_BB2Nodal[iy][j] * M_BB2Nodal[ix][i];
                    }
                }
                c.get_U(v, iy, ix, npts) = val;
            }
        }
    }

    return true;
}

} // namespace Limiters

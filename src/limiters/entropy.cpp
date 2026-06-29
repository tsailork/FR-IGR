/**
 * @file entropy.cpp
 * @brief Implementation of the Entropy Minimum-Preservation Limiter on decoupled Cells.
 */

#include "entropy.hpp"
#include "../core/solver.hpp"
#include "limiter_common.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

Limiters::LimiterStats Limiters::apply_entropy_limiter(Solver &solver) {
    const Parameters &p = solver.p;
    const Basis &basis = solver.basis;
    std::vector<Cell*>& cells = solver.cells;

    // --- 1. Pre-calculate min entropy for every cell ---
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i]->s_min_val = min_entropy_in_cell(*cells[i], p, p.N_PTS);
    }

    // --- 2. Apply limiting cell-by-cell ---
    int num_limited = 0;
    double sum_theta = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:num_limited, sum_theta)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        double s_floor = c->s_min_val;

        // Gather neighborhood entropy minimum from conforming neighbors
        for (int f = 0; f < 4; ++f) {
            if (c->neighbors[f]) {
                s_floor = std::min(s_floor, c->neighbors[f]->s_min_val);
            }
        }

        // Lower s_floor to avoid being overly dissipative
        s_floor -= 1.0E-4;
        if (s_floor < 1.0E-14) s_floor = 1.0E-14;

        // --- Cell average ---
        double r_avg, ru_avg, rv_avg, E_avg;
        compute_cell_average(*c, basis, r_avg, ru_avg, rv_avg, E_avg, p.N_PTS);

        // --- Extrapolate face values for checking ---
        double face_pts[MAX_FACE_PTS][4];
        int n_face = extrapolate_face_values(*c, basis, face_pts, p.N_PTS);

        // --- Find worst θ ---
        double theta_s = 1.0;

        // Check interior solution points
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                double s = specific_entropy(
                    c->get_U(0, iy, ix, p.N_PTS), c->get_U(1, iy, ix, p.N_PTS),
                    c->get_U(2, iy, ix, p.N_PTS), c->get_U(3, iy, ix, p.N_PTS), p.GAMMA);
                if (s < s_floor) {
                    theta_s = std::min(theta_s, bisect_for_theta(
                        c->get_U(0, iy, ix, p.N_PTS), c->get_U(1, iy, ix, p.N_PTS),
                        c->get_U(2, iy, ix, p.N_PTS), c->get_U(3, iy, ix, p.N_PTS),
                        r_avg, ru_avg, rv_avg, E_avg, p.GAMMA, s_floor, false));
                }
            }
        }

        // Check face-extrapolated points (Zhang-Shu requirement for GL nodes)
        for (int f = 0; f < n_face; ++f) {
            double s = specific_entropy(face_pts[f][0], face_pts[f][1],
                                        face_pts[f][2], face_pts[f][3], p.GAMMA);
            if (s < s_floor) {
                theta_s = std::min(theta_s, bisect_for_theta(
                    face_pts[f][0], face_pts[f][1],
                    face_pts[f][2], face_pts[f][3],
                    r_avg, ru_avg, rv_avg, E_avg, p.GAMMA, s_floor, false));
            }
        }

        // --- Apply scaling ---
        if (theta_s < 1.0) {
            for (int iy = 0; iy < p.N_PTS; ++iy) {
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    c->get_U(0, iy, ix, p.N_PTS) = theta_s * (c->get_U(0, iy, ix, p.N_PTS) - r_avg) + r_avg;
                    c->get_U(1, iy, ix, p.N_PTS) = theta_s * (c->get_U(1, iy, ix, p.N_PTS) - ru_avg) + ru_avg;
                    c->get_U(2, iy, ix, p.N_PTS) = theta_s * (c->get_U(2, iy, ix, p.N_PTS) - rv_avg) + rv_avg;
                    c->get_U(3, iy, ix, p.N_PTS) = theta_s * (c->get_U(3, iy, ix, p.N_PTS) - E_avg) + E_avg;
                }
            }
            num_limited++;
            sum_theta += theta_s;
        }
    }

    LimiterStats stats;
    stats.num_limited = num_limited;
    stats.sum_theta = sum_theta;
    return stats;
}

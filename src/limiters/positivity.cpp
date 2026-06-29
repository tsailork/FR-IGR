/**
 * @file positivity.cpp
 * @brief Zhang-Shu bounds-preserving (positivity) limiter on decoupled Cells.
 */

#include "positivity.hpp"
#include "limiter_common.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

Limiters::LimiterStats Limiters::apply_positivity_limiter(std::vector<Cell*>& cells, const Basis& basis,
                                         const Parameters& p)
{
    const double eps = p.POS_LIMITER_EPS;
    const double gm1 = p.GAMMA - 1.0;
    const int npts = p.N_PTS;

    int num_limited = 0;
    double sum_theta = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:num_limited, sum_theta)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];

        // =============================================================
        // 0. Compute cell average
        // =============================================================
        double r_avg, ru_avg, rv_avg, E_avg;
        compute_cell_average(*c, basis, r_avg, ru_avg, rv_avg, E_avg, npts);

        if (r_avg < eps) {
            r_avg = eps;
        }
        double ke_avg = 0.5 * (ru_avg*ru_avg + rv_avg*rv_avg) / r_avg;
        double p_avg  = gm1 * (E_avg - ke_avg);
        if (p_avg < eps) {
            E_avg = eps / gm1 + ke_avg;
        }

        // =============================================================
        // 1. Extrapolate face checking points
        // =============================================================
        double face_pts[MAX_FACE_PTS][4];
        int n_face = extrapolate_face_values(*c, basis, face_pts, npts);

        // =============================================================
        // 2. PASS 1 — Density limiting
        // =============================================================
        double r_min = r_avg;

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                r_min = std::min(r_min, c->get_U(0, iy, ix, npts));
            }
        }

        for (int f = 0; f < n_face; ++f) {
            r_min = std::min(r_min, face_pts[f][0]);
        }

        double theta_r = 1.0;
        if (r_min < eps) {
            theta_r = (r_avg - eps) / (r_avg - r_min);
            theta_r = std::max(0.0, std::min(1.0, theta_r));
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    c->get_U(0, iy, ix, npts) = theta_r*(c->get_U(0, iy, ix, npts) - r_avg)  + r_avg;
                    c->get_U(1, iy, ix, npts) = theta_r*(c->get_U(1, iy, ix, npts) - ru_avg) + ru_avg;
                    c->get_U(2, iy, ix, npts) = theta_r*(c->get_U(2, iy, ix, npts) - rv_avg) + rv_avg;
                    c->get_U(3, iy, ix, npts) = theta_r*(c->get_U(3, iy, ix, npts) - E_avg)  + E_avg;
                }
            }

            n_face = extrapolate_face_values(*c, basis, face_pts, npts);
        }

        // =============================================================
        // 3. PASS 2 — Pressure limiting
        // =============================================================
        double theta_p = 1.0;

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double press = Limiters::pressure(
                    c->get_U(0, iy, ix, npts), c->get_U(1, iy, ix, npts),
                    c->get_U(2, iy, ix, npts), c->get_U(3, iy, ix, npts), p.GAMMA);
                if (press < eps) {
                    theta_p = std::min(theta_p, bisect_for_theta(
                        c->get_U(0, iy, ix, npts), c->get_U(1, iy, ix, npts),
                        c->get_U(2, iy, ix, npts), c->get_U(3, iy, ix, npts),
                        r_avg, ru_avg, rv_avg, E_avg,
                        p.GAMMA, eps, true));
                }
            }
        }

        for (int f = 0; f < n_face; ++f) {
            double press = Limiters::pressure(
                face_pts[f][0], face_pts[f][1],
                face_pts[f][2], face_pts[f][3], p.GAMMA);
            if (press < eps) {
                theta_p = std::min(theta_p, bisect_for_theta(
                    face_pts[f][0], face_pts[f][1],
                    face_pts[f][2], face_pts[f][3],
                    r_avg, ru_avg, rv_avg, E_avg,
                    p.GAMMA, eps, true));
            }
        }

        if (theta_p < 1.0) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    c->get_U(0, iy, ix, npts) = theta_p*(c->get_U(0, iy, ix, npts) - r_avg)  + r_avg;
                    c->get_U(1, iy, ix, npts) = theta_p*(c->get_U(1, iy, ix, npts) - ru_avg) + ru_avg;
                    c->get_U(2, iy, ix, npts) = theta_p*(c->get_U(2, iy, ix, npts) - rv_avg) + rv_avg;
                    c->get_U(3, iy, ix, npts) = theta_p*(c->get_U(3, iy, ix, npts) - E_avg)  + E_avg;
                }
            }
            num_limited++;
            sum_theta += theta_p;
        } else if (r_min < eps) {
            num_limited++;
            sum_theta += theta_r;
        }
    }
    
    LimiterStats stats;
    stats.num_limited = num_limited;
    stats.sum_theta = sum_theta;
    return stats;
}

/**
 * @file positivity.cpp
 * @brief Zhang-Shu bounds-preserving (positivity) limiter on decoupled Cells.
 */
#include "positivity.hpp"
#include "limiter_common.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

static int extrapolate_face_values_ppr(const Cell& c, const Basis& basis,
                                        double face_pts[][4], double face_S[], int npts) {
    int count = 0;

    // Left face (ξ = −1): for each iy, sum over ix with l_L[ix]
    for (int iy = 0; iy < npts; ++iy) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        face_S[count] = 0.0;
        for (int ix = 0; ix < npts; ++ix) {
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_L[ix];
            face_S[count] += c.S_field[iy * npts + ix] * basis.l_L[ix];
        }
        ++count;
    }
    // Right face (ξ = +1): for each iy, sum over ix with l_R[ix]
    for (int iy = 0; iy < npts; ++iy) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        face_S[count] = 0.0;
        for (int ix = 0; ix < npts; ++ix) {
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_R[ix];
            face_S[count] += c.S_field[iy * npts + ix] * basis.l_R[ix];
        }
        ++count;
    }
    // Bottom face (η = −1): for each ix, sum over iy with l_L[iy]
    for (int ix = 0; ix < npts; ++ix) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        face_S[count] = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_L[iy];
            face_S[count] += c.S_field[iy * npts + ix] * basis.l_L[iy];
        }
        ++count;
    }
    // Top face (η = +1): for each ix, sum over iy with l_R[iy]
    for (int ix = 0; ix < npts; ++ix) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        face_S[count] = 0.0;
        for (int iy = 0; iy < npts; ++iy) {
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_R[iy];
            face_S[count] += c.S_field[iy * npts + ix] * basis.l_R[iy];
        }
        ++count;
    }
    return count;
}

static double bisect_for_theta_ppr(
    double r, double ru, double rv, double E, double S,
    double r_avg, double ru_avg, double rv_avg, double E_avg, double S_avg,
    double gamma, double theta_ppr, double target, bool is_reg, int n_iter = 30) {
    double lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < n_iter; ++iter) {
        double mid = 0.5 * (lo + hi);
        double rt  = mid * r  + (1.0 - mid) * r_avg;
        double rut = mid * ru + (1.0 - mid) * ru_avg;
        double rvt = mid * rv + (1.0 - mid) * rv_avg;
        double Et  = mid * E  + (1.0 - mid) * E_avg;
        double St  = mid * S  + (1.0 - mid) * S_avg;
        
        double p_phys = (gamma - 1.0) * (Et - 0.5 * (rut*rut + rvt*rvt) / rt);
        double val = 0.0;
        if (is_reg) {
            double p_phan = St / rt;
            val = p_phys + theta_ppr * (p_phys - p_phan);
        } else {
            val = St / rt; // p_phan
        }
        if (val >= target) lo = mid;
        else               hi = mid;
    }
    return lo * (1.0 - 1e-8);
}

} // namespace

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

        double S_avg = 0.0;
        if (p.ENABLE_PPR) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                    S_avg += w * c->S_field[iy * npts + ix];
                }
            }
            // Clamp average S to ensure P_phan_avg >= eps
            double p_phan_avg = S_avg / r_avg;
            if (p_phan_avg < eps) {
                p_phan_avg = eps;
                S_avg = eps * r_avg;
            }
            // Clamp average S to ensure P_reg_avg >= eps
            double max_p_phan_avg = ((1.0 + p.PPR_THETA) * p_avg - eps) / p.PPR_THETA;
            if (p_phan_avg > max_p_phan_avg) {
                p_phan_avg = max_p_phan_avg;
                S_avg = max_p_phan_avg * r_avg;
            }
        }

        // =============================================================
        // 1. Extrapolate face checking points
        // =============================================================
        double face_pts[MAX_FACE_PTS][4];
        double face_S[MAX_FACE_PTS];
        int n_face = 0;
        if (p.ENABLE_PPR) {
            n_face = extrapolate_face_values_ppr(*c, basis, face_pts, face_S, npts);
        } else {
            n_face = extrapolate_face_values(*c, basis, face_pts, npts);
        }

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
                    if (p.ENABLE_PPR) {
                        c->S_field[iy * npts + ix] = theta_r*(c->S_field[iy * npts + ix] - S_avg) + S_avg;
                    }
                }
            }

            if (p.ENABLE_PPR) {
                n_face = extrapolate_face_values_ppr(*c, basis, face_pts, face_S, npts);
            } else {
                n_face = extrapolate_face_values(*c, basis, face_pts, npts);
            }
        }

        // =============================================================
        // 3. PASS 2 — Pressure & PPR limiting
        // =============================================================
        double theta_p = 1.0;

        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double rho = c->get_U(0, iy, ix, npts);
                double rhou = c->get_U(1, iy, ix, npts);
                double rhov = c->get_U(2, iy, ix, npts);
                double E = c->get_U(3, iy, ix, npts);
                double press = Limiters::pressure(rho, rhou, rhov, E, p.GAMMA);
                if (press < eps) {
                    theta_p = std::min(theta_p, bisect_for_theta(
                        rho, rhou, rhov, E, r_avg, ru_avg, rv_avg, E_avg,
                        p.GAMMA, eps, true));
                }

                if (p.ENABLE_PPR) {
                    double S = c->S_field[iy * npts + ix];
                    double p_phan = S / rho;
                    if (p_phan < eps) {
                        theta_p = std::min(theta_p, bisect_for_theta_ppr(
                            rho, rhou, rhov, E, S, r_avg, ru_avg, rv_avg, E_avg, S_avg,
                            p.GAMMA, p.PPR_THETA, eps, false));
                    }
                    double p_reg = press + p.PPR_THETA * (press - p_phan);
                    if (p_reg < eps) {
                        theta_p = std::min(theta_p, bisect_for_theta_ppr(
                            rho, rhou, rhov, E, S, r_avg, ru_avg, rv_avg, E_avg, S_avg,
                            p.GAMMA, p.PPR_THETA, eps, true));
                    }
                }
            }
        }

        for (int f = 0; f < n_face; ++f) {
            double rho = face_pts[f][0];
            double rhou = face_pts[f][1];
            double rhov = face_pts[f][2];
            double E = face_pts[f][3];
            double press = Limiters::pressure(rho, rhou, rhov, E, p.GAMMA);
            if (press < eps) {
                theta_p = std::min(theta_p, bisect_for_theta(
                    rho, rhou, rhov, E, r_avg, ru_avg, rv_avg, E_avg,
                    p.GAMMA, eps, true));
            }

            if (p.ENABLE_PPR) {
                double S = face_S[f];
                double p_phan = S / rho;
                if (p_phan < eps) {
                    theta_p = std::min(theta_p, bisect_for_theta_ppr(
                        rho, rhou, rhov, E, S, r_avg, ru_avg, rv_avg, E_avg, S_avg,
                        p.GAMMA, p.PPR_THETA, eps, false));
                }
                double p_reg = press + p.PPR_THETA * (press - p_phan);
                if (p_reg < eps) {
                    theta_p = std::min(theta_p, bisect_for_theta_ppr(
                        rho, rhou, rhov, E, S, r_avg, ru_avg, rv_avg, E_avg, S_avg,
                        p.GAMMA, p.PPR_THETA, eps, true));
                }
            }
        }

        if (theta_p < 1.0) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    c->get_U(0, iy, ix, npts) = theta_p*(c->get_U(0, iy, ix, npts) - r_avg)  + r_avg;
                    c->get_U(1, iy, ix, npts) = theta_p*(c->get_U(1, iy, ix, npts) - ru_avg) + ru_avg;
                    c->get_U(2, iy, ix, npts) = theta_p*(c->get_U(2, iy, ix, npts) - rv_avg) + rv_avg;
                    c->get_U(3, iy, ix, npts) = theta_p*(c->get_U(3, iy, ix, npts) - E_avg)  + E_avg;
                    if (p.ENABLE_PPR) {
                        c->S_field[iy * npts + ix] = theta_p*(c->S_field[iy * npts + ix] - S_avg) + S_avg;
                    }
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

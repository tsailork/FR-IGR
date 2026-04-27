/// @file positivity.cpp
/// @brief Zhang-Shu positivity-preserving limiter implementation.
///
/// Enforces ρ ≥ ε and p ≥ ε by scaling the polynomial toward the cell
/// average.  Two passes per element:
///   Pass 1 (density): scale toward mean until ρ_i ≥ ε  at ALL checking pts.
///   Pass 2 (pressure): scale toward mean until p_i ≥ ε at ALL checking pts.
///
/// CRITICAL: For Gauss-Legendre solution points the checking set S must
/// include BOTH the interior GL nodes AND the face-extrapolated values at
/// ξ = ±1.  GL points do not include element endpoints, so the polynomial
/// can overshoot at faces even when all interior points are admissible.
/// Omitting the face points is the classic cause of positivity failure
/// with GL-based DG / FR schemes (Zhang & Shu, JCP 2010, §3.3).
///
/// OpenMP: parallelised over elements.

#include "positivity.hpp"
#include "limiter_common.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void Limiters::apply_positivity_limiter(State& U, const Basis& basis,
                                         const Parameters& p)
{
    const double eps = p.POS_LIMITER_EPS;
    const double gm1 = p.GAMMA - 1.0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
        for (int ex = 0; ex < p.N_ELEM_X; ++ex) {

            // =============================================================
            // 0. Compute cell average (conserved by GL quadrature)
            // =============================================================
            double r_avg, ru_avg, rv_avg, E_avg;
            compute_cell_average(U, basis, p, ey, ex,
                                 r_avg, ru_avg, rv_avg, E_avg);

            // If the cell average itself is non-admissible, the limiter
            // cannot recover through scaling alone.  Apply a conservative
            // hard floor: clamp the average to a minimal admissible state.
            // This is a last-resort safety net; under proper CFL conditions
            // the cell average should remain admissible.
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
            int n_face = extrapolate_face_values(U, basis, p, ey, ex, face_pts);

            // =============================================================
            // 2. PASS 1 — Density limiting
            //    Find the minimum density across ALL checking points
            //    (interior GL nodes + face-extrapolated values).
            // =============================================================
            double r_min = r_avg;

            // Interior GL points
            for (int iy = 0; iy < p.N_PTS; ++iy)
                for (int ix = 0; ix < p.N_PTS; ++ix)
                    r_min = std::min(r_min, U(0, ey, ex, iy, ix));

            // Face-extrapolated points
            for (int f = 0; f < n_face; ++f)
                r_min = std::min(r_min, face_pts[f][0]);

            if (r_min < eps) {
                double theta_r = (r_avg - eps) / (r_avg - r_min);
                theta_r = std::max(0.0, std::min(1.0, theta_r));
                for (int iy = 0; iy < p.N_PTS; ++iy)
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        U(0, ey, ex, iy, ix) = theta_r*(U(0, ey, ex, iy, ix) - r_avg)  + r_avg;
                        U(1, ey, ex, iy, ix) = theta_r*(U(1, ey, ex, iy, ix) - ru_avg) + ru_avg;
                        U(2, ey, ex, iy, ix) = theta_r*(U(2, ey, ex, iy, ix) - rv_avg) + rv_avg;
                        U(3, ey, ex, iy, ix) = theta_r*(U(3, ey, ex, iy, ix) - E_avg)  + E_avg;
                    }

                // Recompute face extrapolations after density limiting
                // (the polynomial has changed)
                n_face = extrapolate_face_values(U, basis, p, ey, ex, face_pts);
            }

            // =============================================================
            // 3. PASS 2 — Pressure limiting (bisection)
            //    Check pressure at ALL checking points.
            // =============================================================
            double theta_p = 1.0;

            // Interior GL points
            for (int iy = 0; iy < p.N_PTS; ++iy)
                for (int ix = 0; ix < p.N_PTS; ++ix) {
                    double press = Limiters::pressure(
                        U(0, ey, ex, iy, ix), U(1, ey, ex, iy, ix),
                        U(2, ey, ex, iy, ix), U(3, ey, ex, iy, ix), p.GAMMA);
                    if (press < eps)
                        theta_p = std::min(theta_p, bisect_for_theta(
                            U(0, ey, ex, iy, ix), U(1, ey, ex, iy, ix),
                            U(2, ey, ex, iy, ix), U(3, ey, ex, iy, ix),
                            r_avg, ru_avg, rv_avg, E_avg,
                            p.GAMMA, eps, true));
                }

            // Face-extrapolated points
            for (int f = 0; f < n_face; ++f) {
                double press = Limiters::pressure(
                    face_pts[f][0], face_pts[f][1],
                    face_pts[f][2], face_pts[f][3], p.GAMMA);
                if (press < eps)
                    theta_p = std::min(theta_p, bisect_for_theta(
                        face_pts[f][0], face_pts[f][1],
                        face_pts[f][2], face_pts[f][3],
                        r_avg, ru_avg, rv_avg, E_avg,
                        p.GAMMA, eps, true));
            }

            if (theta_p < 1.0)
                for (int iy = 0; iy < p.N_PTS; ++iy)
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        U(0, ey, ex, iy, ix) = theta_p*(U(0, ey, ex, iy, ix) - r_avg)  + r_avg;
                        U(1, ey, ex, iy, ix) = theta_p*(U(1, ey, ex, iy, ix) - ru_avg) + ru_avg;
                        U(2, ey, ex, iy, ix) = theta_p*(U(2, ey, ex, iy, ix) - rv_avg) + rv_avg;
                        U(3, ey, ex, iy, ix) = theta_p*(U(3, ey, ex, iy, ix) - E_avg)  + E_avg;
                    }
        }
    }
}

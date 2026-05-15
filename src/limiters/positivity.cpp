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

/// Apply the Zhang-Shu positivity-preserving limiter.
///
/// @details
/// Data Structures & Indexing:
///   - `U(v, ey, ex, iy, ix)`: The global conserved state array. Read and mutated in-place.
/// Assumptions:
///   - Elements are completely independent. Limiting relies strictly on the local element's 
///     interior solution points to compute the cell average and the minimum value. No neighbor 
///     ghost states are accessed, guaranteeing perfect OpenMP thread safety across elements.
Limiters::LimiterStats Limiters::apply_positivity_limiter(State& U, const Basis& basis,
                                         const Parameters& p)
{
    const double eps = p.POS_LIMITER_EPS;
    const double gm1 = p.GAMMA - 1.0;

    int num_limited = 0;
    double sum_theta = 0.0;

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:num_limited, sum_theta)
    for (int ey = 0; ey < U.ny; ++ey) {
        for (int ex = 0; ex < U.nx; ++ex) {

            // =============================================================
            // 0. Compute cell average (conserved by GL quadrature)
            // =============================================================
            double r_avg, ru_avg, rv_avg, E_avg;
            compute_cell_average(U, basis, ey, ex,
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
            int n_face = extrapolate_face_values(U, basis, ey, ex, face_pts);

            // =============================================================
            // 2. PASS 1 — Density limiting
            //    Find the minimum density across ALL checking points
            //    (interior GL nodes + face-extrapolated values).
            // =============================================================
            double r_min = r_avg;

            // Interior GL points
            for (int iy = 0; iy < U.npts; ++iy)
                for (int ix = 0; ix < U.npts; ++ix)
                    r_min = std::min(r_min, U(0, ey, ex, iy, ix));

            // Face-extrapolated points
            for (int f = 0; f < n_face; ++f)
                r_min = std::min(r_min, face_pts[f][0]);

            double theta_r = 1.0;
            if (r_min < eps) {
                theta_r = (r_avg - eps) / (r_avg - r_min);
                theta_r = std::max(0.0, std::min(1.0, theta_r));
                for (int iy = 0; iy < U.npts; ++iy)
                    for (int ix = 0; ix < U.npts; ++ix) {
                        U(0, ey, ex, iy, ix) = theta_r*(U(0, ey, ex, iy, ix) - r_avg)  + r_avg;
                        U(1, ey, ex, iy, ix) = theta_r*(U(1, ey, ex, iy, ix) - ru_avg) + ru_avg;
                        U(2, ey, ex, iy, ix) = theta_r*(U(2, ey, ex, iy, ix) - rv_avg) + rv_avg;
                        U(3, ey, ex, iy, ix) = theta_r*(U(3, ey, ex, iy, ix) - E_avg)  + E_avg;
                    }

                // Recompute face extrapolations after density limiting
                // (the polynomial has changed)
                n_face = extrapolate_face_values(U, basis, ey, ex, face_pts);
            }

            // =============================================================
            // 3. PASS 2 — Pressure limiting (bisection)
            //    Check pressure at ALL checking points.
            // =============================================================
            double theta_p = 1.0;

            // Interior GL points
            for (int iy = 0; iy < U.npts; ++iy)
                for (int ix = 0; ix < U.npts; ++ix) {
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

            if (theta_p < 1.0) {
                for (int iy = 0; iy < U.npts; ++iy)
                    for (int ix = 0; ix < U.npts; ++ix) {
                        U(0, ey, ex, iy, ix) = theta_p*(U(0, ey, ex, iy, ix) - r_avg)  + r_avg;
                        U(1, ey, ex, iy, ix) = theta_p*(U(1, ey, ex, iy, ix) - ru_avg) + ru_avg;
                        U(2, ey, ex, iy, ix) = theta_p*(U(2, ey, ex, iy, ix) - rv_avg) + rv_avg;
                        U(3, ey, ex, iy, ix) = theta_p*(U(3, ey, ex, iy, ix) - E_avg)  + E_avg;
                    }
                num_limited++;
                sum_theta += theta_p;
            } else if (r_min < eps) {
                // If it was limited by density only, we still count it
                num_limited++;
                sum_theta += theta_r;
            }
        }
    }
    
    LimiterStats stats;
    stats.num_limited = num_limited;
    stats.sum_theta = sum_theta;
    return stats;
}

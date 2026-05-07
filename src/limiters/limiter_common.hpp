/// @file limiter_common.hpp
/// @brief Shared helper functions for Zhang-Shu and entropy limiters.
///
/// All functions are inline so they can be used from multiple .cpp files
/// without ODR violations.

#pragma once
#include "../core/state.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"
#include <algorithm>
#include <cmath>

namespace Limiters {

/// Structure to track limiter activity across the mesh.
struct LimiterStats {
    int num_limited = 0;
    double sum_theta = 0.0;
};

/// Thermodynamic pressure from conserved variables.
inline double pressure(double rho, double rhou, double rhov, double E,
                       double gamma) {
    return (gamma - 1.0) * (E - 0.5 * (rhou*rhou + rhov*rhov) / rho);
}

/// Specific entropy  s = p / ρ^γ.
inline double specific_entropy(double rho, double rhou, double rhov,
                                double E, double gamma) {
    double p = pressure(rho, rhou, rhov, E, gamma);
    return p / std::pow(rho, gamma);
}

/// Pressure along the affine path  U(θ) = θ·U_pt + (1−θ)·Ū.
inline double pressure_at_theta(double theta,
                                 double r, double ru, double rv, double E,
                                 double r_avg, double ru_avg, double rv_avg,
                                 double E_avg, double gamma) {
    double rt  = theta * r  + (1.0 - theta) * r_avg;
    double rut = theta * ru + (1.0 - theta) * ru_avg;
    double rvt = theta * rv + (1.0 - theta) * rv_avg;
    double Et  = theta * E  + (1.0 - theta) * E_avg;
    return (gamma - 1.0) * (Et - 0.5 * (rut*rut + rvt*rvt) / rt);
}

/// Specific entropy along the affine path.
inline double entropy_at_theta(double theta,
                                double r, double ru, double rv, double E,
                                double r_avg, double ru_avg, double rv_avg,
                                double E_avg, double gamma) {
    double rt  = theta * r  + (1.0 - theta) * r_avg;
    double rut = theta * ru + (1.0 - theta) * ru_avg;
    double rvt = theta * rv + (1.0 - theta) * rv_avg;
    double Et  = theta * E  + (1.0 - theta) * E_avg;
    double pt  = (gamma - 1.0) * (Et - 0.5 * (rut*rut + rvt*rvt) / rt);
    return pt / std::pow(rt, gamma);
}

/// Bisection to find the largest safe θ ∈ [0,1].
inline double bisect_for_theta(
    double r, double ru, double rv, double E,
    double r_avg, double ru_avg, double rv_avg, double E_avg,
    double gamma, double target, bool is_pressure, int n_iter = 30)
{
    double lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < n_iter; ++iter) {
        double mid = 0.5 * (lo + hi);
        double val = is_pressure
            ? pressure_at_theta(mid, r, ru, rv, E, r_avg, ru_avg, rv_avg, E_avg, gamma)
            : entropy_at_theta(mid, r, ru, rv, E, r_avg, ru_avg, rv_avg, E_avg, gamma);
        if (val >= target) lo = mid;
        else               hi = mid;
    }
    // Safety pullback: the bisection converges to the admissibility boundary
    // where pressure ≈ eps.  Floating-point cancellation in E − ½(ρu²+ρv²)/ρ
    // can push the result to exactly zero.  Pulling θ slightly toward the
    // cell average (θ = 0) guarantees we remain strictly admissible.
    return lo * (1.0 - 1e-8);
}

/// Maximum number of face-extrapolated checking points in 2D.
/// 4 faces × N_PTS points per face edge = 4 * MAX_PTS = 16.
constexpr int MAX_FACE_PTS = 4 * 4;

/// Extrapolate conserved variables to face-centre checking points.
///
/// For Zhang-Shu positivity preservation with Gauss-Legendre quadrature,
/// the checking set S must include the face-extrapolated values at ξ = ±1
/// in addition to the interior GL solution points.  GL points do NOT
/// include the element endpoints, so the polynomial can overshoot at
/// faces even when all interior points are admissible.
///
/// This function evaluates the 2D tensor-product polynomial at the 4N
/// face midpoints:
///   Left   face (ξ = −1): for each iy, extrapolate across ix using l_L.
///   Right  face (ξ = +1): for each iy, extrapolate across ix using l_R.
///   Bottom face (η = −1): for each ix, extrapolate across iy using l_L.
///   Top    face (η = +1): for each ix, extrapolate across iy using l_R.
///
/// @param[out] face_pts  Array of at least 4*N_PTS × 4 conserved states.
/// @return Number of face checking points written (always 4 * N_PTS).
inline int extrapolate_face_values(const State& U, const Basis& basis,
                                    const Parameters& p, int ey, int ex,
                                    double face_pts[][4]) {
    int count = 0;

    // Left face (ξ = −1): for each iy, sum over ix with l_L[ix]
    for (int iy = 0; iy < U.npts; ++iy) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int ix = 0; ix < U.npts; ++ix)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += U(v, ey, ex, iy, ix) * basis.l_L[ix];
        ++count;
    }
    // Right face (ξ = +1): for each iy, sum over ix with l_R[ix]
    for (int iy = 0; iy < U.npts; ++iy) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int ix = 0; ix < U.npts; ++ix)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += U(v, ey, ex, iy, ix) * basis.l_R[ix];
        ++count;
    }
    // Bottom face (η = −1): for each ix, sum over iy with l_L[iy]
    for (int ix = 0; ix < U.npts; ++ix) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int iy = 0; iy < U.npts; ++iy)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += U(v, ey, ex, iy, ix) * basis.l_L[iy];
        ++count;
    }
    // Top face (η = +1): for each ix, sum over iy with l_R[iy]
    for (int ix = 0; ix < U.npts; ++ix) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int iy = 0; iy < U.npts; ++iy)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += U(v, ey, ex, iy, ix) * basis.l_R[iy];
        ++count;
    }
    return count;
}

/// Compute element-average conserved state via GL quadrature weights.
inline void compute_cell_average(const State& U, const Basis& basis,
                                  const Parameters& p, int ey, int ex,
                                  double& r_avg, double& ru_avg,
                                  double& rv_avg, double& E_avg) {
    r_avg = ru_avg = rv_avg = E_avg = 0.0;
    for (int iy = 0; iy < U.npts; ++iy)
        for (int ix = 0; ix < U.npts; ++ix) {
            double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
            r_avg  += w * U(0, ey, ex, iy, ix);
            ru_avg += w * U(1, ey, ex, iy, ix);
            rv_avg += w * U(2, ey, ex, iy, ix);
            E_avg  += w * U(3, ey, ex, iy, ix);
        }
}

/// Minimum specific entropy across all solution points in a cell.
inline double min_entropy_in_cell(const State& U, const Parameters& p,
                                   int ey, int ex) {
    double s_min = 1e30;
    for (int iy = 0; iy < U.npts; ++iy)
        for (int ix = 0; ix < U.npts; ++ix) {
            double r  = std::max(1e-14, U(0, ey, ex, iy, ix));
            double ru = U(1, ey, ex, iy, ix);
            double rv = U(2, ey, ex, iy, ix);
            double E  = U(3, ey, ex, iy, ix);
            s_min = std::min(s_min, specific_entropy(r, ru, rv, E, p.GAMMA));
        }
    return s_min;
}

}  // namespace Limiters

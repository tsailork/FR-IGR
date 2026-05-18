/**
 * @file limiter_common.hpp
 * @brief Shared helper functions and structures for Zhang-Shu and entropy limiters.
 *
 * All functions are defined inline to allow inclusion across multiple compilation
 * units without violating the One Definition Rule (ODR).
 */

#pragma once
#include "../core/state.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"
#include <algorithm>
#include <cmath>

namespace Limiters {

/**
 * @struct LimiterStats
 * @brief Structure to track limiter application statistics across the mesh.
 */
struct LimiterStats {
    int num_limited = 0;      ///< Number of elements that required scaling
    double sum_theta = 0.0;   ///< Sum of scaling factors (theta) for averaging
};

/**
 * @brief Compute thermodynamic pressure from conserved variables.
 *
 * Evaluates the ideal gas equation of state:
 * \f[ p = (\gamma - 1) \left( E - \frac{1}{2} \rho (u^2 + v^2) \right) \f]
 *
 * @param rho Density (\f$\rho\f$)
 * @param rhou X-momentum (\f$\rho u\f$)
 * @param rhov Y-momentum (\f$\rho v\f$)
 * @param E Total energy density (\f$E\f$)
 * @param gamma Specific heat ratio (\f$\gamma\f$)
 * @return Thermodynamic pressure
 */
inline double pressure(double rho, double rhou, double rhov, double E,
                       double gamma) {
    return (gamma - 1.0) * (E - 0.5 * (rhou*rhou + rhov*rhov) / rho);
}

/**
 * @brief Compute specific entropy \f$ s = p / \rho^\gamma \f$.
 *
 * @param rho Density (\f$\rho\f$)
 * @param rhou X-momentum (\f$\rho u\f$)
 * @param rhov Y-momentum (\f$\rho v\f$)
 * @param E Total energy density (\f$E\f$)
 * @param gamma Specific heat ratio (\f$\gamma\f$)
 * @return Specific entropy
 */
inline double specific_entropy(double rho, double rhou, double rhov,
                                double E, double gamma) {
    double p = pressure(rho, rhou, rhov, E, gamma);
    return p / std::pow(rho, gamma);
}

/**
 * @brief Compute pressure along the convex combination path \f$ U(\theta) = \theta U + (1-\theta) \bar{U} \f$.
 *
 * @param theta Scaling factor \f$\theta \in [0, 1]\f$
 * @param r Local point density
 * @param ru Local point X-momentum
 * @param rv Local point Y-momentum
 * @param E Local point total energy
 * @param r_avg Cell-average density
 * @param ru_avg Cell-average X-momentum
 * @param rv_avg Cell-average Y-momentum
 * @param E_avg Cell-average total energy
 * @param gamma Specific heat ratio (\f$\gamma\f$)
 * @return Pressure at the scaled state
 */
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

/**
 * @brief Compute specific entropy along the convex combination path.
 *
 * @param theta Scaling factor \f$\theta \in [0, 1]\f$
 * @param r Local point density
 * @param ru Local point X-momentum
 * @param rv Local point Y-momentum
 * @param E Local point total energy
 * @param r_avg Cell-average density
 * @param ru_avg Cell-average X-momentum
 * @param rv_avg Cell-average Y-momentum
 * @param E_avg Cell-average total energy
 * @param gamma Specific heat ratio (\f$\gamma\f$)
 * @return Specific entropy at the scaled state
 */
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

/**
 * @brief Bisection search to find the maximum admissible \f$\theta \in [0,1]\f$.
 *
 * Finds the largest \f$\theta\f$ such that the scaled state remains strictly above the
 * physical target admissibility bound (e.g. minimum pressure or specific entropy).
 *
 * @param r Local point density
 * @param ru Local point X-momentum
 * @param rv Local point Y-momentum
 * @param E Local point total energy
 * @param r_avg Cell-average density
 * @param ru_avg Cell-average X-momentum
 * @param rv_avg Cell-average Y-momentum
 * @param E_avg Cell-average total energy
 * @param gamma Specific heat ratio (\f$\gamma\f$)
 * @param target Admissibility floor
 * @param is_pressure True to check pressure, false to check specific entropy
 * @param n_iter Number of bisection iterations (default 30)
 * @return Optimal scaling factor \f$\theta\f$
 */
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
    // Safety pullback to ensure strictly admissible bounds in the presence of float variations
    return lo * (1.0 - 1e-8);
}

/**
 * @brief Maximum number of face-extrapolated checking points in 2D.
 *
 * Enforces a bound of 4 faces * 4 solution points per edge.
 */
constexpr int MAX_FACE_PTS = 4 * 4;

/**
 * @brief Extrapolate conserved variables to element face checking points.
 *
 * For positivity preservation under high-order Gauss-Legendre quadrature,
 * the checking set must include face-extrapolated values at \f$\xi = \pm 1, \eta = \pm 1\f$
 * to prevent sub-element polynomial overshoot.
 *
 * @param[in] U Conservated state database
 * @param[in] basis High-order polynomial basis
 * @param[in] ey Element Y index
 * @param[in] ex Element X index
 * @param[out] face_pts Output array to store extrapolated conserved states
 * @return Number of checking points populated (always \f$4 \times N_{pts}\f$)
 */
inline int extrapolate_face_values(const State& U, const Basis& basis,
                                    int ey, int ex,
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

/**
 * @brief Compute the cell-average conserved state using Gauss-Legendre quadrature weights.
 *
 * @param[in] U Conserved state database
 * @param[in] basis High-order polynomial basis
 * @param[in] ey Element Y index
 * @param[in] ex Element X index
 * @param[out] r_avg Output cell-average density
 * @param[out] ru_avg Output cell-average X-momentum
 * @param[out] rv_avg Output cell-average Y-momentum
 * @param[out] E_avg Output cell-average total energy density
 */
inline void compute_cell_average(const State& U, const Basis& basis,
                                  int ey, int ex,
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

/**
 * @brief Find the minimum specific entropy across all interior quadrature points of a cell.
 *
 * @param[in] U Conserved state database
 * @param[in] p Solver parameter configuration
 * @param[in] ey Element Y index
 * @param[in] ex Element X index
 * @return Minimum specific entropy value in the element
 */
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

/**
 * @file limiter_common.hpp
 * @brief Shared helper functions and structures for Cell-level Zhang-Shu and entropy limiters.
 */

#pragma once
#include "../core/cell.hpp"
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
 */
inline double pressure(double rho, double rhou, double rhov, double E,
                       double gamma) {
    return (gamma - 1.0) * (E - 0.5 * (rhou*rhou + rhov*rhov) / rho);
}

/**
 * @brief Compute specific entropy \f$ s = p / \rho^\gamma \f$.
 */
inline double specific_entropy(double rho, double rhou, double rhov,
                                double E, double gamma) {
    double p = pressure(rho, rhou, rhov, E, gamma);
    return p / std::pow(rho, gamma);
}

/**
 * @brief Compute pressure along the convex combination path \f$ U(\theta) = \theta U + (1-\theta) \bar{U} \f$.
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
    return lo * (1.0 - 1e-8);
}

/**
 * @brief Maximum number of face-extrapolated checking points in 2D.
 */
constexpr int MAX_FACE_PTS = 4 * 4;

/**
 * @brief Extrapolate conserved variables to element face checking points.
 */
inline int extrapolate_face_values(const Cell& c, const Basis& basis,
                                    double face_pts[][4], int npts) {
    int count = 0;

    // Left face (ξ = −1): for each iy, sum over ix with l_L[ix]
    for (int iy = 0; iy < npts; ++iy) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int ix = 0; ix < npts; ++ix)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_L[ix];
        ++count;
    }
    // Right face (ξ = +1): for each iy, sum over ix with l_R[ix]
    for (int iy = 0; iy < npts; ++iy) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int ix = 0; ix < npts; ++ix)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_R[ix];
        ++count;
    }
    // Bottom face (η = −1): for each ix, sum over iy with l_L[iy]
    for (int ix = 0; ix < npts; ++ix) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int iy = 0; iy < npts; ++iy)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_L[iy];
        ++count;
    }
    // Top face (η = +1): for each ix, sum over iy with l_R[iy]
    for (int ix = 0; ix < npts; ++ix) {
        for (int v = 0; v < 4; ++v) face_pts[count][v] = 0.0;
        for (int iy = 0; iy < npts; ++iy)
            for (int v = 0; v < 4; ++v)
                face_pts[count][v] += c.get_U(v, iy, ix, npts) * basis.l_R[iy];
        ++count;
    }
    return count;
}

/**
 * @brief Compute the cell-average conserved state using Gauss-Legendre quadrature weights.
 */
inline void compute_cell_average(const Cell& c, const Basis& basis,
                                  double& r_avg, double& ru_avg,
                                  double& rv_avg, double& E_avg, int npts) {
    r_avg = ru_avg = rv_avg = E_avg = 0.0;
    for (int iy = 0; iy < npts; ++iy)
        for (int ix = 0; ix < npts; ++ix) {
            double w = (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
            r_avg  += w * c.get_U(0, iy, ix, npts);
            ru_avg += w * c.get_U(1, iy, ix, npts);
            rv_avg += w * c.get_U(2, iy, ix, npts);
            E_avg  += w * c.get_U(3, iy, ix, npts);
        }
}

/**
 * @brief Find the minimum specific entropy across all interior quadrature points of a cell.
 */
inline double min_entropy_in_cell(const Cell& c, const Parameters& p, int npts) {
    double s_min = 1e30;
    for (int iy = 0; iy < npts; ++iy)
        for (int ix = 0; ix < npts; ++ix) {
            double r  = std::max(1e-14, c.get_U(0, iy, ix, npts));
            double ru = c.get_U(1, iy, ix, npts);
            double rv = c.get_U(2, iy, ix, npts);
            double E  = c.get_U(3, iy, ix, npts);
            s_min = std::min(s_min, specific_entropy(r, ru, rv, E, p.GAMMA));
        }
    return s_min;
}

constexpr int MAX_FACE_PTS_3D = 6 * 4 * 4;

inline double pressure_3d(double rho, double rhou, double rhov, double rhow, double E, double gamma) {
    return (gamma - 1.0) * (E - 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rho);
}

inline double specific_entropy_3d(double rho, double rhou, double rhov, double rhow, double E, double gamma) {
    double p = pressure_3d(rho, rhou, rhov, rhow, E, gamma);
    return p / std::pow(rho, gamma);
}

inline double pressure_at_theta_3d(double theta,
                                    double r, double ru, double rv, double rw, double E,
                                    double r_avg, double ru_avg, double rv_avg, double rw_avg, double E_avg, double gamma) {
    double rt  = theta * r  + (1.0 - theta) * r_avg;
    double rut = theta * ru + (1.0 - theta) * ru_avg;
    double rvt = theta * rv + (1.0 - theta) * rv_avg;
    double rwt = theta * rw + (1.0 - theta) * rw_avg;
    double Et  = theta * E  + (1.0 - theta) * E_avg;
    return (gamma - 1.0) * (Et - 0.5 * (rut*rut + rvt*rvt + rwt*rwt) / rt);
}

inline double entropy_at_theta_3d(double theta,
                                   double r, double ru, double rv, double rw, double E,
                                   double r_avg, double ru_avg, double rv_avg, double rw_avg, double E_avg, double gamma) {
    double rt  = theta * r  + (1.0 - theta) * r_avg;
    double rut = theta * ru + (1.0 - theta) * ru_avg;
    double rvt = theta * rv + (1.0 - theta) * rv_avg;
    double rwt = theta * rw + (1.0 - theta) * rw_avg;
    double Et  = theta * E  + (1.0 - theta) * E_avg;
    double pt  = (gamma - 1.0) * (Et - 0.5 * (rut*rut + rvt*rvt + rwt*rwt) / rt);
    return pt / std::pow(rt, gamma);
}

inline double bisect_for_theta_3d(
    double r, double ru, double rv, double rw, double E,
    double r_avg, double ru_avg, double rv_avg, double rw_avg, double E_avg,
    double gamma, double target, bool is_pressure, int n_iter = 30)
{
    double lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < n_iter; ++iter) {
        double mid = 0.5 * (lo + hi);
        double val = is_pressure
            ? pressure_at_theta_3d(mid, r, ru, rv, rw, E, r_avg, ru_avg, rv_avg, rw_avg, E_avg, gamma)
            : entropy_at_theta_3d(mid, r, ru, rv, rw, E, r_avg, ru_avg, rv_avg, rw_avg, E_avg, gamma);
        if (val >= target) lo = mid;
        else               hi = mid;
    }
    return lo * (1.0 - 1e-8);
}

inline int extrapolate_face_values(const Cell3D& c, const Basis& basis, double face_pts[][5], int npts) {
    int count = 0;
    int npts3 = npts * npts * npts;

    // Left (ξ = −1) & Right (ξ = +1)
    for (int iz = 0; iz < npts; ++iz) {
        for (int iy = 0; iy < npts; ++iy) {
            // Left
            for (int v = 0; v < 5; ++v) face_pts[count][v] = 0.0;
            for (int ix = 0; ix < npts; ++ix) {
                for (int v = 0; v < 5; ++v) {
                    face_pts[count][v] += c.U[v * npts3 + iz * npts * npts + iy * npts + ix] * basis.l_L[ix];
                }
            }
            ++count;

            // Right
            for (int v = 0; v < 5; ++v) face_pts[count][v] = 0.0;
            for (int ix = 0; ix < npts; ++ix) {
                for (int v = 0; v < 5; ++v) {
                    face_pts[count][v] += c.U[v * npts3 + iz * npts * npts + iy * npts + ix] * basis.l_R[ix];
                }
            }
            ++count;
        }
    }

    // Bottom (η = −1) & Top (η = +1)
    for (int iz = 0; iz < npts; ++iz) {
        for (int ix = 0; ix < npts; ++ix) {
            // Bottom
            for (int v = 0; v < 5; ++v) face_pts[count][v] = 0.0;
            for (int iy = 0; iy < npts; ++iy) {
                for (int v = 0; v < 5; ++v) {
                    face_pts[count][v] += c.U[v * npts3 + iz * npts * npts + iy * npts + ix] * basis.l_L[iy];
                }
            }
            ++count;

            // Top
            for (int v = 0; v < 5; ++v) face_pts[count][v] = 0.0;
            for (int iy = 0; iy < npts; ++iy) {
                for (int v = 0; v < 5; ++v) {
                    face_pts[count][v] += c.U[v * npts3 + iz * npts * npts + iy * npts + ix] * basis.l_R[iy];
                }
            }
            ++count;
        }
    }

    // Front (ζ = −1) & Back (ζ = +1)
    for (int iy = 0; iy < npts; ++iy) {
        for (int ix = 0; ix < npts; ++ix) {
            // Front
            for (int v = 0; v < 5; ++v) face_pts[count][v] = 0.0;
            for (int iz = 0; iz < npts; ++iz) {
                for (int v = 0; v < 5; ++v) {
                    face_pts[count][v] += c.U[v * npts3 + iz * npts * npts + iy * npts + ix] * basis.l_L[iz];
                }
            }
            ++count;

            // Back
            for (int v = 0; v < 5; ++v) face_pts[count][v] = 0.0;
            for (int iz = 0; iz < npts; ++iz) {
                for (int v = 0; v < 5; ++v) {
                    face_pts[count][v] += c.U[v * npts3 + iz * npts * npts + iy * npts + ix] * basis.l_R[iz];
                }
            }
            ++count;
        }
    }
    return count;
}

inline void compute_cell_average(const Cell3D& c, const Basis& basis,
                                  double& r_avg, double& ru_avg, double& rv_avg, double& rw_avg,
                                  double& E_avg, int npts) {
    r_avg = ru_avg = rv_avg = rw_avg = E_avg = 0.0;
    int npts3 = npts * npts * npts;
    for (int iz = 0; iz < npts; ++iz) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double w = (basis.w[iz] * 0.5) * (basis.w[iy] * 0.5) * (basis.w[ix] * 0.5);
                int idx = iz * npts * npts + iy * npts + ix;
                r_avg  += w * c.U[0 * npts3 + idx];
                ru_avg += w * c.U[1 * npts3 + idx];
                rv_avg += w * c.U[2 * npts3 + idx];
                rw_avg += w * c.U[3 * npts3 + idx];
                E_avg  += w * c.U[4 * npts3 + idx];
            }
        }
    }
}

inline double min_entropy_in_cell(const Cell3D& c, const Parameters& p, int npts) {
    double s_min = 1e30;
    int npts3 = npts * npts * npts;
    for (int iz = 0; iz < npts; ++iz) {
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                int idx = iz * npts * npts + iy * npts + ix;
                double r  = std::max(1e-14, c.U[0 * npts3 + idx]);
                double ru = c.U[1 * npts3 + idx];
                double rv = c.U[2 * npts3 + idx];
                double rw = c.U[3 * npts3 + idx];
                double E  = c.U[4 * npts3 + idx];
                s_min = std::min(s_min, specific_entropy_3d(r, ru, rv, rw, E, p.GAMMA));
            }
        }
    }
    return s_min;
}

}  // namespace Limiters

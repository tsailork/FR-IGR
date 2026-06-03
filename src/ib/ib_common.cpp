#include "ib.hpp"
#include "../core/solver.hpp"
#include <cmath>
#include <algorithm>
#include <string>

namespace ImmersedBoundary {

double get_circle_sdf(double x, double y, double center_x, double center_y, double radius) {
    double rx = x - center_x;
    double ry = y - center_y;
    return std::sqrt(rx * rx + ry * ry) - radius;
}

double compute_circle_mask(double x, double y, double center_x, double center_y, double radius,
                           bool sharp, double smooth_width, double dx, double dy)
{
    // Signed distance function (phi < 0 inside solid, phi > 0 in fluid)
    double phi = get_circle_sdf(x, y, center_x, center_y, radius);

    if (sharp) {
        return (phi <= 0.0) ? 1.0 : 0.0;
    } else {
        // Regularized/smoothed Heaviside function interface
        double h_size = std::max(dx, dy);
        double epsilon = smooth_width * h_size;

        if (phi < -epsilon) {
            return 1.0;
        } else if (phi > epsilon) {
            return 0.0;
        } else {
            // Smooth transition: chi = 1 - H_epsilon(phi)
            // H_epsilon(phi) = 0.5 * (1 + phi/epsilon + 1/pi * sin(pi * phi / epsilon))
            double ratio = phi / epsilon;
            static const double PI = 3.14159265358979323846;
            double H = 0.5 * (1.0 + ratio + (1.0 / PI) * std::sin(PI * ratio));
            return 1.0 - H;
        }
    }
}

double get_naca_sdf(double x, double y, double x_le, double y_le, double chord,
                    const std::string& naca_code, double aoa_deg)
{
    // Parse NACA 4-digit code (e.g., "0012", "2412")
    double m = 0.0;
    double p = 0.0;
    double t = 0.12; // Default to NACA 0012

    if (naca_code.length() == 4 && std::all_of(naca_code.begin(), naca_code.end(), ::isdigit)) {
        m = (naca_code[0] - '0') * 0.01;
        p = (naca_code[1] - '0') * 0.1;
        t = ((naca_code[2] - '0') * 10 + (naca_code[3] - '0')) * 0.01;
    }

    // Convert angle of attack to radians
    static const double PI = 3.14159265358979323846;
    double alpha = aoa_deg * PI / 180.0;

    // Shift relative to leading edge
    double dx_pt = x - x_le;
    double dy_pt = y - y_le;

    // Rotate query point by -alpha around leading edge to align chord along positive X axis.
    double x_rot = dx_pt * std::cos(alpha) + dy_pt * std::sin(alpha);
    double y_rot = -dx_pt * std::sin(alpha) + dy_pt * std::cos(alpha);

    double xc = x_rot;
    double yc = y_rot;

    double phi = 0.0;

    if (xc < 0.0) {
        // Distance to leading edge
        phi = std::sqrt(xc * xc + yc * yc);
    } else if (xc > chord) {
        // Distance to trailing edge (trailing edge is at chord, 0 in rotated frame)
        phi = std::sqrt((xc - chord) * (xc - chord) + yc * yc);
    } else {
        // Coordinate fraction along chord (0 to 1)
        double xc_frac = xc / chord;

        // Thickness profile
        double yt = 5.0 * t * chord * (
            0.2969 * std::sqrt(xc_frac)
            - 0.1260 * xc_frac
            - 0.3516 * xc_frac * xc_frac
            + 0.2843 * xc_frac * xc_frac * xc_frac
            - 0.1015 * xc_frac * xc_frac * xc_frac * xc_frac
        );

        // Camber line
        double y_camber = 0.0;
        if (m > 0.0 && p > 0.0) {
            if (xc_frac <= p) {
                y_camber = (m * chord / (p * p)) * (2.0 * p * xc_frac - xc_frac * xc_frac);
            } else {
                y_camber = (m * chord / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xc_frac - xc_frac * xc_frac);
            }
        }

        // Signed distance to the airfoil surface
        phi = std::abs(yc - y_camber) - yt;
    }
    return phi;
}

double compute_naca_mask(double x, double y, double x_le, double y_le, double chord,
                         const std::string& naca_code, double aoa_deg,
                         bool sharp, double smooth_width, double dx, double dy)
{
    double phi = get_naca_sdf(x, y, x_le, y_le, chord, naca_code, aoa_deg);
    static const double PI = 3.14159265358979323846;

    if (sharp) {
        return (phi <= 0.0) ? 1.0 : 0.0;
    } else {
        // Smoothed Heaviside
        double h_size = std::max(dx, dy);
        double epsilon = smooth_width * h_size;

        if (phi < -epsilon) {
            return 1.0;
        } else if (phi > epsilon) {
            return 0.0;
        } else {
            double ratio = phi / epsilon;
            double H = 0.5 * (1.0 + ratio + (1.0 / PI) * std::sin(PI * ratio));
            return 1.0 - H;
        }
    }
}

double get_quad_sdf(double px, double py, const QuadShape& quad) {
    double dmin_sq = 1e20;
    int positive_cross_count = 0;
    int negative_cross_count = 0;

    for (int i = 0; i < 4; ++i) {
        double ax = quad.x[i];
        double ay = quad.y[i];
        double bx = quad.x[(i + 1) % 4];
        double by = quad.y[(i + 1) % 4];

        double vx = bx - ax;
        double vy = by - ay;
        double wx = px - ax;
        double wy = py - ay;

        double len_sq = vx * vx + vy * vy;
        double t = 0.0;
        if (len_sq > 1e-20) {
            t = (wx * vx + wy * vy) / len_sq;
        }
        t = std::max(0.0, std::min(1.0, t));

        double proj_x = ax + t * vx;
        double proj_y = ay + t * vy;
        double dx = px - proj_x;
        double dy = py - proj_y;
        double dist_sq = dx * dx + dy * dy;

        if (dist_sq < dmin_sq) {
            dmin_sq = dist_sq;
        }

        double cross = vx * wy - vy * wx;
        if (cross > 0.0) {
            positive_cross_count++;
        } else if (cross < 0.0) {
            negative_cross_count++;
        }
    }

    bool inside = (positive_cross_count == 4 || negative_cross_count == 4);
    double dmin = std::sqrt(dmin_sq);
    return inside ? -dmin : dmin;
}

double get_parabola_sdf(double px, double py, const ParabolaShape& poly, double q) {
    double a = (1.0 - q) * poly.a0 + q * poly.a1;
    double b = (1.0 - q) * poly.b0 + q * poly.b1;
    double c = (1.0 - q) * poly.c0 + q * poly.c1;

    double phi = 0.0;
    if (poly.dir == 'Y') {
        double f_x = a * px * px + b * px + c;
        if (poly.side == "ABOVE") {
            phi = f_x - py;
        } else { // "BELOW"
            phi = py - f_x;
        }
    } else { // 'X'
        double f_y = a * py * py + b * py + c;
        if (poly.side == "RIGHT") {
            phi = f_y - px;
        } else { // "LEFT"
            phi = px - f_y;
        }
    }
    return phi;
}

} // namespace ImmersedBoundary

// Precomputes or updates the cached solid volume fraction mask (ib_mask) for all blocks
void Solver::update_ib_mask_field(double time) {
    if (!p.ENABLE_IB) return;

    for (auto &b : blocks) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int iy = 0; iy < p.N_PTS; ++iy) {
                    for (int ix = 0; ix < p.N_PTS; ++ix) {
                        double x = b.x_min + (ex + 0.5 * (1.0 + basis.z[ix])) * b.dx;
                        double y = b.y_min + (ey + 0.5 * (1.0 + basis.z[iy])) * b.dy;
                        int idx = b.get_flat_idx(ey, ex, iy, ix, p.N_PTS);
                        b.ib_mask[idx] = get_ib_mask_at_time(x, y, time, b.dx, b.dy);
                    }
                }
            }
        }
    }
}

// Implements the Solver-level mask check (backward compatibility)
double Solver::get_ib_mask(double x, double y, double dx, double dy) const {
    return get_ib_mask_at_time(x, y, current_time, dx, dy);
}

// Time-aware immersed boundary mask lookup/evaluation
double Solver::get_ib_mask_at_time(double x, double y, double time, double dx, double dy) const {
    if (!p.ENABLE_IB) return 0.0;

    if (p.IB_SHAPE == "CIRCLE") {
        return ImmersedBoundary::compute_circle_mask(
            x, y, p.IB_CENTER_X, p.IB_CENTER_Y, p.IB_RADIUS,
            p.IB_SHARP, p.IB_SMOOTH_WIDTH, dx, dy
        );
    } else if (p.IB_SHAPE == "NACA") {
        return ImmersedBoundary::compute_naca_mask(
            x, y, p.IB_CENTER_X, p.IB_CENTER_Y, p.IB_RADIUS,
            p.IB_NACA_CODE, p.IB_AOA, p.IB_SHARP, p.IB_SMOOTH_WIDTH, dx, dy
        );
    } else if (p.IB_SHAPE == "MULTI") {
        double phi = 1e20; // Default outside fluid state

        // 1. Union with all quadrilaterals
        for (const auto& quad : p.ib_quads) {
            double d = ImmersedBoundary::get_quad_sdf(x, y, quad);
            if (d < phi) phi = d;
        }

        // 2. Union with all lines/parabolas
        double q_val = p.evaluate_ib_q(time);
        for (const auto& poly : p.ib_polys) {
            double d = ImmersedBoundary::get_parabola_sdf(x, y, poly, q_val);
            if (d < phi) phi = d;
        }

        if (p.IB_SHARP) {
            return (phi <= 0.0) ? 1.0 : 0.0;
        } else {
            // Evaluates regularized smooth Heaviside
            double h_size = std::max(dx, dy);
            double epsilon = p.IB_SMOOTH_WIDTH * h_size;

            if (phi < -epsilon) {
                return 1.0;
            } else if (phi > epsilon) {
                return 0.0;
            } else {
                double ratio = phi / epsilon;
                static const double PI = 3.14159265358979323846;
                double H = 0.5 * (1.0 + ratio + (1.0 / PI) * std::sin(PI * ratio));
                return 1.0 - H;
            }
        }
    }

    return 0.0;
}

double Solver::get_ib_sdf_at_time(double x, double y, double time) const {
    if (!p.ENABLE_IB) return 1e20; // safe distance

    if (p.IB_SHAPE == "CIRCLE") {
        return ImmersedBoundary::get_circle_sdf(
            x, y, p.IB_CENTER_X, p.IB_CENTER_Y, p.IB_RADIUS
        );
    } else if (p.IB_SHAPE == "NACA") {
        return ImmersedBoundary::get_naca_sdf(
            x, y, p.IB_CENTER_X, p.IB_CENTER_Y, p.IB_RADIUS, // actually used as chord and x_le,y_le
            p.IB_NACA_CODE, p.IB_AOA
        );
    } else if (p.IB_SHAPE == "MULTI") {
        double phi = 1e20;

        for (const auto& quad : p.ib_quads) {
            double d = ImmersedBoundary::get_quad_sdf(x, y, quad);
            if (d < phi) phi = d;
        }

        double q_val = p.evaluate_ib_q(time);
        for (const auto& poly : p.ib_polys) {
            double d = ImmersedBoundary::get_parabola_sdf(x, y, poly, q_val);
            if (d < phi) phi = d;
        }
        return phi;
    }
    return 1e20;
}

void Solver::get_ib_sdf_gradient_at_time(double x, double y, double time, double& nx, double& ny) const {
    const double eps = 1e-6;
    double dphi_dx = (get_ib_sdf_at_time(x + eps, y, time) - get_ib_sdf_at_time(x - eps, y, time)) / (2.0 * eps);
    double dphi_dy = (get_ib_sdf_at_time(x, y + eps, time) - get_ib_sdf_at_time(x, y - eps, time)) / (2.0 * eps);
    
    double mag = std::sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy);
    if (mag > 1e-12) {
        nx = dphi_dx / mag;
        ny = dphi_dy / mag;
    } else {
        nx = 0.0;
        ny = 0.0;
    }
}

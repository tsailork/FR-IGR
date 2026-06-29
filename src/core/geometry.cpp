/**
 * @file geometry.cpp
 * @brief Implementation of shape containment and intersection query functions.
 */

#include "geometry.hpp"
#include <cmath>
#include <algorithm>
#include <cctype>

namespace Geometry {

bool point_in_polygon(double x, double y, const std::vector<double>& px, const std::vector<double>& py) {
    if (px.empty()) return false;
    int n = px.size();
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (((py[i] > y) != (py[j] > y)) &&
            (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i])) {
            inside = !inside;
        }
    }
    return inside;
}

bool segment_intersects_aabb(double x1, double y1, double x2, double y2,
                             double xmin, double xmax, double ymin, double ymax) {
    double tmin = 0.0;
    double tmax = 1.0;
    double dx = x2 - x1;
    double dy = y2 - y1;

    // Check X boundaries
    for (int i = 0; i < 2; ++i) {
        double p = (i == 0) ? -dx : dx;
        double q = (i == 0) ? (x1 - xmin) : (xmax - x1);
        if (std::abs(p) < 1e-12) {
            if (q < 0.0) return false; // Parallel and outside
        } else {
            double t = q / p;
            if (p < 0.0) {
                if (t > tmax) return false;
                if (t > tmin) tmin = t;
            } else {
                if (t < tmin) return false;
                if (t < tmax) tmax = t;
            }
        }
    }

    // Check Y boundaries
    for (int i = 0; i < 2; ++i) {
        double p = (i == 0) ? -dy : dy;
        double q = (i == 0) ? (y1 - ymin) : (ymax - y1);
        if (std::abs(p) < 1e-12) {
            if (q < 0.0) return false; // Parallel and outside
        } else {
            double t = q / p;
            if (p < 0.0) {
                if (t > tmax) return false;
                if (t > tmin) tmin = t;
            } else {
                if (t < tmin) return false;
                if (t < tmax) tmax = t;
            }
        }
    }

    return tmin <= tmax;
}

bool polygon_intersects_aabb(const std::vector<double>& px, const std::vector<double>& py,
                             double xmin, double xmax, double ymin, double ymax) {
    if (px.empty()) return false;
    int n = px.size();

    // 1. Is any polygon vertex inside the AABB?
    for (int i = 0; i < n; ++i) {
        if (px[i] >= xmin && px[i] <= xmax && py[i] >= ymin && py[i] <= ymax) {
            return true;
        }
    }

    // 2. Is any corner of the AABB inside the polygon?
    // (Handles the case where AABB is fully inside the polygon)
    if (point_in_polygon(xmin, ymin, px, py)) return true;

    // 3. Do any polygon edges intersect the AABB?
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (segment_intersects_aabb(px[j], py[j], px[i], py[i], xmin, xmax, ymin, ymax)) {
            return true;
        }
    }

    return false;
}

bool Naca::contains(double px, double py) const {
    double m = 0.0;
    double p = 0.0;
    double t = 0.12;

    if (naca_code.length() == 4 && std::all_of(naca_code.begin(), naca_code.end(), ::isdigit)) {
        m = (naca_code[0] - '0') * 0.01;
        p = (naca_code[1] - '0') * 0.1;
        t = ((naca_code[2] - '0') * 10 + (naca_code[3] - '0')) * 0.01;
    }

    static const double PI = 3.14159265358979323846;
    double alpha = aoa_deg * PI / 180.0;

    double dx_pt = px - x_le;
    double dy_pt = py - y_le;

    // Rotate query point by -alpha around leading edge
    double x_rot = dx_pt * std::cos(alpha) + dy_pt * std::sin(alpha);
    double y_rot = -dx_pt * std::sin(alpha) + dy_pt * std::cos(alpha);

    double xc = x_rot;
    double yc = y_rot;

    double phi = 0.0;

    if (xc < 0.0) {
        phi = std::sqrt(xc * xc + yc * yc);
    } else if (xc > chord) {
        phi = std::sqrt((xc - chord) * (xc - chord) + yc * yc);
    } else {
        double xc_frac = xc / chord;

        double yt = 5.0 * t * chord * (
            0.2969 * std::sqrt(xc_frac)
            - 0.1260 * xc_frac
            - 0.3516 * xc_frac * xc_frac
            + 0.2843 * xc_frac * xc_frac * xc_frac
            - 0.1015 * xc_frac * xc_frac * xc_frac * xc_frac
        );

        double y_camber = 0.0;
        if (m > 0.0 && p > 0.0) {
            if (xc_frac <= p) {
                y_camber = (m * chord / (p * p)) * (2.0 * p * xc_frac - xc_frac * xc_frac);
            } else {
                y_camber = (m * chord / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xc_frac - xc_frac * xc_frac);
            }
        }

        phi = std::abs(yc - y_camber) - yt;
    }
    return phi <= 0.0;
}

Polygon Naca::to_polygon(int num_points) const {
    double m = 0.0;
    double p = 0.0;
    double t = 0.12;

    if (naca_code.length() == 4 && std::all_of(naca_code.begin(), naca_code.end(), ::isdigit)) {
        m = (naca_code[0] - '0') * 0.01;
        p = (naca_code[1] - '0') * 0.1;
        t = ((naca_code[2] - '0') * 10 + (naca_code[3] - '0')) * 0.01;
    }

    static const double PI = 3.14159265358979323846;
    double alpha = aoa_deg * PI / 180.0;

    std::vector<double> x_upper, y_upper;
    std::vector<double> x_lower, y_lower;

    for (int i = 0; i <= num_points; ++i) {
        double theta = i * PI / num_points;
        double xc_frac = 0.5 * (1.0 - std::cos(theta));

        double yt = 5.0 * t * chord * (
            0.2969 * std::sqrt(xc_frac)
            - 0.1260 * xc_frac
            - 0.3516 * xc_frac * xc_frac
            + 0.2843 * xc_frac * xc_frac * xc_frac
            - 0.1015 * xc_frac * xc_frac * xc_frac * xc_frac
        );

        double y_camber = 0.0;
        double dyc_dxc = 0.0;
        if (m > 0.0 && p > 0.0) {
            if (xc_frac <= p) {
                y_camber = (m * chord / (p * p)) * (2.0 * p * xc_frac - xc_frac * xc_frac);
                dyc_dxc = (2.0 * m / (p * p)) * (p - xc_frac);
            } else {
                y_camber = (m * chord / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xc_frac - xc_frac * xc_frac);
                dyc_dxc = (2.0 * m / ((1.0 - p) * (1.0 - p))) * (p - xc_frac);
            }
        }

        double theta_c = std::atan(dyc_dxc);

        double xu = xc_frac * chord - yt * std::sin(theta_c);
        double yu = y_camber + yt * std::cos(theta_c);

        double xl = xc_frac * chord + yt * std::sin(theta_c);
        double yl = y_camber - yt * std::cos(theta_c);

        // Rotate and translate
        double xu_rot = xu * std::cos(alpha) - yu * std::sin(alpha) + x_le;
        double yu_rot = xu * std::sin(alpha) + yu * std::cos(alpha) + y_le;

        double xl_rot = xl * std::cos(alpha) - yl * std::sin(alpha) + x_le;
        double yl_rot = xl * std::sin(alpha) + yl * std::cos(alpha) + y_le;

        x_upper.push_back(xu_rot);
        y_upper.push_back(yu_rot);
        x_lower.push_back(xl_rot);
        y_lower.push_back(yl_rot);
    }

    std::vector<double> px, py;
    for (size_t i = 0; i < x_upper.size(); ++i) {
        px.push_back(x_upper[i]);
        py.push_back(y_upper[i]);
    }
    for (int i = (int)x_lower.size() - 2; i >= 0; --i) {
        px.push_back(x_lower[i]);
        py.push_back(y_lower[i]);
    }

    Polygon poly;
    poly.x = px;
    poly.y = py;
    return poly;
}

bool Naca::intersects_aabb(double xmin, double xmax, double ymin, double ymax) const {
    Polygon poly = to_polygon(50);
    return poly.intersects_aabb(xmin, xmax, ymin, ymax);
}

} // namespace Geometry

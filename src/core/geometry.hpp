/**
 * @file geometry.hpp
 * @brief Decoupled shape definitions and intersection query functions.
 */

#pragma once

#include <vector>
#include <string>

namespace Geometry {

/**
 * @brief Check if a point is inside a polygon using ray-casting.
 */
bool point_in_polygon(double x, double y, const std::vector<double>& px, const std::vector<double>& py);

/**
 * @brief Check if a line segment intersects an axis-aligned bounding box (AABB).
 * Implements the Liang-Barsky line clipping algorithm.
 */
bool segment_intersects_aabb(double x1, double y1, double x2, double y2,
                             double xmin, double xmax, double ymin, double ymax);

/**
 * @brief Check if a polygon intersects an axis-aligned bounding box (AABB).
 * Checks if any vertex is inside, if any corner is inside, or if any edge crosses.
 */
bool polygon_intersects_aabb(const std::vector<double>& px, const std::vector<double>& py,
                             double xmin, double xmax, double ymin, double ymax);

/**
 * @struct Polygon
 * @brief Representation of an arbitrary polygon shape.
 */
struct Polygon {
    std::vector<double> x;
    std::vector<double> y;

    bool contains(double px, double py) const {
        return point_in_polygon(px, py, x, y);
    }

    bool intersects_aabb(double xmin, double xmax, double ymin, double ymax) const {
        return polygon_intersects_aabb(x, y, xmin, xmax, ymin, ymax);
    }
};

/**
 * @struct Circle
 * @brief Representation of a circular shape.
 */
struct Circle {
    double cx = 0.0;
    double cy = 0.0;
    double r = 0.0;

    bool contains(double px, double py) const {
        double dx = px - cx;
        double dy = py - cy;
        return (dx * dx + dy * dy) <= (r * r);
    }

    bool intersects_aabb(double xmin, double xmax, double ymin, double ymax) const {
        // Find closest point on AABB to circle center
        double x_closest = (cx < xmin) ? xmin : ((cx > xmax) ? xmax : cx);
        double y_closest = (cy < ymin) ? ymin : ((cy > ymax) ? ymax : cy);
        double dx = x_closest - cx;
        double dy = y_closest - cy;
        return (dx * dx + dy * dy) <= (r * r);
    }
};

/**
 * @struct Naca
 * @brief Representation of a NACA 4-digit airfoil shape.
 */
struct Naca {
    double x_le = 0.0;
    double y_le = 0.0;
    double chord = 1.0;
    std::string naca_code = "0012";
    double aoa_deg = 0.0;

    bool contains(double px, double py) const;
    bool intersects_aabb(double xmin, double xmax, double ymin, double ymax) const;

    /**
     * @brief Generates polygon vertices approximating the NACA airfoil profile.
     */
    Polygon to_polygon(int num_points = 50) const;
};

} // namespace Geometry

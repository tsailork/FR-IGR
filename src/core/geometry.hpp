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
 * 
 * @param[in] x X-coordinate of the point
 * @param[in] y Y-coordinate of the point
 * @param[in] px X-coordinates of the polygon vertices
 * @param[in] py Y-coordinates of the polygon vertices
 * @return True if the point is strictly inside the polygon, false otherwise
 */
bool point_in_polygon(double x, double y, const std::vector<double>& px, const std::vector<double>& py);

/**
 * @brief Check if a line segment intersects an axis-aligned bounding box (AABB).
 * Implements the Liang-Barsky line clipping algorithm.
 * 
 * @param[in] x1 X-coordinate of segment start
 * @param[in] y1 Y-coordinate of segment start
 * @param[in] x2 X-coordinate of segment end
 * @param[in] y2 Y-coordinate of segment end
 * @param[in] xmin Minimum X coordinate of AABB
 * @param[in] xmax Maximum X coordinate of AABB
 * @param[in] ymin Minimum Y coordinate of AABB
 * @param[in] ymax Maximum Y coordinate of AABB
 * @return True if the segment intersects or lies inside the AABB, false otherwise
 */
bool segment_intersects_aabb(double x1, double y1, double x2, double y2,
                             double xmin, double xmax, double ymin, double ymax);

/**
 * @brief Check if a polygon intersects an axis-aligned bounding box (AABB).
 * Checks if any vertex is inside, if any corner is inside, or if any edge crosses.
 * 
 * @param[in] px X-coordinates of the polygon vertices
 * @param[in] py Y-coordinates of the polygon vertices
 * @param[in] xmin Minimum X coordinate of AABB
 * @param[in] xmax Maximum X coordinate of AABB
 * @param[in] ymin Minimum Y coordinate of AABB
 * @param[in] ymax Maximum Y coordinate of AABB
 * @return True if there is any overlap between the polygon and the AABB, false otherwise
 */
bool polygon_intersects_aabb(const std::vector<double>& px, const std::vector<double>& py,
                             double xmin, double xmax, double ymin, double ymax);

/**
 * @struct Polygon
 * @brief Representation of an arbitrary polygon shape.
 */
struct Polygon {
    std::vector<double> x; ///< X-coordinates of vertices
    std::vector<double> y; ///< Y-coordinates of vertices

    /**
     * @brief Check if a point lies inside the polygon.
     * @param[in] px X-coordinate of the point
     * @param[in] py Y-coordinate of the point
     * @return True if contained
     */
    bool contains(double px, double py) const {
        return point_in_polygon(px, py, x, y);
    }

    /**
     * @brief Check if the polygon intersects an AABB.
     * @param[in] xmin Minimum X of AABB
     * @param[in] xmax Maximum X of AABB
     * @param[in] ymin Minimum Y of AABB
     * @param[in] ymax Maximum Y of AABB
     * @return True if intersecting
     */
    bool intersects_aabb(double xmin, double xmax, double ymin, double ymax) const {
        return polygon_intersects_aabb(x, y, xmin, xmax, ymin, ymax);
    }
};

/**
 * @struct Circle
 * @brief Representation of a circular shape.
 */
struct Circle {
    double cx = 0.0; ///< Center X-coordinate
    double cy = 0.0; ///< Center Y-coordinate
    double r = 0.0;  ///< Radius of the circle

    /**
     * @brief Check if a point lies inside the circle.
     * @param[in] px X-coordinate of the point
     * @param[in] py Y-coordinate of the point
     * @return True if contained
     */
    bool contains(double px, double py) const {
        double dx = px - cx;
        double dy = py - cy;
        return (dx * dx + dy * dy) <= (r * r);
    }

    /**
     * @brief Check if the circle intersects an AABB.
     * @param[in] xmin Minimum X of AABB
     * @param[in] xmax Maximum X of AABB
     * @param[in] ymin Minimum Y of AABB
     * @param[in] ymax Maximum Y of AABB
     * @return True if intersecting
     */
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
    double x_le = 0.0;         ///< Leading edge X-coordinate
    double y_le = 0.0;         ///< Leading edge Y-coordinate
    double chord = 1.0;        ///< Chord length
    std::string naca_code = "0012"; ///< NACA 4-digit designation string
    double aoa_deg = 0.0;      ///< Angle of attack in degrees

    /**
     * @brief Check if a point lies inside the NACA airfoil.
     * @param[in] px X-coordinate of the point
     * @param[in] py Y-coordinate of the point
     * @return True if contained
     */
    bool contains(double px, double py) const;

    /**
     * @brief Check if the NACA airfoil intersects an AABB.
     * @param[in] xmin Minimum X of AABB
     * @param[in] xmax Maximum X of AABB
     * @param[in] ymin Minimum Y of AABB
     * @param[in] ymax Maximum Y of AABB
     * @return True if intersecting
     */
    bool intersects_aabb(double xmin, double xmax, double ymin, double ymax) const;

    /**
     * @brief Generates polygon vertices approximating the NACA airfoil profile.
     * @param[in] num_points Number of points to sample along the camber line (default 50)
     * @return Approximated Polygon shape
     */
    Polygon to_polygon(int num_points = 50) const;
};

} // namespace Geometry

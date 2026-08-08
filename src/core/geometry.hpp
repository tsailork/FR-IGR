/**
 * @file geometry.hpp
 * @brief Decoupled shape definitions, ray-casting point containment, and Liang-Barsky clipping.
 *
 * @details
 * Mathematical Algorithms:
 *
 * 1. Ray-Casting Point-in-Polygon Containment:
 *    Casts a horizontal ray \f$ y = y_0 \f$ from \f$ (x_0, y_0) \f$ to \f$ +\infty \f$ and counts edge crossings:
 *    \f[
 *    (y_i > y_0) \neq (y_j > y_0) \quad \text{and} \quad x_0 < \frac{(x_j - x_i)(y_0 - y_i)}{y_j - y_i} + x_i
 *    \f]
 *
 * 2. Liang-Barsky Line Clipping (Segment vs AABB):
 *    Parametric line representation \f$ x(t) = x_1 + t \Delta x, y(t) = y_1 + t \Delta y \f$ for \f$ t \in [0, 1] \f$.
 *    Evaluates boundary inequalities \f$ p_k t \le q_k \f$ to determine valid clipping interval \f$ [t_1, t_2] \f$.
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

namespace fr::geometry {

/**
 * @struct Triangle
 * @brief Representation of a 3D surface triangle for CAD geometry.
 */
struct Triangle {
    double v0[3];       ///< Coordinates of vertex 0 (x, y, z)
    double v1[3];       ///< Coordinates of vertex 1 (x, y, z)
    double v2[3];       ///< Coordinates of vertex 2 (x, y, z)
    double normal[3];   ///< Unit face normal vector (nx, ny, nz)
    double bbox_min[3]; ///< Axis-aligned bounding box min (x, y, z)
    double bbox_max[3]; ///< Axis-aligned bounding box max (x, y, z)

    /**
     * @brief Computes minimum squared distance from query point p to triangle.
     * Implements Christer Ericson 3D distance formulation checking 7 geometric regions.
     * @param p Query point coordinates [x, y, z]
     * @param closest_pt Output closest point on triangle surface [x, y, z]
     * @return Minimum squared distance
     */
    double sq_distance_to_point(const double p[3], double closest_pt[3]) const;

    /**
     * @brief Recalculate bounding box and face normal vector.
     */
    void update_bounds_and_normal();
};

/**
 * @struct BvhNode
 * @brief Node structure for SAH Bounding Volume Hierarchy (BVH) tree acceleration.
 */
struct BvhNode {
    double bbox_min[3];
    double bbox_max[3];
    int left_child = -1;
    int right_child = -1;
    int triangle_start = -1;
    int triangle_count = 0;

    bool is_leaf() const { return triangle_count > 0; }
};

/**
 * @class CadSdfEngine
 * @brief High-performance, zero-dependency 3D CAD mesh parser and SAH BVH Level-Set engine.
 */
class CadSdfEngine {
public:
    CadSdfEngine() = default;

    /**
     * @brief Load 3D CAD surface mesh from file (.stl or .obj). Automatically detects ASCII vs Binary STL.
     * @param filepath Path to CAD surface mesh file
     * @return True if loaded successfully, false otherwise
     */
    bool load_mesh(const std::string& filepath);

    /**
     * @brief Load ASCII or Binary STL file.
     */
    bool load_stl(const std::string& filepath);

    /**
     * @brief Load Wavefront OBJ file.
     */
    bool load_obj(const std::string& filepath);

    /**
     * @brief Load 2D boundary polygon CSV file (x, y or x, y, z).
     */
    bool load_csv(const std::string& filepath);

    /**
     * @brief Apply geometric transformations (scale, translation, Euler rotations in degrees).
     */
    void transform(double scale, double translate_x, double translate_y, double translate_z,
                   double pitch_deg, double yaw_deg, double roll_deg);

    /**
     * @brief Build SAH (Surface Area Heuristic) Bounding Volume Hierarchy (BVH) tree acceleration.
     * @param max_leaf_triangles Target maximum number of triangles per leaf node (default 8)
     */
    void build_bvh(int max_leaf_triangles = 8);

    /**
     * @brief Query Signed Distance Function phi(p) and unit surface normal n at query point p.
     * Thread-safe for OpenMP multi-threading.
     * @param p Query point coordinates [x, y, z]
     * @param normal Optional output array to receive unit surface normal vector [nx, ny, nz]
     * @return Signed distance value (negative inside solid body, positive outside in fluid)
     */
    double query_sdf(const double p[3], double normal[3] = nullptr) const;

    /**
     * @brief Get total triangle facet count.
     */
    size_t get_triangle_count() const { return triangles_.size(); }

    /**
     * @brief Clear all mesh data and BVH tree.
     */
    void clear();

private:
    std::vector<Triangle> triangles_;
    std::vector<BvhNode>  bvh_nodes_;
    int root_idx_ = 0;

    void build_bvh_recursive(int node_idx, std::vector<int>& tri_indices, int max_leaf_triangles, int depth, std::vector<Triangle>& reordered_triangles);

    void query_bvh_recursive(int node_idx, const double p[3],
                             double& min_sq_dist, double closest_pt[3],
                             int& closest_tri_idx) const;

    bool ray_triangle_intersection(const double orig[3], const double dir[3],
                                   const Triangle& tri, double& t) const;

    int compute_ray_intersections(const double p[3], const double dir[3]) const;
};

} // namespace fr::geometry


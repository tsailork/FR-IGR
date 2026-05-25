/**
 * @file ib.hpp
 * @brief Declarations of Immersed Boundary (IB) shapes, masks, and helper functions.
 */

#pragma once

#include <string>
#include <vector>

namespace ImmersedBoundary {

/**
 * @struct QuadShape
 * @brief Represents a convex quadrilateral solid boundary.
 *
 * Defined by 4 ordered vertices. Can be oriented either clockwise (CW) or
 * counter-clockwise (CCW).
 */
struct QuadShape {
    double x[4]; ///< X coordinates of the 4 vertices
    double y[4]; ///< Y coordinates of the 4 vertices
};

/**
 * @struct ParabolaShape
 * @brief Represents a dynamic line or parabola solid boundary.
 *
 * Defined as f(xi) = a*xi^2 + b*xi + c along X or Y, where coefficients
 * interpolate linearly with dynamic time parameter q:
 *   a(q) = (1-q)*a0 + q*a1
 *   b(q) = (1-q)*b0 + q*b1
 *   c(q) = (1-q)*c0 + q*c1
 */
struct ParabolaShape {
    char dir;           ///< Coordinate alignment: 'Y' for y = f(x), 'X' for x = f(y)
    double a0, b0, c0;  ///< Coefficients [a, b, c] at state q = 0
    double a1, b1, c1;  ///< Coefficients [a, b, c] at state q = 1
    std::string side;   ///< Masked region: "ABOVE", "BELOW", "LEFT", "RIGHT"
};

/**
 * @brief Calculates the signed distance from a query point (x, y) to a convex quadrilateral.
 *
 * Inside is negative, outside is positive.
 *
 * @param px Query point X coordinate
 * @param py Query point Y coordinate
 * @param quad The quadrilateral shape
 * @return Signed distance value
 */
double get_quad_sdf(double px, double py, const QuadShape& quad);

/**
 * @brief Calculates the signed distance from a query point (x, y) to a line or parabola.
 *
 * Inside (masked side) is negative, outside (fluid side) is positive.
 *
 * @param px Query point X coordinate
 * @param py Query point Y coordinate
 * @param poly The line or parabola shape
 * @param q The dynamic time parameter q in [0, 1]
 * @return Signed distance value
 */
double get_parabola_sdf(double px, double py, const ParabolaShape& poly, double q);

/**
 * @brief Computes the mask value chi (solid indicator) for a cylinder/circle shape.
 *
 * @param x Physical X coordinate of the query point
 * @param y Physical Y coordinate of the query point
 * @param center_x Circle center X coordinate
 * @param center_y Circle center Y coordinate
 * @param radius Circle radius
 * @param sharp True if using sharp mask, false if using smoothed Heaviside
 * @param smooth_width Interface smoothing width factor (in units of grid spacing)
 * @param dx Element size in X
 * @param dy Element size in Y
 * @return Indicator value chi in [0, 1] (1 = fully inside solid, 0 = fully in fluid)
 */
double compute_circle_mask(double x, double y, double center_x, double center_y, double radius,
                           bool sharp, double smooth_width, double dx, double dy);

/**
 * @brief Computes the mask value chi (solid indicator) for a NACA 4-digit airfoil.
 *
 * @param x Physical X coordinate of the query point
 * @param y Physical Y coordinate of the query point
 * @param x_le Airfoil leading edge X coordinate
 * @param y_le Airfoil leading edge Y coordinate
 * @param chord Airfoil chord length
 * @param naca_code NACA 4-digit code (e.g. "0012")
 * @param aoa_deg Angle of attack in degrees
 * @param sharp True if using sharp mask, false if using smoothed Heaviside
 * @param smooth_width Interface smoothing width factor (in units of grid spacing)
 * @param dx Element size in X
 * @param dy Element size in Y
 * @return Indicator value chi in [0, 1] (1 = fully inside solid, 0 = fully in fluid)
 */
double compute_naca_mask(double x, double y, double x_le, double y_le, double chord,
                         const std::string& naca_code, double aoa_deg,
                         bool sharp, double smooth_width, double dx, double dy);

} // namespace ImmersedBoundary

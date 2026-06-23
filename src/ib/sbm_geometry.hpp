/**
 * @file sbm_geometry.hpp
 * @brief Geometry definitions and ray-casting utilities for the Shifted Boundary Method (SBM).
 *
 * Defines structures for donor point stencils, surrogate boundary faces, and 
 * geometrical diagnostics. Orchestrates the construction of 1D normal rays and Lagrange 
 * extrapolation coefficients to support high-order boundary conditions on surrogate Cartesian faces.
 *
 * @see ImmersedBoundary::initialize_sbm_geometry
 * @see ImmersedBoundary::compute_sbm_state
 */

#pragma once

#include <vector>

/**
 * @class Solver
 * @brief Forward declaration of the main Solver class to break dependency cycles.
 */
class Solver; 

namespace ImmersedBoundary {

/**
 * @struct DonorPoint
 * @brief Represents a sampled interpolation node inside the background fluid grid for 1D SBM ray reconstruction.
 */
struct DonorPoint {
    int b_id;       ///< Block ID containing this donor point.
    int ey;         ///< Element Y coordinate index within the block.
    int ex;         ///< Element X coordinate index within the block.
    double xi;      ///< Local reference coordinate \f$\xi \in [-1, 1]\f$ within the element.
    double eta;     ///< Local reference coordinate \f$\eta \in [-1, 1]\f$ within the element.
};

/**
 * @struct SurrogateFluxPoint
 * @brief Represents a single flux integration node located on a surrogate interface face.
 *
 * Contains pre-computed distance, normal vector, and extrapolation stencil mapping 
 * the surrogate face node to the physical boundary point and outward fluid donor points.
 */
struct SurrogateFluxPoint {
    int block_id;            ///< Block ID of the surrogate face element.
    int element_x;           ///< Element X index of the surrogate face element.
    int element_y;           ///< Element Y index of the surrogate face element.
    int face_idx;            ///< Face index (0: Left, 1: Right, 2: Bottom, 3: Top).
    int node_idx;            ///< Solution point index along the face (0 to P).
    
    double D;                ///< Physical distance from this surrogate point to the true boundary.
    double nx_true;          ///< Component x of the unit normal vector at the physical boundary (pointing into the fluid).
    double ny_true;          ///< Component y of the unit normal vector at the physical boundary (pointing into the fluid).
    
    std::vector<DonorPoint> donor_points; ///< Vector of donor points sampled along the 1D normal ray.
    std::vector<double> l_weights;        ///< Interpolation/extrapolation weights of the 1D Lagrange polynomial.
};

/**
 * @struct SBMDiagnostics
 * @brief Diagnostic counters and parameters to assess SBM extrapolation quality.
 *
 * Used to track the Lebesgue constant (interpolation stability), the number of positivity limiters 
 * triggered, and geometric ratios to detect elements with poor layout.
 */
struct SBMDiagnostics {
    double max_lebesgue = 0.0;     ///< Maximum Lebesgue constant (sum of absolute Lagrange weights) across all rays.
    int limiter_count = 0;         ///< Cumulative count of positivity limiter triggers on extrapolated SBM boundary states.
    double max_dist_ratio = 0.0;   ///< Maximum ratio of distance D to the donor offset L.
    double max_d_dl_ratio = 0.0;   ///< Maximum ratio of distance D to the donor interval length dL.
};

extern SBMDiagnostics current_sbm_diags; ///< Global SBM diagnostic tracking instance.

/**
 * @brief Reset the active SBM diagnostics structure.
 *
 * Wipes cumulative counters but preserves maximum historic scaling ratios to avoid losing peak data.
 */
void reset_sbm_diagnostics();

/**
 * @brief Retrieve a copy of the current SBM diagnostics.
 *
 * @return Copy of the active SBMDiagnostics tracker.
 */
SBMDiagnostics get_sbm_diagnostics();

extern std::vector<SurrogateFluxPoint> sbm_registry; ///< Active registry containing all registered surrogate boundary points.

/**
 * @brief Detects shifted faces, constructs 1D rays, and computes surrogate boundary metadata.
 * Outputs initial geometric diagnostics.
 *
 * @param[in,out] solver Reference to the main solver instance
 */
void initialize_sbm_geometry(Solver& solver);

/**
 * @brief Retrieve a SurrogateFluxPoint if it exists for the given face and node.
 * 
 * @param[in] b_id Block ID of the target face.
 * @param[in] ey Element Y index of the target face.
 * @param[in] ex Element X index of the target face.
 * @param[in] face Face direction index (0 to 3).
 * @param[in] node Nodal index along the face (0 to P).
 * @return Pointer to SurrogateFluxPoint, or nullptr if not a shifted face.
 */
const SurrogateFluxPoint* get_sbm_face(int b_id, int ey, int ex, int face, int node);

/**
 * @brief Computes the shifted boundary state u_sb for the given surrogate point.
 *
 * Interpolates fluid states from donor points and extrapolates them to the surrogate boundary, 
 * applying the physical boundary conditions (e.g. wall slip/no-slip) and positivity limits.
 *
 * @param[in] solver Reference to the solver instance.
 * @param[in] sfp Pointer to the surrogate flux point descriptor.
 * @param[out] u_sb Array containing the calculated boundary state conservative variables \f$[\rho, \rho u, \rho v, E]\f$.
 */
void compute_sbm_state(const Solver& solver, const SurrogateFluxPoint* sfp, double u_sb[4]);

} // namespace ImmersedBoundary

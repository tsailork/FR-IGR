#pragma once

#include <vector>

class Solver; // Forward declaration

namespace ImmersedBoundary {

/**
 * @brief Represents a sampled point inside the background fluid grid for 1D ray reconstruction.
 */
struct DonorPoint {
    int b_id;       ///< Block ID
    int ey;         ///< Element Y index
    int ex;         ///< Element X index
    double xi;      ///< Local reference coordinate xi in [-1, 1]
    double eta;     ///< Local reference coordinate eta in [-1, 1]
};

/**
 * @brief Represents a single flux integration point on a surrogate interface face.
 */
struct SurrogateFluxPoint {
    int block_id;
    int element_x;
    int element_y;
    int face_idx;            ///< 0: Left, 1: Right, 2: Bottom, 3: Top
    int node_idx;            ///< Solution point index along the face (0 to P)
    
    double D;                ///< Physical distance from this surrogate point to the true boundary
    double nx_true, ny_true; ///< Normal vector at the physical boundary, pointing into the fluid
    
    std::vector<DonorPoint> donor_points; ///< P points sampled along the 1D normal ray
    std::vector<double> l_weights;        ///< P+1 weights of the 1D Lagrange polynomial evaluated at the surrogate face (index 0 = physical boundary)
};

extern std::vector<SurrogateFluxPoint> sbm_registry;

/**
 * @brief Detects shifted faces, constructs 1D rays, and computes surrogate boundary metadata.
 * Outputs initial geometric diagnostics.
 *
 * @param solver Reference to the main solver instance
 */
void initialize_sbm_geometry(Solver& solver);

/**
 * @brief Retrieve a SurrogateFluxPoint if it exists for the given face and node.
 * 
 * @return Pointer to SurrogateFluxPoint, or nullptr if not a shifted face.
 */
const SurrogateFluxPoint* get_sbm_face(int b_id, int ey, int ex, int face, int node);

/**
 * @brief Computes the shifted boundary state u_sb for the given surrogate point.
 */
void compute_sbm_state(const Solver& solver, const SurrogateFluxPoint* sfp, double u_sb[4]);

} // namespace ImmersedBoundary

/**
 * @file ib_gcm.hpp
 * @brief Declarations for High-Order Ghost-Cell Method for Flux Reconstruction (GCM-FR).
 */

#pragma once

#include <vector>
#include <array>
#include <cstddef>

template<int Dim> class SolverDim;
using Solver = SolverDim<2>;

namespace fr::ib {

/**
 * @struct GhostNodeInfo
 * @brief Metadata for an individual ghost solution node inside a cut element.
 */
struct GhostNodeInfo {
    int cell_idx = -1;       ///< Flat cell index in solver.cells
    int iy = 0;              ///< Nodal row index (0 .. N_PTS-1)
    int ix = 0;              ///< Nodal column index (0 .. N_PTS-1)
    double ghost_pos[3];     ///< Physical coordinates of ghost solution node (x, y, z)
    double normal[3];        ///< Unit surface normal vector n pointing into fluid
    double dist = 0.0;       ///< Magnitude of signed distance |phi| (<= 0)
    std::vector<std::array<double, 3>> probe_pts; ///< K = P+1 normal probing locations in fluid
};

/**
 * @class GhostNodeIbSolver
 * @brief High-Order Ghost-Cell Method for Flux Reconstruction (GCM-FR).
 */
class GhostNodeIbSolver {
public:
    GhostNodeIbSolver() = default;

    /**
     * @brief Identify solid ghost nodes (phi <= 0) and establish normal probe locations in fluid.
     * @param solver Reference to FR-IGR Solver instance
     * @param time Current simulation time
     */
    void update_geometry_and_masks(Solver& solver, double time);

    /**
     * @brief Extrapolate fluid states along normal probe lines and set high-order ghost nodal states.
     * @param solver Reference to FR-IGR Solver instance
     * @param time Current simulation time
     */
    void apply_ghost_nodal_states(Solver& solver, double time);

    /**
     * @brief Apply Legendre modal interface damping to IB boundary-intersecting elements to prevent high-order aliasing blowup.
     * @param solver Reference to FR-IGR Solver instance
     */
    void apply_ib_interface_damping(Solver& solver);

    /**
     * @brief Handle freshly cleared fluid nodes (phi^n <= 0 -> phi^{n+1} > 0) when body moves across stationary grid.
     * @param solver Reference to FR-IGR Solver instance
     */
    void handle_freshly_cleared_nodes(Solver& solver);

    /**
     * @brief Get active count of ghost solution nodes.
     */
    size_t get_ghost_node_count() const { return ghost_nodes_.size(); }

    /**
     * @brief Clear internal nodal registries.
     */
    void clear() { ghost_nodes_.clear(); cleared_nodes_.clear(); }

private:
    std::vector<GhostNodeInfo> ghost_nodes_;
    std::vector<GhostNodeInfo> cleared_nodes_;

    /**
     * @brief Sample fluid conservative state at arbitrary physical coordinate (x, y) using host element FR Lagrange tensor-product polynomial.
     */
    bool sample_fluid_state(const Solver& solver, const double pos[3], double state_out[5]) const;
};

} // namespace fr::ib

/**
 * @file cell.hpp
 * @brief Storage and connectivity class representing a single leaf node cell in the solver.
 */



#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "state.hpp"

/**
 * @struct NeighborInfo
 * @brief Connectivity and boundary configuration metadata for a block interface.
 *
 * Forward-declared here; the full definition remains in solver.hpp.
 */
struct NeighborInfo {
    int id = -1;                      ///< Neighbor block ID. Set to -1 if periodic boundary or physical wall.
    char face = ' ';                  ///< Target neighbor face being contacted ('L', 'R', 'B', 'T').
    bool is_wall = false;             ///< Slip wall boundary condition indicator.
    bool is_supersonic_inflow = false;  ///< Supersonic inflow boundary condition indicator.
    bool is_supersonic_outflow = false; ///< Supersonic outflow boundary condition indicator.
    bool is_characteristic = false;   ///< Characteristic Riemann invariant-based boundary indicator.
    bool is_total_pressure_comp = false; ///< Compressible total pressure boundary condition.
    bool is_total_pressure_incomp = false; ///< Incompressible total pressure boundary condition.
    bool is_static_pressure = false;   ///< Static pressure boundary condition.
    double ref_rho = 1.0;             ///< Reference density imposed at the boundary.
    double ref_u = 0.0;               ///< Reference X-velocity imposed at the boundary.
    double ref_v = 0.0;               ///< Reference Y-velocity imposed at the boundary.
    double ref_p = 1.0;               ///< Reference pressure imposed at the boundary.
    bool is_noslip_wall = false;      ///< Viscous no-slip wall boundary condition indicator.
    bool is_moving_wall = false;      ///< Viscous moving no-slip wall boundary condition indicator.
    double wall_velocity = 0.0;       ///< Tangential velocity of the moving wall (positive = rightward/upward).
    bool is_isothermal = false;       ///< Isothermal wall indicator (fixed temperature).
    double wall_temperature = 1.0;    ///< Prescribed temperature value for isothermal walls.
};

/**
 * @struct Cell
 * @brief Represents a single computational element/cell in the quadtree domain.
 */
struct Cell {
    uint64_t morton_id = 0;           ///< Unique Morton ID for space-filling curve sorting.
    int level = 0;                    ///< Refinement depth level (0 for conforming grid).
    double dx = 0.0;                  ///< Element size along the X coordinate direction.
    double dy = 0.0;                  ///< Element size along the Y coordinate direction.
    double x_min = 0.0;               ///< Minimum physical X coordinate of the cell.
    double y_min = 0.0;               ///< Minimum physical Y coordinate of the cell.
    double x_center = 0.0;            ///< Cell center X coordinate.
    double y_center = 0.0;            ///< Cell center Y coordinate.

    // Mapping back to the origin structured Block and logical indices
    int block_id = -1;                ///< Source Block ID.
    int ey = -1;                      ///< Source element Y coordinate index.
    int ex = -1;                      ///< Source element X coordinate index.

    // Solution and residual storage arrays (size: N_VARS * npts * npts)
    std::vector<double> U;            ///< Conserved variables: [v * npts * npts + iy * npts + ix]
    std::vector<double> RHS;          ///< Accumulator for flow solver explicit stage RHS.
    std::vector<double> U_accum;      ///< Accumulated multirate updates.
    std::vector<double> U_old;        ///< Saved state at the start of RK3 stage.

    // Local regularization fields (size: npts * npts)
    std::vector<double> sigma_field;  ///< Scalar entropic pressure regularization field (Sigma).
    std::vector<double> sigma_old;    ///< Saved Sigma at start of stage.
    std::vector<double> sigma_RHS;    ///< Explicit stage RHS for Parabolic IGR.
    std::vector<double> S_buf;        ///< Raw shock sensor source term field array.
    std::vector<double> qx_buf;       ///< Gradient auxiliary buffer (d_x Sigma) for Parabolic BR2.
    std::vector<double> qy_buf;       ///< Gradient auxiliary buffer (d_y Sigma) for Parabolic BR2.

    // Local gradient fields (size: N_VARS * npts * npts)
    std::vector<double> grad_Ux;      ///< Extrapolated gradient buffer d_x U.
    std::vector<double> grad_Uy;      ///< Extrapolated gradient buffer d_y U.

    // Local Immersed Boundary fields
    std::vector<double> ib_mask;      ///< Cached solid volume fraction mask (chi), size: npts * npts.
    bool solid_mask = false;          ///< True if this element is fully inside the solid.

    // Conforming neighbors: 0=Left, 1=Right, 2=Bottom, 3=Top
    Cell* neighbors[4] = {nullptr, nullptr, nullptr, nullptr};
    char neighbor_faces[4] = {' ', ' ', ' ', ' '};             ///< Faces of neighbors being contacted ('L', 'R', 'B', 'T')
    
    // Boundary conditions: if neighbors[f] is nullptr, BC is active on face f
    bool is_boundary[4] = {false, false, false, false};
    std::vector<NeighborInfo> boundary_info;                  ///< Size 4, stores BC metadata for L, R, B, T.

    // Multirate parameters
    double element_time = 0.0;
    double element_dt = 0.0;
    bool element_active = true;

    double s_min_val = 0.0;           ///< Cached minimum entropy value for the entropy limiter.
    int target_level = 0;             ///< Target refinement level.

    /**
     * @brief Construct a Cell with allocated local arrays.
     */
    Cell(int npts) {
        int n_dofs = N_VARS * npts * npts;
        int n_pts = npts * npts;

        U.resize(n_dofs, 0.0);
        RHS.resize(n_dofs, 0.0);
        U_accum.resize(n_dofs, 0.0);
        U_old.resize(n_dofs, 0.0);

        sigma_field.resize(n_pts, 0.0);
        sigma_old.resize(n_pts, 0.0);
        sigma_RHS.resize(n_pts, 0.0);
        S_buf.resize(n_pts, 0.0);
        qx_buf.resize(n_pts, 0.0);
        qy_buf.resize(n_pts, 0.0);

        grad_Ux.resize(n_dofs, 0.0);
        grad_Uy.resize(n_dofs, 0.0);
        ib_mask.resize(n_pts, 0.0);

        // Boundary info initialization
        boundary_info.resize(4);
    }

    /**
     * @brief Indexing accessor for conserved variables.
     */
    inline double& get_U(int v, int iy, int ix, int npts) {
        return U[v * npts * npts + iy * npts + ix];
    }

    /**
     * @brief Read-only indexing accessor for conserved variables.
     */
    inline double get_U(int v, int iy, int ix, int npts) const {
        return U[v * npts * npts + iy * npts + ix];
    }

    /**
     * @brief Indexing accessor for residual accumulator RHS.
     */
    inline double& get_RHS(int v, int iy, int ix, int npts) {
        return RHS[v * npts * npts + iy * npts + ix];
    }

    /**
     * @brief Read-only indexing accessor for residual accumulator RHS.
     */
    inline double get_RHS(int v, int iy, int ix, int npts) const {
        return RHS[v * npts * npts + iy * npts + ix];
    }
};

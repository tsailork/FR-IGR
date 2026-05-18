/**
 * @file solver.hpp
 * @brief Main Solver class and grid block structures for the FR-IGR simulation engine.
 *
 * Declares the core Solver class which manages multi-block grid domains, coordinates
 * numerical sweeps (inviscid and viscous), handles artificial viscosity smoothing (IGR),
 * and advances the solution in time.
 */

#pragma once

#include "basis.hpp"
#include "parameters.hpp"
#include "state.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include "../limiters/limiter_common.hpp"

/**
 * @brief Maximum polynomial degree supported by static buffers (P = 3 -> 4 solution points).
 */
constexpr int MAX_PTS = 4;

/**
 * @struct NeighborInfo
 * @brief Connectivity and boundary configuration metadata for a block interface.
 *
 * Tracks whether a face connects to another computational block or maps to a physical
 * boundary condition (e.g. wall, characteristic boundary, supersonic inlet/outlet).
 */
struct NeighborInfo {
    int id = -1;                      ///< Neighbor block ID. Set to -1 if periodic boundary or physical wall.
    char face = ' ';                  ///< Target neighbor face being contacted ('L', 'R', 'B', 'T').
    bool is_wall = false;             ///< Slip wall boundary condition indicator.
    bool is_supersonic_inflow = false;  ///< Supersonic inflow boundary condition indicator.
    bool is_supersonic_outflow = false; ///< Supersonic outflow boundary condition indicator.
    bool is_characteristic = false;   ///< Characteristic Riemann invariant-based boundary indicator.
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
 * @struct Block
 * @brief Represents an individual computational block in the multi-block domain.
 *
 * Encapsulates the conservative states, spatial dimensions, boundary conditions,
 * and solver workspace buffers required for localized operators.
 */
struct Block {
    int id;                           ///< Unique identification index of this block.
    int nx;                           ///< Number of elements in the X-direction.
    int ny;                           ///< Number of elements in the Y-direction.
    double dx;                        ///< Physical spacing/width of elements along X.
    double dy;                        ///< Physical spacing/height of elements along Y.
    double x_min;                     ///< Minimum physical X coordinate of this block's boundaries.
    double y_min;                     ///< Minimum physical Y coordinate of this block's boundaries.
    std::string bc_l;                 ///< Left face boundary condition type string.
    std::string bc_r;                 ///< Right face boundary condition type string.
    std::string bc_b;                 ///< Bottom face boundary condition type string.
    std::string bc_t;                 ///< Top face boundary condition type string.
    
    NeighborInfo ni_l;                ///< Connectivity metadata for the left block boundary.
    NeighborInfo ni_r;                ///< Connectivity metadata for the right block boundary.
    NeighborInfo ni_b;                ///< Connectivity metadata for the bottom block boundary.
    NeighborInfo ni_t;                ///< Connectivity metadata for the top block boundary.

    State U;                          ///< Conserved flow variables array: \f$ (\rho, \rho u, \rho v, E) \f$.
    State RHS;                        ///< Accumulator array for flow solver explicit stage RHS.
    std::vector<double> sigma_field;  ///< Scalar entropic pressure regularization field (\f$\Sigma\f$).
    std::vector<double> S_buf;        ///< Raw shock sensor source term field array.
    std::vector<double> sigma_xy_buf; ///< ADI intermediate horizontal-split sweep storage buffer.
    std::vector<double> sigma_yx_buf; ///< ADI intermediate vertical-split sweep storage buffer.
    std::vector<double> sigma_RHS;    ///< Explicit stage RHS array for Parabolic IGR.
    std::vector<double> qx_buf;       ///< Gradient auxiliary buffer (\f$\partial_x \Sigma\f$) for Parabolic BR2.
    std::vector<double> qy_buf;       ///< Gradient auxiliary buffer (\f$\partial_y \Sigma\f$) for Parabolic BR2.

    std::vector<double> grad_Ux;      ///< Extrapolated gradient buffer \f$\partial_x U\f$ for all 4 conservative variables.
    std::vector<double> grad_Uy;      ///< Extrapolated gradient buffer \f$\partial_y U\f$ for all 4 conservative variables.

    /**
     * @brief Construct and allocate all workspace arrays for a given block topology.
     *
     * @param config Configurations for block sizing and boundaries
     * @param npts Number of solution points per element direction
     */
    Block(const BlockConfig& config, int npts) 
        : id(config.id), nx(config.N_ELEM_X), ny(config.N_ELEM_Y),
          x_min(config.X_MIN), y_min(config.Y_MIN),
          bc_l(config.BC_L), bc_r(config.BC_R), bc_b(config.BC_B), bc_t(config.BC_T),
          U(config.N_ELEM_X, config.N_ELEM_Y, npts),
          RHS(config.N_ELEM_X, config.N_ELEM_Y, npts) 
    {
        dx = (config.X_MAX - config.X_MIN) / nx;
        dy = (config.Y_MAX - config.Y_MIN) / ny;
        
        int n_dofs = nx * ny * npts * npts;
        sigma_field.resize(n_dofs, 0.0);
        S_buf.resize(n_dofs, 0.0);
        sigma_xy_buf.resize(n_dofs, 0.0);
        sigma_yx_buf.resize(n_dofs, 0.0);
        sigma_RHS.resize(n_dofs, 0.0);
        qx_buf.resize(n_dofs, 0.0);
        qy_buf.resize(n_dofs, 0.0);
        grad_Ux.resize(N_VARS * n_dofs, 0.0);
        grad_Uy.resize(N_VARS * n_dofs, 0.0);
    }

    /**
     * @brief Map element and quadrature coordinates to a flattened 1D array index.
     *
     * @param ey Element Y coordinate index
     * @param ex Element X coordinate index
     * @param iy Solution point Y coordinate index
     * @param ix Solution point X coordinate index
     * @param npts Number of solution points per direction
     * @return Flattened array index
     */
    inline int get_flat_idx(int ey, int ex, int iy, int ix, int npts) const {
        return ey * (nx * npts * npts) + ex * (npts * npts) + iy * npts + ix;
    }
};

/**
 * @class Solver
 * @brief Central solver class governing numerical integration, multi-block communication, and limiters.
 */
class Solver {
public:
    const Parameters& p;              ///< Reference to the global solver parameters database.
    std::vector<Block> blocks;        ///< Registered element blocks partition.
    Basis  basis;                     ///< 1-D Lagrange basis mapping polynomials for reconstruction.
    Limiters::LimiterStats current_limiter_stats; ///< Thread-safe collector for current step's limiter activity.

    /**
     * @brief Construct the solver engine and initialize block layouts.
     *
     * @param params Configuration dataset
     */
    explicit Solver(const Parameters& params);

    /**
     * @brief Fetch neighbor cell or physical ghost state across X-interfaces.
     *
     * @param[in] b The current block to evaluate
     * @param[in] ey Element Y index
     * @param[in] ex Element X index
     * @param[in] iy Solution point Y index
     * @param[in] is_right True to check the right interface, false for left
     * @param[in] face_state Local reconstructed state on the interface face
     * @param[in] sig_face Local entropic pressure on the interface face
     * @param[out] neigh_state Returned neighboring/ghost state conserved variables
     * @param[out] sig_neigh Returned neighboring/ghost state entropic pressure
     */
    void get_neigh_state_x(const Block& b, int ey, int ex, int iy, bool is_right,
                           const double* face_state, double sig_face,
                           double* neigh_state, double& sig_neigh) const;

    /**
     * @brief Fetch neighbor cell or physical ghost state across Y-interfaces.
     *
     * @param[in] b The current block to evaluate
     * @param[in] ey Element Y index
     * @param[in] ex Element X index
     * @param[in] ix Solution point X index
     * @param[in] is_top True to check the top interface, false for bottom
     * @param[in] face_state Local reconstructed state on the interface face
     * @param[in] sig_face Local entropic pressure on the interface face
     * @param[out] neigh_state Returned neighboring/ghost state conserved variables
     * @param[out] sig_neigh Returned neighboring/ghost state entropic pressure
     */
    void get_neigh_state_y(const Block& b, int ey, int ex, int ix, bool is_top,
                           const double* face_state, double sig_face,
                           double* neigh_state, double& sig_neigh) const;

    /**
     * @brief Evaluate point-wise inviscid Euler fluxes at a solution node.
     *
     * Computes inviscid fluxes:
     * \f[ F(U) = [\rho u, \rho u^2 + p, \rho u v, (E+p)u]^T \f]
     * \f[ G(U) = [\rho v, \rho u v, \rho v^2 + p, (E+p)v]^T \f]
     * If IGR is enabled, incorporates the isotropic entropic pressure term (\f$\Sigma\f$) to the momentum components.
     *
     * @param[in] b Active computational block
     * @param[in] ey Element Y coordinate index
     * @param[in] ex Element X coordinate index
     * @param[in] iy Solution point Y coordinate index
     * @param[in] ix Solution point X coordinate index
     * @param[out] F Computed flux vector in X
     * @param[out] G Computed flux vector in Y
     * @param[in] sigma Entropic pressure value (\f$\Sigma\f$) at the node
     */
    void get_flux_pointwise(const Block& b, int ey, int ex, int iy, int ix,
                            double* F, double* G, double sigma) const;

    /**
     * @brief Solve the Riemann interface numerical flux problem.
     *
     * Utilizes the Rusanov (Local Lax-Friedrichs) approximate Riemann solver.
     *
     * @param[in] UL Left conserved state
     * @param[in] UR Right conserved state
     * @param[out] F_comm Reconstructed common numerical flux vector on the face interface
     * @param[in] dir Direction of sweep (0 = X direction, 1 = Y direction)
     * @param[in] sigl Entropic pressure at left face
     * @param[in] sigr Entropic pressure at right face
     */
    void solve_riemann(const double* UL, const double* UR, double* F_comm,
                       int dir, double sigl, double sigr) const;

    /**
     * @brief Perform Flux Reconstruction 1D inviscid sweep in the X coordinate direction.
     */
    void sweep_x();

    /**
     * @brief Perform Flux Reconstruction 1D inviscid sweep in the Y coordinate direction.
     */
    void sweep_y();

    /**
     * @brief Compute artificial viscosity shock sensor source terms.
     */
    void compute_sensor_source();

    /**
     * @brief Perform a single directional ADI Helmholtz pass.
     *
     * @param[in,out] b Block to sweep
     * @param[in] S Source array
     * @param[out] Out Smoothed output
     * @param[in] x_first True to sweep horizontal-first, false for vertical-first
     */
    void solve_adi_pass(Block& b, const std::vector<double>& S,
                        std::vector<double>& Out, bool x_first);

    /**
     * @brief Compute the explicit RHS terms for Parabolic BR2 IGR solver.
     */
    void compute_igr_parabolic_rhs();

    /**
     * @brief Smooth the entropic pressure using the active IGR (Elliptic ADI or Parabolic BR2).
     */
    void compute_entropic_pressure();

    /**
     * @brief Compute raw and corrected conservative variable gradients (BR2 Phase 1).
     */
    void compute_gradients();

    /**
     * @brief Perform Flux Reconstruction viscous fluxes sweep in the X coordinate direction (BR2 Phase 2).
     */
    void viscous_sweep_x();

    /**
     * @brief Perform Flux Reconstruction viscous fluxes sweep in the Y coordinate direction (BR2 Phase 2).
     */
    void viscous_sweep_y();

    /**
     * @brief Verify the physical admissibility of conserved states (no NaN/Inf density or pressure).
     */
    void check_stability() const;

    /**
     * @brief Compute dynamic timestep size (\f$\Delta t\f$) based on CFL criteria.
     *
     * @return Calculated timestep size
     */
    double compute_dt() const;

    /**
     * @brief Accumulate all spatial discretization fluxes (inviscid and viscous) into stage RHS.
     */
    void compute_rhs();

    /**
     * @brief Perform one full SSP-RK3 integration step.
     *
     * @param dt Timestep size
     */
    void step_rk3(double dt);
};

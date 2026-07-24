/**
 * @file solver.hpp
 * @brief Main Solver class and grid block structures for the FR-IGR simulation engine.
 *
 * Declares the core Solver class which manages multi-block grid domains, coordinates
 * numerical sweeps (inviscid and viscous), handles artificial viscosity smoothing (IGR),
 * and advances the solution in time using SSP-RK3. 
 *
 * @see Solver
 * @see Block
 */

#pragma once

#include "basis.hpp"
#include "parameters.hpp"
#include "state.hpp"
#include "cell.hpp"
#include "../ib/ib.hpp"
#include "../boundary/boundary.hpp"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <vector>
#include "../limiters/limiter_common.hpp"

/**
 * @brief Maximum polynomial degree supported by static buffers (P = 3 -> 4 solution points).
 */
constexpr int MAX_PTS = 4;

/**
 * @struct Block
 * @brief Represents an individual computational block in the multi-block domain.
 *
 * Encapsulates the conservative states, spatial dimensions, boundary conditions,
 * and solver workspace buffers required for localized operators.
 */
template<int Dim>
struct BlockDim;

/**
 * @struct BlockDim<2>
 * @brief Represents an individual computational block in the 2D multi-block domain.
 */
template<>
struct BlockDim<2> {
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

    /**
     * @brief Construct and allocate all workspace arrays for a given block topology.
     */
    BlockDim(const BlockConfig& config, int npts) 
        : id(config.id), nx(config.N_ELEM_X), ny(config.N_ELEM_Y),
          x_min(config.X_MIN), y_min(config.Y_MIN),
          bc_l(config.BC_L), bc_r(config.BC_R), bc_b(config.BC_B), bc_t(config.BC_T)
    {
        (void)npts;
        dx = (config.X_MAX - config.X_MIN) / nx;
        dy = (config.Y_MAX - config.Y_MIN) / ny;
    }

    mutable std::vector<double> U_data;
    double& U(int v, int, int, int, int) {
        if (U_data.empty()) U_data.resize(4, 0.0);
        return U_data[v];
    }
    double U(int v, int, int, int, int) const {
        if (U_data.empty()) return 0.0;
        return U_data[v];
    }

    inline int get_flat_idx(int ey, int ex, int iy, int ix, int npts) const {
        return ey * (nx * npts * npts) + ex * (npts * npts) + iy * npts + ix;
    }
};

/**
 * @struct BlockDim<3>
 * @brief Represents an individual computational block in the 3D multi-block domain.
 */
template<>
struct BlockDim<3> {
    int id;                           ///< Unique identification index of this block.
    int nx;                           ///< Number of elements in the X-direction.
    int ny;                           ///< Number of elements in the Y-direction.
    int nz;                           ///< Number of elements in the Z-direction.
    double dx;                        ///< Physical spacing/width of elements along X.
    double dy;                        ///< Physical spacing/height of elements along Y.
    double dz;                        ///< Physical spacing/depth of elements along Z.
    double x_min;                     ///< Minimum physical X coordinate of this block's boundaries.
    double y_min;                     ///< Minimum physical Y coordinate of this block's boundaries.
    double z_min;                     ///< Minimum physical Z coordinate of this block's boundaries.
    std::string bc_l;                 ///< Left face boundary condition type string.
    std::string bc_r;                 ///< Right face boundary condition type string.
    std::string bc_b;                 ///< Bottom face boundary condition type string.
    std::string bc_t;                 ///< Top face boundary condition type string.
    std::string bc_f;                 ///< Front face (Z_MIN) boundary condition type string.
    std::string bc_k;                 ///< Back face (Z_MAX) boundary condition type string.
    
    NeighborInfo ni_l;                ///< Connectivity metadata for the left block boundary.
    NeighborInfo ni_r;                ///< Connectivity metadata for the right block boundary.
    NeighborInfo ni_b;                ///< Connectivity metadata for the bottom block boundary.
    NeighborInfo ni_t;                ///< Connectivity metadata for the top block boundary.
    NeighborInfo ni_f;                ///< Connectivity metadata for the front block boundary.
    NeighborInfo ni_k;                ///< Connectivity metadata for the back block boundary.

    /**
     * @brief Construct and allocate all workspace arrays for a given block topology.
     */
    BlockDim(const BlockConfig& config, int npts) 
        : id(config.id), nx(config.N_ELEM_X), ny(config.N_ELEM_Y), nz(config.N_ELEM_Z),
          x_min(config.X_MIN), y_min(config.Y_MIN), z_min(config.Z_MIN),
          bc_l(config.BC_L), bc_r(config.BC_R), bc_b(config.BC_B), bc_t(config.BC_T),
          bc_f(config.BC_F), bc_k(config.BC_K)
    {
        (void)npts;
        dx = (config.X_MAX - config.X_MIN) / nx;
        dy = (config.Y_MAX - config.Y_MIN) / ny;
        dz = (config.Z_MAX - config.Z_MIN) / nz;
    }

    mutable std::vector<double> U_data;
    double& U(int v, int, int, int, int, int, int) {
        if (U_data.empty()) U_data.resize(5, 0.0);
        return U_data[v];
    }
    double U(int v, int, int, int, int, int, int) const {
        if (U_data.empty()) return 0.0;
        return U_data[v];
    }

    inline int get_flat_idx(int ez, int ey, int ex, int iz, int iy, int ix, int npts) const {
        return ez * (ny * nx * npts * npts * npts) +
               ey * (nx * npts * npts * npts) +
               ex * (npts * npts * npts) +
               iz * (npts * npts) +
               iy * npts + ix;
    }
};

using Block2D = BlockDim<2>;
using Block3D = BlockDim<3>;
using Block = Block2D;

template<int Dim>
class SolverDim;

/**
 * @class SolverDim<2>
 * @brief Central solver class governing numerical integration, multi-block communication, and limiters in 2D.
 */
template<>
class SolverDim<2> {
public:
    const Parameters& p;              ///< Reference to the global solver parameters database.
    std::vector<Block2D> blocks;      ///< Registered element blocks partition.
    double current_time = 0.0;        ///< Current simulation time tracker.
    Basis  basis;                     ///< 1-D Lagrange basis mapping polynomials for reconstruction.
    Limiters::LimiterStats current_limiter_stats; ///< Thread-safe collector for current step's limiter activity.
    mutable std::atomic<int> sbm_nonphysical_count{0}; ///< Counter for nonphysical state evaluations on SBM faces.

    // Cell-level solver storage (decoupled from blocks)
    std::vector<Cell2D*> cells;
    std::vector<std::vector<std::vector<Cell2D*>>> block_cells;

    // Global contiguous gradient buffers (allocated only when ENABLE_NS is true)
    std::vector<double> global_grad_Ux;
    std::vector<double> global_grad_Uy;

    /**
     * @brief Construct the solver engine and initialize block layouts.
     */
    explicit SolverDim(const Parameters& params);

    /**
     * @brief Destructor to safely clean up Cell pointers.
     */
    ~SolverDim();

    /**
     * @brief Allocate Cells and populate their spatial metrics and coordinates.
     */
    void initialize_cells();

    /**
     * @brief Compute the level-padded Morton ID for a cell.
     */
    uint64_t get_morton_id(int block_id, int level, uint32_t ex, uint32_t ey) const;

    /**
     * @brief Check if a cell is an ancestor of a target key.
     */
    bool is_ancestor(uint64_t ancestor_id, uint64_t key_id) const;

    /**
     * @brief Enforce a strict 2:1 refinement ratio between neighboring cells.
     */
    void enforce_21_ratio();

    /**
     * @brief Build neighbor connectivity pointers between leaf Cells.
     */
    void setup_cell_connectivity();

    /**
     * @brief Split a parent cell into 4 child cells.
     */
    void split_cell(Cell2D* parent, std::vector<Cell2D*>& new_cells);

    /**
     * @brief Merge 4 sibling cells back into 1 parent cell.
     */
    void merge_cells(const std::vector<Cell2D*>& siblings, Cell2D*& parent);

    /**
     * @brief Modify tree (splits/merges) based on target level requests.
     */
    void update_tree(const std::vector<int>& target_levels);

    /**
     * @brief Flag and trigger refinement/coarsening based on walls and manual zones.
     */
    void flag_refinement_coarsening();

    /**
     * @brief Find the active leaf cell containing a physical point in a block.
     */
    Cell2D* find_leaf_cell(int block_id, double x, double y) const;

    /**
     * @brief Fetch neighbor cell or physical ghost state for a Cell.
     */
    void get_neigh_state_cell(const Cell2D& c, int node_idx, bool is_right_or_top,
                             const double* face_state, double sig_face,
                             double* neigh_state, double& sig_neigh, int dir) const;

    /**
     * @brief Evaluate point-wise inviscid Euler fluxes at a cell's solution node.
     */
    void get_flux_pointwise_cell(const Cell2D& c, int iy, int ix,
                                 double* F, double* G, double sigma) const;

    /**
     * @brief Evaluate point-wise inviscid Euler fluxes at a solution node.
     */
    void get_flux_pointwise(const Block2D& b, int ey, int ex, int iy, int ix,
                            double* F, double* G, double sigma) const;

    /**
     * @brief Solve the Riemann interface numerical flux problem.
     */
    void solve_riemann(const double* UL, const double* UR, double* F_comm,
                       int dir, double SL = 0.0, double SR = 0.0,
                       double thetaL = 0.0, double thetaR = 0.0) const;

    /**
     * @brief Compute the complete interface flux.
     */
    void compute_interface_flux(const double* UL, const double* UR,
                                double sigL, double sigR,
                                double SL, double SR,
                                double thetaL, double thetaR,
                                int dir,
                                double* Flux_comm, double& Flux_S_comm) const;

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
     */
    void solve_adi_pass(Block2D& b, const std::vector<double>& S,
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
     * @brief Compute dynamic timestep size based on CFL criteria.
     */
    double compute_dt() const;

    /**
     * @brief Accumulate all spatial discretization fluxes into stage RHS.
     */
    void compute_rhs();

    /**
     * @brief Compute the element-average adaptive theta for PPR.
     */
    void compute_ppr_theta_avg();

    /**
     * @brief Perform one full SSP-RK3 integration step.
     */
    void step_rk3(double dt);

    /**
     * @brief Computes local stable timesteps for each individual element.
     */
    void compute_local_dt();

    // Immersed Boundary Method (VPM)
    void update_ib_mask_field(double time);
    double get_ib_mask(double x, double y, double dx, double dy) const;
    double get_ib_mask_at_time(double x, double y, double time, double dx, double dy) const;
    double get_ib_sdf_at_time(double x, double y, double time) const;
    void get_ib_sdf_gradient_at_time(double x, double y, double time, double& nx, double& ny) const;
    void apply_ib_explicit();
    void apply_ib_analytical(double dt_stage);

    std::vector<double> grad_x_buf;
    std::vector<double> grad_y_buf;
};

/**
 * @class SolverDim<3>
 * @brief Central solver class governing numerical integration, multi-block communication, and limiters in 3D.
 */
template<>
class SolverDim<3> {
public:
    const Parameters& p;              ///< Reference to the global solver parameters database.
    std::vector<Block3D> blocks;      ///< Registered element blocks partition.
    double current_time = 0.0;        ///< Current simulation time tracker.
    Basis  basis;                     ///< 1-D Lagrange basis mapping polynomials for reconstruction.
    Limiters::LimiterStats current_limiter_stats; ///< Thread-safe collector for current step's limiter activity.
    mutable std::atomic<int> sbm_nonphysical_count{0}; ///< Counter for nonphysical state evaluations on SBM faces.

    // Cell-level solver storage (decoupled from blocks)
    std::vector<Cell3D*> cells;
    std::vector<std::vector<std::vector<std::vector<Cell3D*>>>> block_cells;

    // Global contiguous gradient buffers (allocated only when ENABLE_NS is true)
    std::vector<double> global_grad_Ux;
    std::vector<double> global_grad_Uy;
    std::vector<double> global_grad_Uz;

    /**
     * @brief Construct the solver engine and initialize block layouts.
     */
    explicit SolverDim(const Parameters& params);

    /**
     * @brief Destructor to safely clean up Cell pointers.
     */
    ~SolverDim();

    /**
     * @brief Allocate Cells and populate their spatial metrics and coordinates.
     */
    void initialize_cells();

    /**
     * @brief Compute the level-padded Morton ID for a cell.
     */
    uint64_t get_morton_id(int block_id, int level, uint32_t ex, uint32_t ey, uint32_t ez) const;

    /**
     * @brief Check if a cell is an ancestor of a target key.
     */
    bool is_ancestor(uint64_t ancestor_id, uint64_t key_id) const;

    /**
     * @brief Enforce a strict 2:1 refinement ratio between neighboring cells.
     */
    void enforce_21_ratio();

    /**
     * @brief Build neighbor connectivity pointers between leaf Cells.
     */
    void setup_cell_connectivity();

    /**
     * @brief Split a parent cell into 8 child cells.
     */
    void split_cell(Cell3D* parent, std::vector<Cell3D*>& new_cells);

    /**
     * @brief Merge 8 sibling cells back into 1 parent cell.
     */
    void merge_cells(const std::vector<Cell3D*>& siblings, Cell3D*& parent);

    /**
     * @brief Modify tree (splits/merges) based on target level requests.
     */
    void update_tree(const std::vector<int>& target_levels);

    /**
     * @brief Flag and trigger refinement/coarsening based on walls and manual zones.
     */
    void flag_refinement_coarsening();

    Cell3D* find_leaf_cell(int block_id, double x, double y, double z) const;

    void get_neigh_state_cell(const Cell3D& c, int node_idx, bool is_right_or_top,
                              const double* face_state, double sig_face,
                              double* neigh_state, double& sig_neigh, int dir) const;

    void get_flux_pointwise(const Block3D& b, int ez, int ey, int ex, int iz, int iy, int ix,
                            double* F, double* G, double* H, double sigma) const;

    void get_flux_pointwise_cell(const Cell3D& c, int iz, int iy, int ix,
                                 double* F, double* G, double* H, double sigma) const;

    void solve_riemann(const double* UL, const double* UR, double* F_comm,
                       int dir, double SL = 0.0, double SR = 0.0,
                       double thetaL = 0.0, double thetaR = 0.0) const;

    void compute_interface_flux(const double* UL, const double* UR,
                                double sigL, double sigR,
                                double SL, double SR,
                                double thetaL, double thetaR,
                                int dir,
                                double* Flux_comm, double& Flux_S_comm) const;

    void sweep_x();
    void sweep_y();
    void sweep_z();

    void viscous_sweep_x();
    void viscous_sweep_y();
    void viscous_sweep_z();

    void compute_gradients();

    void check_stability() const;
    double compute_dt() const;
    void step_rk3(double dt);

    void compute_sensor_source();
    void step_parabolic_igr(double dt_stage_ratio);

    void apply_ib_explicit();
    void apply_ib_analytical(double dt_stage);

    std::vector<double> grad_x_buf;
    std::vector<double> grad_y_buf;
    std::vector<double> grad_z_buf;
};

using Solver2D = SolverDim<2>;
using Solver3D = SolverDim<3>;
using Solver = Solver2D;



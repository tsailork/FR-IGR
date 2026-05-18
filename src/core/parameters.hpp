/**
 * @file parameters.hpp
 * @brief Global simulation parameters and INI-format configuration parser.
 *
 * Defines structures to represent spatial grid topologies, point probe definitions,
 * and comprehensive solver/physics parameters parsed from standard input files.
 */

#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <string>

/**
 * @struct ProbeDef
 * @brief Definition of a spatial diagnostic point probe.
 *
 * Tracks physical flow properties at a specific 2D coordinate over time.
 */
struct ProbeDef {
    double x;              ///< Physical X coordinate of the probe location.
    double y;              ///< Physical Y coordinate of the probe location.
    std::string variable;  ///< Target variable name to probe (e.g. "Density", "Pressure", "Mach", "Sigma").
};

/**
 * @struct BlockConfig
 * @brief Input parameters defining an individual element block within a multi-block grid.
 *
 * Defines the logical dimension, physical bounds, and edge boundary conditions for a block.
 */
struct BlockConfig {
    int id;                ///< Unique identifier for the computational block.
    int N_ELEM_X;          ///< Number of elements along the X direction.
    int N_ELEM_Y;          ///< Number of elements along the Y direction.
    double X_MIN;          ///< Minimum physical X coordinate of the block.
    double X_MAX;          ///< Maximum physical X coordinate of the block.
    double Y_MIN;          ///< Minimum physical Y coordinate of the block.
    double Y_MAX;          ///< Maximum physical Y coordinate of the block.
    std::string BC_L;      ///< Boundary condition type on the left edge.
    std::string BC_R;      ///< Boundary condition type on the right edge.
    std::string BC_B;      ///< Boundary condition type on the bottom edge.
    std::string BC_T;      ///< Boundary condition type on the top edge.
};

/**
 * @class Parameters
 * @brief Global simulation database holding all options parsed from inputs.dat and domain.grid.
 */
struct Parameters {
    // -------------------------------------------------------------------------
    // Grid & Polynomial (from domain.grid)
    // -------------------------------------------------------------------------
    std::vector<BlockConfig> blocks;  ///< Sub-block configuration definitions in the multi-block grid.
    int P_DEG    = 0;                 ///< Polynomial degree within each element.
    int N_PTS    = 1;                 ///< Solution points per coordinate direction (automatically computed as P_DEG + 1).

    // -------------------------------------------------------------------------
    // Physics & Solver
    // -------------------------------------------------------------------------
    double CFL   = 0.5;               ///< Courant-Friedrichs-Lewy stability safety factor.
    double GAMMA = 1.4;               ///< Specific heat ratio for the ideal gas.
    std::string IC_TYPE = "RIEMANN_2D_C3"; ///< Target physical initial condition profile name.

    // Freestream (for IC_TYPE = FREESTREAM)
    double RHO_INF = 1.0;             ///< Reference freestream density.
    double U_INF   = 0.0;             ///< Reference freestream X-velocity.
    double V_INF   = 0.0;             ///< Reference freestream Y-velocity.
    double P_INF   = 1.0;             ///< Reference freestream pressure.

    // -------------------------------------------------------------------------
    // Navier-Stokes (Viscous Fluxes)
    // -------------------------------------------------------------------------
    bool   ENABLE_NS    = false;      ///< Toggle physical Navier-Stokes viscous fluxes.
    double RE           = 1000.0;     ///< Reynolds number for viscous flows.
    double PR           = 0.72;       ///< Prandtl number for viscous thermal transport.
    double MACH_REF     = 0.1;        ///< Reference Mach number for non-dimensionalization.
    double NS_BR2_ETA   = 1.0;        ///< Penalty parameter for viscous BR2 fluxes.

    // -------------------------------------------------------------------------
    // IGR (Isotropic Gradient Regularisation)
    // -------------------------------------------------------------------------
    bool   ENABLE_IGR         = false;    ///< Toggle Isotropic Gradient Regularization (artificial viscosity).
    double ALPHA_SCALE        = 0.5;       ///< Viscosity scale factor controlling dissipation levels.
    std::string IGR_GRADIENT_TYPE = "LOCAL";  ///< Gradient formulation ("LOCAL" or "CORRECTED").
    std::string IGR_TYPE      = "ELLIPTIC";  ///< Helmholtz solver equation mode ("ELLIPTIC" or "PARABOLIC").
    double IGR_TAU_R          = 0.1;         ///< Relaxation timescale factor for Parabolic IGR.
    double IGR_BR2_ETA        = 1.0;         ///< BR2 penalty coefficient for Parabolic IGR.
    int    IGR_SUB_ITERS      = 1;           ///< Forward-Euler sub-iterations per flow step under Parabolic IGR.

    // -------------------------------------------------------------------------
    // Time Stepping & I/O
    // -------------------------------------------------------------------------
    double T_FINAL   = 0.3;           ///< Target end time of the simulation.
    double OUTPUT_DT = 0.01;          ///< Deprecated, maps to OUTPUT_INTERVAL.
    double OUTPUT_INTERVAL  = 0.01;   ///< Periodicity of structured grid visual output snapshots (.vts).
    double RESTART_INTERVAL = 0.1;    ///< Periodicity of exact binary restart checkpoints (.vts).
    
    // New parameters for diagnostics
    double RESIDUAL_INTERVAL = 0.001; ///< Output time interval for tracking global residual norms.
    double PROBE_INTERVAL    = 0.001; ///< Output time interval for recording diagnostic point probe histories.
    double PRINT_INTERVAL    = 0.01;  ///< Terminal console logging frequency.

    // -------------------------------------------------------------------------
    // Stabilisation Limiters
    // -------------------------------------------------------------------------
    bool   ENABLE_POS_LIMITER     = false;  ///< Toggle density/pressure positivity-preserving limiter (Zhang-Shu).
    double POS_LIMITER_EPS        = 1e-10;  ///< Physical cutoff tolerance for density and pressure floors.
    bool   ENABLE_ENTROPY_LIMITER = false;  ///< Toggle high-order specific entropy minimum preservation limiter.

    // -------------------------------------------------------------------------
    // Restart
    // -------------------------------------------------------------------------
    std::string RESTART_FILE = "";    ///< Path to the XML-based multiblock file (.vtm) to resume from.
    double      RESTART_TIME = 0.0;   ///< Simulation time corresponding to the loaded restart snapshot.

    // -------------------------------------------------------------------------
    // Parallelism
    // -------------------------------------------------------------------------
    int NUM_THREADS = 1;              ///< Maximum OpenMP threads to allocate (1 = serial mode).

    // -------------------------------------------------------------------------
    // Probes
    // -------------------------------------------------------------------------
    std::vector<ProbeDef> probes;     ///< Set of spatial point probes defined in the simulation.

    // -------------------------------------------------------------------------
    // Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Helper utility to parse an INI-formatted text configuration file.
     *
     * Maps sections, keys, and values into standard map arrays:
     * \f[ \text{Config: Section} \rightarrow (\text{Key} \rightarrow \text{Value}) \f]
     *
     * @param filename Path to the INI file
     * @return Multi-level map populated with the configuration data
     */
    static std::map<std::string, std::map<std::string, std::string>> parse_ini(const std::string& filename);

    /**
     * @brief Parse block domains and physical boundaries from a grid file.
     *
     * @param filename Path to the grid coordinate config file (typically "domain.grid")
     */
    void load_domain(const std::string& filename);

    /**
     * @brief Parse all solver, numerical, and physics parameters from a configuration file.
     *
     * @param filename Path to the input config file (typically "inputs.dat")
     */
    void load_inputs(const std::string& filename);
};

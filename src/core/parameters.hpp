/**
 * @file parameters.hpp
 * @brief Global simulation parameters and INI-format configuration parser.
 *
 * Defines structures to represent spatial grid topologies, point probe definitions,
 * and comprehensive solver/physics parameters parsed from standard input files.
 * The `Parameters` struct acts as the central state registry used by all components.
 * 
 * @see Parameters
 * @see Solver
 */

#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <string>
#include "../ib/ib.hpp"


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
 *
 * This struct is passed to the core `Solver` and dictates everything from the polynomial 
 * degree \f$ P_{deg} \f$ to the choice of Immersed Boundary methods and fluid properties.
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
    bool   ENABLE_SUTHERLAND = false; ///< Enable temperature-dependent viscosity via Sutherland's Law.
    double SUTH_C       = 0.404;      ///< Non-dimensional Sutherland constant Sc = S_suth / T_ref.

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
    bool   USE_DUCROS_SWITCH  = false;       ///< Enable Ducros switch to zero out artificial viscosity in shear layers.
    bool   USE_PRESSURE_SENSOR = false;      ///< Use local pressure-jump sensor instead of density-gradient.
    bool   USE_MOMENTUM_DIV   = false;       ///< Use divergence of momentum for shock sensor.
    bool   USE_PRESSURE_SOURCE_CAP = true;   ///< Cap the sensor source term by local pressure.
    bool   USE_PRESSURE_FIELD_CAP  = true;   ///< Cap the resolved entropic pressure field by local pressure.
    double SOURCE_CAP_COEFF   = 1.0;         ///< Tuning coefficient C for pressure-bounded source capping.
    double IGR_DIVERGENCE_THRESHOLD = 1.0e99; ///< Divergence threshold under which regularizer activates (must be compressive).
    double IGR_SENSOR_THRESHOLD     = -9.0e99;///< Cutoff threshold for raw sensor magnitude to activate.
    double IGR_SUB_ITER_TOL         = 0.0;    ///< Convergence tolerance for IGR sub-iterations (0 = inactive, runs lock-step).

    // -------------------------------------------------------------------------
    // PPR (Phantom Pressure Regularization)
    // -------------------------------------------------------------------------
    bool   ENABLE_PPR         = false;    ///< Toggle Phantom Pressure Regularization.
    double PPR_THETA          = 1.0;      ///< Regularization feedback coefficient (theta).
    double PPR_C_TAU          = 0.2;      ///< Scaling coefficient for the relaxation time (C_tau).

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
    bool   ENABLE_MULTIRATE  = false; ///< Enable time-accurate multirate sub-cycling.
    int    MAX_MULTIRATE_LEVEL = 3;   ///< Maximum power-of-two level for multirate sub-cycling.

    // -------------------------------------------------------------------------
    // Stabilisation Limiters
    // -------------------------------------------------------------------------
    bool   ENABLE_POS_LIMITER     = false;  ///< Toggle density/pressure positivity-preserving limiter (Zhang-Shu).
    double POS_LIMITER_EPS        = 1e-10;  ///< Physical cutoff tolerance for density and pressure floors.
    bool   ENABLE_ENTROPY_LIMITER = false;  ///< Toggle high-order specific entropy minimum preservation limiter.
    double ENTROPY_LIMITER_EPS    = 1e-4;   ///< Offset tolerance for specific entropy limiter (relax violation check).

    // -------------------------------------------------------------------------
    // Immersed Boundary (IB)
    // -------------------------------------------------------------------------
    bool        ENABLE_IB           = false;      ///< Toggle Immersed Boundary Method.
    bool        ENABLE_IB_3C        = false;      ///< Toggle Colombo Constraint 3.c (strips opposing flags).
    bool        ENABLE_SBM_DIAGNOSTICS = false;   ///< Toggle runtime diagnostics for SBM extrapolation quality.
    double      IB_DL_SCALE         = 1.0;        ///< Scale factor for the shifted boundary 1D donor interval (dL)
    double      IB_L_SCALE          = 1.0;        ///< Scale factor for the shifted boundary 1D donor offset (L)
    std::string IB_METHOD           = "SBM"; ///< Immersed Boundary method ("SBM", "VPM_ANALYTICAL", "VPM_EXPLICIT").
    std::string IB_SHAPE            = "CIRCLE";   ///< Shape definition type (e.g. "CIRCLE", "NACA").
    std::string IB_NACA_CODE        = "0012";     ///< NACA 4-digit code (e.g. "0012", "2412").
    double      IB_AOA              = 0.0;        ///< Angle of attack in degrees for the airfoil.
    double      IB_CENTER_X         = 0.0;        ///< Circle/Cylinder center X coordinate.
    double      IB_CENTER_Y         = 0.0;        ///< Circle/Cylinder center Y coordinate.
    double      IB_RADIUS           = 0.5;        ///< Circle/Cylinder radius.
    double      IB_PENALIZATION_ETA = 1e-7;       ///< Solid permeability penalization parameter.
    double      IB_VELOCITY_X       = 0.0;        ///< Immersed solid velocity along X.
    double      IB_VELOCITY_Y       = 0.0;        ///< Immersed solid velocity along Y.
    std::string IB_THERMAL_TYPE     = "ADIABATIC"; ///< IB thermal boundary type ("ADIABATIC" or "ISOTHERMAL").
    double      IB_TEMPERATURE      = 1.0;        ///< Solid wall temperature target for isothermal.
    double      IB_CHORD            = 1.0;        ///< Reference chord length for Lift/Drag force coefficients.
    bool        IB_SHARP            = false;      ///< Use sharp solid mask (true) or smoothed Heaviside mask (false).
    double      IB_SMOOTH_WIDTH     = 1.5;        ///< Smooth interface thickness factor (in units of grid spacing).
    std::vector<ImmersedBoundary::QuadShape> ib_quads; ///< User-defined static/dynamic quadrilaterals.
    std::vector<ImmersedBoundary::ParabolaShape> ib_polys; ///< User-defined static/dynamic lines/parabolas.
    std::vector<std::pair<double, double>> ib_q_time_map; ///< Piecewise-linear checkpoints for time parameter q(t) (time, q).
    bool ib_is_dynamic = false; ///< True if any geometry is moving / dynamic.
    double evaluate_ib_q(double t) const; ///< Evaluates dynamic time parameter q(t) by interpolating time map.


    // -------------------------------------------------------------------------
    // Tree Decomposition (AMR)
    // -------------------------------------------------------------------------
    struct RefinementZone {
        std::string shape = "";
        double center_x = 0.0;
        double center_y = 0.0;
        double radius = 0.0;
        double width = 0.0;
        double height = 0.0;
        std::string naca_code = "0012";
        double aoa = 0.0;
        int target_level = 0;
        std::vector<double> poly_x;
        std::vector<double> poly_y;
    };
    int WALL_REFINEMENT_LEVEL = 0;    ///< Target refinement level for wall cells.
    int WALL_REFINEMENT_CELLS = 0;    ///< Number of cell layers to refine from the wall.
    std::vector<RefinementZone> refinement_zones; ///< Geometry refinement zones.

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

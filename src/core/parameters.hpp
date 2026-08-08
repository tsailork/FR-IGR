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
    double z = 0.0;        ///< Physical Z coordinate of the probe location.
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
    int N_ELEM_Z = 1;      ///< Number of elements along the Z direction.
    double X_MIN;          ///< Minimum physical X coordinate of the block.
    double X_MAX;          ///< Maximum physical X coordinate of the block.
    double Y_MIN;          ///< Minimum physical Y coordinate of the block.
    double Y_MAX;          ///< Maximum physical Y coordinate of the block.
    double Z_MIN = 0.0;    ///< Minimum physical Z coordinate of the block.
    double Z_MAX = 1.0;    ///< Maximum physical Z coordinate of the block.
    std::string BC_L;      ///< Boundary condition type on the left edge.
    std::string BC_R;      ///< Boundary condition type on the right edge.
    std::string BC_B;      ///< Boundary condition type on the bottom edge.
    std::string BC_T;      ///< Boundary condition type on the top edge.
    std::string BC_F = "TRANSMISSIVE"; ///< Boundary condition type on the front edge.
    std::string BC_K = "TRANSMISSIVE"; ///< Boundary condition type on the back edge.
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
    std::string RIEMANN_SOLVER = "HLLC";   ///< Unified Riemann numerical flux solver ("HLLC" or "RUSANOV").

    // Freestream (for IC_TYPE = FREESTREAM)
    double RHO_INF = 1.0;             ///< Reference freestream density.
    double U_INF   = 0.0;             ///< Reference freestream X-velocity.
    double V_INF   = 0.0;             ///< Reference freestream Y-velocity.
    double W_INF   = 0.0;             ///< Reference freestream Z-velocity.
    double P_INF   = 1.0;             ///< Reference freestream pressure.

    // Oblique Shock (for IC_TYPE = OBLIQUE_SHOCK or dynamically calculated BCs)
    double OBLIQUE_SHOCK_M        = 5.0;  ///< Inflow Mach number for oblique shock.
    double OBLIQUE_SHOCK_BETA_DEG = 32.0; ///< Oblique shock wave angle in degrees.

    // Shock-Vortex Interaction (for IC_TYPE = SHOCK_VORTEX)
    double SHOCK_VORTEX_MS = 1.2;     ///< Incident shock Mach number.
    double SHOCK_VORTEX_MV = 0.25;    ///< Vortex Mach number (strength).
    double SHOCK_VORTEX_XS = 0.5;     ///< Initial shock X location.
    double SHOCK_VORTEX_XV = 1.5;     ///< Initial vortex core X location.
    double SHOCK_VORTEX_YV = 0.0;     ///< Initial vortex core Y location.
    double SHOCK_VORTEX_RC = 0.2;     ///< Vortex core radius scale.

    // Richtmyer-Meshkov Instability (for IC_TYPE = RICHTMYER_MESHKOV or RMI)
    double RMI_MS    = 1.5;           ///< Incident shock Mach number.
    double RMI_RHO1  = 1.0;           ///< Light fluid density (unshocked region 1).
    double RMI_RHO2  = 3.0;           ///< Heavy fluid density (unshocked region 2).
    double RMI_XS    = 0.2;           ///< Initial shock X location.
    double RMI_X0    = 0.5;           ///< Mean interface X location.
    double RMI_AMP   = 0.05;          ///< Initial perturbation amplitude a0.
    double RMI_LY    = 1.0;           ///< Perturbation wavelength / domain height Ly.
    double RMI_SIGMA = 0.01;          ///< Smooth interface transition width.


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
    std::string IGR_SOLVER    = "PARABOLIC"; ///< Linear solver mode ("PARABOLIC", "PCG", "ADI").
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
    // PPR (Phantom Pressure Regularization) & APSR (Anisotropic Phantom Stress)
    // -------------------------------------------------------------------------
    bool   ENABLE_PPR                 = false;  ///< Toggle Phantom Pressure Regularization.
    bool   ENABLE_APSR                = false;  ///< Toggle Anisotropic Phantom Stress Relaxation (APSR-R).
    std::string PPR_WALL_BC           = "EQUILIBRATED"; ///< Wall BC for phantom pressure: EQUILIBRATED.
    double PPR_N_CELLS_SHOCK          = 2.5;    ///< Target shock transition width in cell units.
    double PPR_C_TAU                  = 0.25;   ///< Baseline non-equilibrium relaxation time scale ratio C_tau_base.
    double PPR_C_POS                  = 0.05;   ///< Lower positivity bound coefficient for regularized pressure.
    double PPR_C_MAX                  = 2.5;    ///< Upper bound coefficient for regularized pressure.
    bool   PPR_USE_DUCROS             = true;   // Apply Ducros vorticity shielding in shear layers.
    bool   PPR_USE_STENCIL_EXPANSION   = true;   ///< Expand theta and C_tau to 1-ring face neighbors.
    bool   PPR_USE_VON_NEUMANN_CEILING = true;   ///< Apply C_tau_vonNeumann ceiling outside shocks.
    bool   PPR_USE_LIMITER            = true;   ///< Apply phantom pressure bounds limiter.
    bool   PPR_CONSTANT_MODE          = false;  ///< Use spatially-uniform user-specified theta and C_tau everywhere.
    double PPR_CONSTANT_THETA         = 2.0;    ///< Constant coupling intensity theta when PPR_CONSTANT_MODE = true.
    double PPR_CONSTANT_C_TAU_VAL     = 0.25;   ///< Constant relaxation timescale ratio C_tau when PPR_CONSTANT_MODE = true.
    double PPR_SENSOR_NOISE_FLOOR     = 0.05;   ///< Acoustic noise floor for kinematic shock sensor.
    double PPR_SENSOR_SATURATION      = 0.50;   ///< Shock saturation threshold for kinematic shock sensor.
    double APSR_ALPHA                 = 0.02;   ///< Transverse-to-normal dissipation ratio alpha in [0, 1].
    double APSR_ETA_BR2               = 2.0;    ///< Penalty factor for APSR BR2 face gradient lifting.

    // -------------------------------------------------------------------------
    // Time Stepping & I/O
    // -------------------------------------------------------------------------
    double T_FINAL   = 0.3;           ///< Target end time of the simulation.
    double OUTPUT_DT = 0.01;          ///< Deprecated, maps to OUTPUT_INTERVAL.
    double OUTPUT_INTERVAL  = 0.01;   ///< Periodicity of structured grid visual output snapshots (.vts).
    bool   OUTPUT_DIV_ND    = false;  ///< Output the non-dimensional velocity divergence (div*).
    bool   OUTPUT_ADAPTIVE_THETA = false;  ///< Output the element-wise adaptive PPR theta field.
    double RESTART_INTERVAL = 0.1;    ///< Periodicity of exact binary restart checkpoints (.vts).
    int    PLOT_SUB_DIVISIONS = 0;    ///< Number of equidistant sub-element visualization intervals per dimension (0 = auto max(1, P_DEG)).
    bool   SMOOTH_PLOT_OUTPUTS = true;///< Apply BBCH C0 interface smoothing to exported VTK plot files (.vtu).
    
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
    std::string LIMITER_STRATEGY  = "ZHANG_SHU"; ///< Limiter strategy: "ZHANG_SHU", "BBCH", "MODAL", "HERMITE".

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
    // High-Mach High-Order Immersed Boundary (GCM-FR) & Wall-Function Inputs
    // -------------------------------------------------------------------------
    int         IB_GCM_ORDER                = -1;             ///< Polynomial order of probe extension (-1 = auto-match P_DEG, 1 = linear, 2 = quadratic, 3 = cubic).
    double      IB_GCM_PROBE_DISTANCE_SCALE = 1.0;            ///< Scaling factor C_probe for normal probe distance (h_probe = 2|phi| + C_probe * h).
    std::string IB_GCM_INTERPOLATION_SCHEME = "LAGRANGE_TENSOR"; ///< Interpolation scheme: "LAGRANGE_TENSOR", "HERMITE", "INVERSE_DISTANCE_WEIGHTED".
    bool        IB_GCM_SHOCK_DAMPING        = true;           ///< Enable adaptive shock sensor damping on ghost states near strong shocks.

    // Compressible Off-Body Law-of-the-Wall Model
    bool        ENABLE_IB_WALL_FUNCTION     = false;          ///< Toggle off-body compressible law-of-the-wall model at IB boundaries.
    std::string IB_WF_TYPE                  = "VAN_DRIEST";   ///< Wall function model: "VAN_DRIEST", "MUSKER", "REICHARDT", "LOG_LAW".
    double      IB_WF_KAPPA                 = 0.41;           ///< Von Kármán constant kappa.
    double      IB_WF_B                     = 5.2;            ///< Log-law intercept constant B.
    double      IB_WF_Y_PLUS_TARGET         = 50.0;           ///< Target non-dimensional height y+ for wall-function sampling probe.
    std::string IB_WF_COUPLING              = "EFFECTIVE_SLIP_VELOCITY"; ///< Coupling method: "EFFECTIVE_SLIP_VELOCITY", "ROBIN_DERIVATIVE", "DIRECT_SHEAR_FORCE".
    int         IB_WF_MAX_ITER              = 50;             ///< Max Newton-Raphson iterations for u_tau solver.
    double      IB_WF_TOL                   = 1e-6;           ///< Newton-Raphson convergence tolerance for u_tau solver.

    // CAD SDF Level-Set Engine
    std::string IB_CAD_FILE                 = "";             ///< Path to 3D CAD surface mesh file (.stl, .obj).
    double      IB_CAD_SCALE                = 1.0;            ///< Uniform scale factor for imported CAD geometry.
    double      IB_CAD_TRANSLATE_X          = 0.0;            ///< CAD translation X offset.
    double      IB_CAD_TRANSLATE_Y          = 0.0;            ///< CAD translation Y offset.
    double      IB_CAD_TRANSLATE_Z          = 0.0;            ///< CAD translation Z offset.
    double      IB_CAD_ROTATE_PITCH         = 0.0;            ///< CAD rotation Pitch angle (degrees).
    double      IB_CAD_ROTATE_YAW           = 0.0;            ///< CAD rotation Yaw angle (degrees).
    double      IB_CAD_ROTATE_ROLL          = 0.0;            ///< CAD rotation Roll angle (degrees).
    int         IB_CAD_BVH_MAX_LEAF_TRIANGLES = 8;            ///< Max triangles per BVH leaf node.

    // Dynamic Body Motion & Stationary Grid Mechanics
    bool        ENABLE_IB_DYNAMIC_MOTION    = false;          ///< Toggle dynamic body motion on stationary grid.
    std::string IB_MOTION_TYPE              = "SINUSOIDAL";   ///< Motion type: "TRANSLATIONAL", "ROTATIONAL", "SINUSOIDAL", "PIECEWISE_MAP".
    double      IB_MOTION_AMPLITUDE_X       = 0.0;            ///< Motion oscillation amplitude X.
    double      IB_MOTION_AMPLITUDE_Y       = 0.1;            ///< Motion oscillation amplitude Y.
    double      IB_MOTION_AMPLITUDE_Z       = 0.0;            ///< Motion oscillation amplitude Z.
    double      IB_MOTION_FREQ_X            = 0.0;            ///< Motion frequency X (Hz).
    double      IB_MOTION_FREQ_Y            = 1.0;            ///< Motion frequency Y (Hz).
    double      IB_MOTION_FREQ_Z            = 0.0;            ///< Motion frequency Z (Hz).
    double      IB_MOTION_PHASE_X           = 0.0;            ///< Motion phase X (rad).
    double      IB_MOTION_PHASE_Y           = 0.0;            ///< Motion phase Y (rad).
    double      IB_MOTION_PHASE_Z           = 0.0;            ///< Motion phase Z (rad).
    bool        ENABLE_IB_CLEARED_NODE_HANDLER = true;        ///< Toggle freshly cleared fluid node re-initialization prior to state advancement.
    std::string IB_CLEARED_NODE_INTERP      = "NORMAL_PROBE"; ///< Cleared node extrapolation: "NORMAL_PROBE", "NEAREST_FLUID_AVERAGE".


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

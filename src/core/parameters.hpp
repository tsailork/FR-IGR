/// @file parameters.hpp
/// @brief Simulation parameters and input file parser.
///
/// The Parameters struct holds every user-configurable option for the FR-IGR
/// solver. Values can be loaded from standard INI-format text files.

#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <string>

struct ProbeDef {
    double x;
    double y;
    std::string variable;
};

struct Parameters {
    // -------------------------------------------------------------------------
    // Grid & Polynomial (from domain.grid)
    // -------------------------------------------------------------------------
    int N_ELEM_X = 400;       ///< Number of elements in X.
    int N_ELEM_Y = 400;       ///< Number of elements in Y.
    int P_DEG    = 0;         ///< Polynomial degree per element.
    int N_PTS    = 1;         ///< Solution points per dim (computed: P_DEG+1).

    double X_MIN = 0.0;
    double X_MAX = 1.0;
    double Y_MIN = 0.0;
    double Y_MAX = 1.0;

    std::string BC_L = "TRANSMISSIVE";
    std::string BC_R = "TRANSMISSIVE";
    std::string BC_B = "TRANSMISSIVE";
    std::string BC_T = "TRANSMISSIVE";

    // -------------------------------------------------------------------------
    // Physics & Solver
    // -------------------------------------------------------------------------
    double CFL   = 0.5;       ///< CFL number for explicit time-stepping.
    double GAMMA = 1.4;       ///< Ratio of specific heats.
    std::string IC_TYPE = "RIEMANN_2D_C3";

    // -------------------------------------------------------------------------
    // IGR (Isotropic Gradient Regularisation)
    // -------------------------------------------------------------------------
    bool   ENABLE_IGR         = false;
    double ALPHA_SCALE        = 0.5;       ///< Viscosity scaling coefficient.
    std::string IGR_GRADIENT_TYPE = "LOCAL";  ///< "LOCAL" or "CORRECTED" (BR2).
    std::string IGR_TYPE      = "ELLIPTIC";  ///< "ELLIPTIC" or "PARABOLIC".
    double IGR_TAU_R          = 0.1;         ///< Parabolic relaxation time.
    double IGR_BR2_ETA        = 1.0;         ///< BR2 penalty parameter.
    int    IGR_SUB_ITERS      = 1;           ///< Parabolic sub-iterations per flow step.

    // -------------------------------------------------------------------------
    // Time Stepping & I/O
    // -------------------------------------------------------------------------
    double T_FINAL   = 0.3;
    double OUTPUT_DT = 0.01;    ///< VTK Output snapshot interval.
    
    // New parameters for diagnostics
    double RESIDUAL_INTERVAL = 0.001;
    double PROBE_INTERVAL    = 0.001;
    double PRINT_INTERVAL    = 0.01;

    // -------------------------------------------------------------------------
    // Stabilisation Limiters
    // -------------------------------------------------------------------------
    bool   ENABLE_POS_LIMITER     = false;
    double POS_LIMITER_EPS        = 1e-10;
    bool   ENABLE_ENTROPY_LIMITER = false;

    // -------------------------------------------------------------------------
    // Restart
    // -------------------------------------------------------------------------
    std::string RESTART_FILE = "";
    double      RESTART_TIME = 0.0;

    // -------------------------------------------------------------------------
    // Parallelism
    // -------------------------------------------------------------------------
    int NUM_THREADS = 1;  ///< Number of OpenMP threads (1 = serial).

    // -------------------------------------------------------------------------
    // Probes
    // -------------------------------------------------------------------------
    std::vector<ProbeDef> probes;

    // -------------------------------------------------------------------------
    // Methods
    // -------------------------------------------------------------------------

    /// Helper to parse a single INI file into a map of section->(key->value)
    static std::map<std::string, std::map<std::string, std::string>> parse_ini(const std::string& filename);

    /// Load the domain config from domain.grid
    void load_domain(const std::string& filename);

    /// Load solver/physics parameters from inputs.dat
    void load_inputs(const std::string& filename);
};

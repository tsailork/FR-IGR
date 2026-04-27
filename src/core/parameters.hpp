/// @file parameters.hpp
/// @brief Simulation parameters and input file parser.
///
/// The Parameters struct holds every user-configurable option for the FR-IGR
/// solver.  Values can be loaded from a key=value text file (inputs.dat).
/// Any parameter not found in the file retains its default value.

#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

struct Parameters {
    // -------------------------------------------------------------------------
    // Grid & Polynomial
    // -------------------------------------------------------------------------
    int N_ELEM_X = 400;       ///< Number of elements in X.
    int N_ELEM_Y = 400;       ///< Number of elements in Y.
    int P_DEG    = 0;         ///< Polynomial degree per element.
    int N_PTS    = 1;         ///< Solution points per dim (computed: P_DEG+1).
    double CFL   = 0.5;       ///< CFL number for explicit time-stepping.
    double GAMMA = 1.4;       ///< Ratio of specific heats.

    // -------------------------------------------------------------------------
    // IGR (Isotropic Gradient Regularisation)
    // -------------------------------------------------------------------------
    bool   ENABLE_IGR         = false;
    double ALPHA_SCALE        = 0.5;       ///< Viscosity scaling coefficient.
    std::string IGR_GRADIENT_TYPE = "LOCAL";  ///< "LOCAL" or "CORRECTED" (BR2).
    std::string IGR_TYPE      = "ELLIPTIC";  ///< "ELLIPTIC" or "PARABOLIC".
    double IGR_TAU_R          = 0.1;         ///< Parabolic relaxation time.
    double IGR_BR2_ETA        = 1.0;         ///< BR2 penalty parameter.

    // -------------------------------------------------------------------------
    // Domain
    // -------------------------------------------------------------------------
    double X_MIN = 0.0;
    double X_MAX = 1.0;
    double Y_MIN = 0.0;
    double Y_MAX = 1.0;

    // -------------------------------------------------------------------------
    // Time Stepping
    // -------------------------------------------------------------------------
    double T_FINAL   = 0.3;
    double DT        = 0.0005;  ///< Initial / fallback time-step.
    double OUTPUT_DT = 0.01;    ///< Output snapshot interval.

    // -------------------------------------------------------------------------
    // Stabilisation Limiters
    // -------------------------------------------------------------------------
    bool   ENABLE_POS_LIMITER     = false;
    double POS_LIMITER_EPS        = 1e-10;
    bool   ENABLE_ENTROPY_LIMITER = false;

    // -------------------------------------------------------------------------
    // Physics & Boundaries
    // -------------------------------------------------------------------------
    std::string IC_TYPE = "RIEMANN_2D_C3";
    std::string BC_L    = "TRANSMISSIVE";
    std::string BC_R    = "TRANSMISSIVE";
    std::string BC_B    = "TRANSMISSIVE";
    std::string BC_T    = "TRANSMISSIVE";

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
    // Methods
    // -------------------------------------------------------------------------

    /// Load parameters from a key=value text file.  Lines beginning with '#'
    /// are treated as comments.  Unknown keys are silently ignored.
    void load_from_file(const std::string& filename);
};

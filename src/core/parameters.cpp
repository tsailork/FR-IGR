/// @file parameters.cpp
/// @brief Input file parser implementation.

#include "parameters.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

void Parameters::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename
                  << ". Using defaults." << std::endl;
        N_PTS = P_DEG + 1;
        return;
    }

    // --- Parse key = value pairs, stripping comments ---
    std::string line;
    std::map<std::string, std::string> kv;
    while (std::getline(file, line)) {
        size_t comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        std::stringstream ss(line);
        std::string key, eq, val;
        if (ss >> key >> eq >> val && eq == "=") {
            kv[key] = val;
        }
    }

    // --- Grid & Polynomial ---
    if (kv.count("N_ELEM_X"))  N_ELEM_X = std::stoi(kv["N_ELEM_X"]);
    if (kv.count("N_ELEM_Y"))  N_ELEM_Y = std::stoi(kv["N_ELEM_Y"]);
    if (kv.count("P_DEG"))     P_DEG    = std::stoi(kv["P_DEG"]);
    if (kv.count("CFL"))       CFL      = std::stod(kv["CFL"]);
    if (kv.count("GAMMA"))     GAMMA    = std::stod(kv["GAMMA"]);

    // --- IGR ---
    if (kv.count("ENABLE_IGR"))        ENABLE_IGR        = (std::stoi(kv["ENABLE_IGR"]) != 0);
    if (kv.count("ALPHA_SCALE"))       ALPHA_SCALE       = std::stod(kv["ALPHA_SCALE"]);
    if (kv.count("IGR_GRADIENT_TYPE")) IGR_GRADIENT_TYPE = kv["IGR_GRADIENT_TYPE"];
    if (kv.count("IGR_TYPE"))          IGR_TYPE          = kv["IGR_TYPE"];
    if (kv.count("IGR_TAU_R"))         IGR_TAU_R         = std::stod(kv["IGR_TAU_R"]);
    if (kv.count("IGR_BR2_ETA"))       IGR_BR2_ETA       = std::stod(kv["IGR_BR2_ETA"]);
    if (kv.count("IGR_SUB_ITERS"))     IGR_SUB_ITERS     = std::stoi(kv["IGR_SUB_ITERS"]);

    // --- Domain ---
    if (kv.count("X_MIN"))     X_MIN    = std::stod(kv["X_MIN"]);
    if (kv.count("X_MAX"))     X_MAX    = std::stod(kv["X_MAX"]);
    if (kv.count("Y_MIN"))     Y_MIN    = std::stod(kv["Y_MIN"]);
    if (kv.count("Y_MAX"))     Y_MAX    = std::stod(kv["Y_MAX"]);

    // --- Time ---
    if (kv.count("T_FINAL"))   T_FINAL  = std::stod(kv["T_FINAL"]);
    if (kv.count("DT"))        DT       = std::stod(kv["DT"]);
    if (kv.count("OUTPUT_DT")) OUTPUT_DT = std::stod(kv["OUTPUT_DT"]);

    // --- Stabilisation ---
    if (kv.count("ENABLE_POS_LIMITER"))     ENABLE_POS_LIMITER     = (std::stoi(kv["ENABLE_POS_LIMITER"]) != 0);
    if (kv.count("POS_LIMITER_EPS"))        POS_LIMITER_EPS        = std::stod(kv["POS_LIMITER_EPS"]);
    if (kv.count("ENABLE_ENTROPY_LIMITER")) ENABLE_ENTROPY_LIMITER = (std::stoi(kv["ENABLE_ENTROPY_LIMITER"]) != 0);

    // --- Physics & BCs ---
    if (kv.count("IC_TYPE")) IC_TYPE = kv["IC_TYPE"];
    if (kv.count("BC_L"))    BC_L    = kv["BC_L"];
    if (kv.count("BC_R"))    BC_R    = kv["BC_R"];
    if (kv.count("BC_B"))    BC_B    = kv["BC_B"];
    if (kv.count("BC_T"))    BC_T    = kv["BC_T"];

    // --- Restart ---
    if (kv.count("RESTART_FILE")) RESTART_FILE = kv["RESTART_FILE"];
    if (kv.count("RESTART_TIME")) RESTART_TIME = std::stod(kv["RESTART_TIME"]);

    // --- Parallelism ---
    if (kv.count("NUM_THREADS")) NUM_THREADS = std::stoi(kv["NUM_THREADS"]);

    // --- Derived quantities ---
    N_PTS = P_DEG + 1;

    // --- Apply OpenMP thread count ---
#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#endif

    std::cout << "Parameters loaded from " << filename << std::endl;
}

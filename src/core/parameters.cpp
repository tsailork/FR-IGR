/// @file parameters.cpp
/// @brief Input file parser implementation (INI format).

#include "parameters.hpp"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// Helper function to trim whitespace from a string
static inline void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

std::map<std::string, std::map<std::string, std::string>> Parameters::parse_ini(const std::string& filename) {
    std::map<std::string, std::map<std::string, std::string>> ini;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << std::endl;
        return ini;
    }

    std::string line, current_section = "Global";
    
    while (std::getline(file, line)) {
        // Strip comments (# or ;)
        size_t comment_pos = line.find_first_of("#;");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        trim(line);
        if (line.empty()) continue;

        // Check for section header [SectionName]
        if (line.front() == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            trim(current_section);
        } else {
            // Parse key = value
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string val = line.substr(eq_pos + 1);
                trim(key);
                trim(val);
                ini[current_section][key] = val;
            }
        }
    }
    return ini;
}

void Parameters::load_domain(const std::string& filename) {
    auto ini = parse_ini(filename);
    if (ini.empty()) return;

    // We assume a single [Grid] or [Domain] section for now, 
    // or just look globally if no section was specified.
    // To be robust, we'll check across all sections or a specific one.
    // For multiblock prep, let's look in [Block0] or [Domain]
    
    std::map<std::string, std::string> kv;
    if (ini.count("Domain")) kv = ini["Domain"];
    else if (ini.count("Block0")) kv = ini["Block0"];
    else if (ini.count("Grid")) kv = ini["Grid"];
    else kv = ini["Global"]; // fallback

    if (kv.count("N_ELEM_X")) N_ELEM_X = std::stoi(kv["N_ELEM_X"]);
    if (kv.count("N_ELEM_Y")) N_ELEM_Y = std::stoi(kv["N_ELEM_Y"]);
    if (kv.count("X_MIN"))    X_MIN    = std::stod(kv["X_MIN"]);
    if (kv.count("X_MAX"))    X_MAX    = std::stod(kv["X_MAX"]);
    if (kv.count("Y_MIN"))    Y_MIN    = std::stod(kv["Y_MIN"]);
    if (kv.count("Y_MAX"))    Y_MAX    = std::stod(kv["Y_MAX"]);
    if (kv.count("BC_L"))     BC_L     = kv["BC_L"];
    if (kv.count("BC_R"))     BC_R     = kv["BC_R"];
    if (kv.count("BC_B"))     BC_B     = kv["BC_B"];
    if (kv.count("BC_T"))     BC_T     = kv["BC_T"];

    std::cout << "Domain configuration loaded from " << filename << std::endl;
}

void Parameters::load_inputs(const std::string& filename) {
    auto ini = parse_ini(filename);
    if (ini.empty()) {
        N_PTS = P_DEG + 1;
        return;
    }

    // --- [Physics] ---
    if (ini.count("Physics")) {
        auto& kv = ini["Physics"];
        if (kv.count("GAMMA"))   GAMMA   = std::stod(kv["GAMMA"]);
        if (kv.count("IC_TYPE")) IC_TYPE = kv["IC_TYPE"];
    }

    // --- [Solver] ---
    if (ini.count("Solver")) {
        auto& kv = ini["Solver"];
        if (kv.count("P_DEG"))     P_DEG    = std::stoi(kv["P_DEG"]);
        if (kv.count("CFL"))       CFL      = std::stod(kv["CFL"]);
        if (kv.count("T_FINAL"))   T_FINAL  = std::stod(kv["T_FINAL"]);
        if (kv.count("NUM_THREADS")) NUM_THREADS = std::stoi(kv["NUM_THREADS"]);
    }

    // --- [Regularization] ---
    if (ini.count("Regularization")) {
        auto& kv = ini["Regularization"];
        if (kv.count("ENABLE_IGR"))        ENABLE_IGR        = (kv["ENABLE_IGR"] == "true" || kv["ENABLE_IGR"] == "1");
        if (kv.count("ALPHA_SCALE"))       ALPHA_SCALE       = std::stod(kv["ALPHA_SCALE"]);
        if (kv.count("IGR_GRADIENT_TYPE")) IGR_GRADIENT_TYPE = kv["IGR_GRADIENT_TYPE"];
        if (kv.count("IGR_TYPE"))          IGR_TYPE          = kv["IGR_TYPE"];
        if (kv.count("IGR_TAU_R"))         IGR_TAU_R         = std::stod(kv["IGR_TAU_R"]);
        if (kv.count("IGR_BR2_ETA"))       IGR_BR2_ETA       = std::stod(kv["IGR_BR2_ETA"]);
        if (kv.count("IGR_SUB_ITERS"))     IGR_SUB_ITERS     = std::stoi(kv["IGR_SUB_ITERS"]);
    }

    // --- [Stabilization] ---
    if (ini.count("Stabilization")) {
        auto& kv = ini["Stabilization"];
        if (kv.count("ENABLE_POS_LIMITER"))     ENABLE_POS_LIMITER     = (kv["ENABLE_POS_LIMITER"] == "true" || kv["ENABLE_POS_LIMITER"] == "1");
        if (kv.count("POS_LIMITER_EPS"))        POS_LIMITER_EPS        = std::stod(kv["POS_LIMITER_EPS"]);
        if (kv.count("ENABLE_ENTROPY_LIMITER")) ENABLE_ENTROPY_LIMITER = (kv["ENABLE_ENTROPY_LIMITER"] == "true" || kv["ENABLE_ENTROPY_LIMITER"] == "1");
    }

    // --- [IO] ---
    if (ini.count("IO")) {
        auto& kv = ini["IO"];
        if (kv.count("OUTPUT_DT"))         OUTPUT_DT         = std::stod(kv["OUTPUT_DT"]);
        if (kv.count("RESIDUAL_INTERVAL")) RESIDUAL_INTERVAL = std::stod(kv["RESIDUAL_INTERVAL"]);
        if (kv.count("PROBE_INTERVAL"))    PROBE_INTERVAL    = std::stod(kv["PROBE_INTERVAL"]);
        if (kv.count("PRINT_INTERVAL"))    PRINT_INTERVAL    = std::stod(kv["PRINT_INTERVAL"]);
        if (kv.count("RESTART_FILE"))      RESTART_FILE      = kv["RESTART_FILE"];
        if (kv.count("RESTART_TIME"))      RESTART_TIME      = std::stod(kv["RESTART_TIME"]);
    }

    // --- [Probes] ---
    if (ini.count("Probes")) {
        auto& kv = ini["Probes"];
        for (const auto& [key, val] : kv) {
            // val should be "x, y, variable"
            std::stringstream ss(val);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ',')) {
                trim(token);
                tokens.push_back(token);
            }
            if (tokens.size() == 3) {
                ProbeDef p;
                p.x = std::stod(tokens[0]);
                p.y = std::stod(tokens[1]);
                p.variable = tokens[2];
                probes.push_back(p);
            } else {
                std::cerr << "Warning: Invalid probe format for " << key << " = " << val << std::endl;
            }
        }
    }

    // --- Derived quantities ---
    N_PTS = P_DEG + 1;

    // --- Apply OpenMP thread count ---
#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#endif

    std::cout << "Solver parameters loaded from " << filename << std::endl;
}

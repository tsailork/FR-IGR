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

    blocks.clear();

    for (const auto& [section, kv] : ini) {
        if (section.find("Block") == 0) {
            BlockConfig b;
            try {
                b.id = std::stoi(section.substr(5));
            } catch (...) {
                continue; // Skip sections that don't match [BlockN]
            }

            b.N_ELEM_X = kv.count("N_ELEM_X") ? std::stoi(kv.at("N_ELEM_X")) : 0;
            b.N_ELEM_Y = kv.count("N_ELEM_Y") ? std::stoi(kv.at("N_ELEM_Y")) : 0;
            b.X_MIN    = kv.count("X_MIN")    ? std::stod(kv.at("X_MIN"))    : 0.0;
            b.X_MAX    = kv.count("X_MAX")    ? std::stod(kv.at("X_MAX"))    : 1.0;
            b.Y_MIN    = kv.count("Y_MIN")    ? std::stod(kv.at("Y_MIN"))    : 0.0;
            b.Y_MAX    = kv.count("Y_MAX")    ? std::stod(kv.at("Y_MAX"))    : 1.0;
            b.BC_L     = kv.count("BC_L")     ? kv.at("BC_L")                : "TRANSMISSIVE";
            b.BC_R     = kv.count("BC_R")     ? kv.at("BC_R")                : "TRANSMISSIVE";
            b.BC_B     = kv.count("BC_B")     ? kv.at("BC_B")                : "TRANSMISSIVE";
            b.BC_T     = kv.count("BC_T")     ? kv.at("BC_T")                : "TRANSMISSIVE";
            
            blocks.push_back(b);
        }
    }

    // Sort blocks by ID
    std::sort(blocks.begin(), blocks.end(), [](const BlockConfig& a, const BlockConfig& b){
        return a.id < b.id;
    });

    // Validation pass
    for (const auto& b : blocks) {
        if (b.N_ELEM_X <= 0 || b.N_ELEM_Y <= 0) {
            std::cerr << "[GRID ERROR] Block " << b.id << " has invalid element counts (" << b.N_ELEM_X << ", " << b.N_ELEM_Y << ")\n";
            exit(1);
        }
        if (b.X_MAX <= b.X_MIN || b.Y_MAX <= b.Y_MIN) {
            std::cerr << "[GRID ERROR] Block " << b.id << " has invalid dimensions (X: " << b.X_MIN << " to " << b.X_MAX << ", Y: " << b.Y_MIN << " to " << b.Y_MAX << ")\n";
            exit(1);
        }

        auto check_face = [&](const std::string& bc, char self_face) {
            // Only treat as block-to-block connectivity if the first character
            // before ':' is a digit (e.g., "1:L").  BC strings like
            // "WALL_NOSLIP:300" or "WALL_MOVING:1.0" are NOT connectivity.
            if (bc.find(':') != std::string::npos && !bc.empty() && std::isdigit(static_cast<unsigned char>(bc[0]))) {
                size_t sep = bc.find(':');
                int nid = std::stoi(bc.substr(0, sep));
                char nface = bc[sep + 1];

                auto it = std::find_if(blocks.begin(), blocks.end(), [nid](const BlockConfig& config) { return config.id == nid; });
                if (it == blocks.end()) {
                    std::cerr << "[GRID ERROR] Block " << b.id << " face " << self_face << " points to missing block " << nid << "\n";
                    exit(1);
                }
                const auto& nb = *it;

                std::string expected_bc = std::to_string(b.id) + ":" + self_face;
                std::string actual_bc = (nface == 'L') ? nb.BC_L : (nface == 'R') ? nb.BC_R : (nface == 'B') ? nb.BC_B : (nface == 'T') ? nb.BC_T : "";
                
                if (actual_bc != expected_bc) {
                    std::cerr << "[GRID ERROR] Asymmetric boundary condition. Block " << b.id << " face " << self_face 
                              << " points to Block " << nid << " face " << nface 
                              << ", but that face has BC '" << actual_bc << "' (expected '" << expected_bc << "')\n";
                    exit(1);
                }

                int my_elems = (self_face == 'L' || self_face == 'R') ? b.N_ELEM_Y : b.N_ELEM_X;
                int n_elems = (nface == 'L' || nface == 'R') ? nb.N_ELEM_Y : nb.N_ELEM_X;
                if (my_elems != n_elems) {
                    std::cerr << "[GRID ERROR] Element count mismatch between Block " << b.id << " (" << my_elems << " elems) and Block " 
                              << nid << " (" << n_elems << " elems) on shared interface.\n";
                    exit(1);
                }

                double my_len = (self_face == 'L' || self_face == 'R') ? (b.Y_MAX - b.Y_MIN) : (b.X_MAX - b.X_MIN);
                double n_len = (nface == 'L' || nface == 'R') ? (nb.Y_MAX - nb.Y_MIN) : (nb.X_MAX - nb.X_MIN);
                if (std::abs(my_len - n_len) > 1e-8) {
                    std::cerr << "[GRID ERROR] Physical length mismatch between Block " << b.id << " (" << my_len << ") and Block " 
                              << nid << " (" << n_len << ") on shared interface.\n";
                    exit(1);
                }
            }
        };

        check_face(b.BC_L, 'L');
        check_face(b.BC_R, 'R');
        check_face(b.BC_B, 'B');
        check_face(b.BC_T, 'T');
    }

    std::cout << "Domain configuration loaded from " << filename << " (" << blocks.size() << " blocks)" << std::endl;
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
        if (kv.count("RHO_INF")) RHO_INF = std::stod(kv["RHO_INF"]);
        if (kv.count("U_INF"))   U_INF   = std::stod(kv["U_INF"]);
        if (kv.count("V_INF"))   V_INF   = std::stod(kv["V_INF"]);
        if (kv.count("P_INF"))   P_INF   = std::stod(kv["P_INF"]);
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

    // --- [NavierStokes] ---
    if (ini.count("NavierStokes")) {
        auto& kv = ini["NavierStokes"];
        if (kv.count("ENABLE_NS"))    ENABLE_NS    = (kv["ENABLE_NS"] == "true" || kv["ENABLE_NS"] == "1");
        if (kv.count("RE"))           RE           = std::stod(kv["RE"]);
        if (kv.count("PR"))           PR           = std::stod(kv["PR"]);
        if (kv.count("MACH_REF"))     MACH_REF     = std::stod(kv["MACH_REF"]);
        if (kv.count("NS_BR2_ETA"))   NS_BR2_ETA   = std::stod(kv["NS_BR2_ETA"]);
    }

    // --- [IO] ---
    if (ini.count("IO")) {
        auto& kv = ini["IO"];
        if (kv.count("OUTPUT_DT")) {
            OUTPUT_DT = std::stod(kv["OUTPUT_DT"]);
            OUTPUT_INTERVAL = OUTPUT_DT;
        }
        if (kv.count("OUTPUT_INTERVAL"))   OUTPUT_INTERVAL  = std::stod(kv["OUTPUT_INTERVAL"]);
        if (kv.count("RESTART_INTERVAL"))  RESTART_INTERVAL = std::stod(kv["RESTART_INTERVAL"]);
        if (kv.count("RESIDUAL_INTERVAL")) RESIDUAL_INTERVAL = std::stod(kv["RESIDUAL_INTERVAL"]);
        if (kv.count("PROBE_INTERVAL"))    PROBE_INTERVAL    = std::stod(kv["PROBE_INTERVAL"]);
        if (kv.count("PRINT_INTERVAL"))    PRINT_INTERVAL    = std::stod(kv["PRINT_INTERVAL"]);
        if (kv.count("RESTART_FILE"))      RESTART_FILE      = kv["RESTART_FILE"];
        if (kv.count("RESTART_TIME"))      RESTART_TIME      = std::stod(kv["RESTART_TIME"]);
    }

    // --- [ImmersedBoundary] ---
    if (ini.count("ImmersedBoundary")) {
        auto& kv = ini["ImmersedBoundary"];
        if (kv.count("ENABLE_IB"))           ENABLE_IB           = (kv["ENABLE_IB"] == "true");
        if (kv.count("ENABLE_IB_3C"))        ENABLE_IB_3C        = (kv["ENABLE_IB_3C"] == "true");
        if (kv.count("IB_DL_SCALE"))         IB_DL_SCALE         = std::stod(kv["IB_DL_SCALE"]);
        if (kv.count("IB_METHOD"))           IB_METHOD           = kv["IB_METHOD"];
        if (kv.count("IB_SHAPE"))            IB_SHAPE            = kv["IB_SHAPE"];
        if (kv.count("IB_NACA_CODE"))        IB_NACA_CODE        = kv["IB_NACA_CODE"];
        if (kv.count("IB_AOA"))              IB_AOA              = std::stod(kv["IB_AOA"]);
        if (kv.count("IB_CENTER_X"))         IB_CENTER_X         = std::stod(kv["IB_CENTER_X"]);
        if (kv.count("IB_CENTER_Y"))         IB_CENTER_Y         = std::stod(kv["IB_CENTER_Y"]);
        if (kv.count("IB_RADIUS"))           IB_RADIUS           = std::stod(kv["IB_RADIUS"]);
        if (kv.count("IB_PENALIZATION_ETA")) IB_PENALIZATION_ETA = std::stod(kv["IB_PENALIZATION_ETA"]);
        if (kv.count("IB_VELOCITY_X"))       IB_VELOCITY_X       = std::stod(kv["IB_VELOCITY_X"]);
        if (kv.count("IB_VELOCITY_Y"))       IB_VELOCITY_Y       = std::stod(kv["IB_VELOCITY_Y"]);
        if (kv.count("IB_THERMAL_TYPE"))     IB_THERMAL_TYPE     = kv["IB_THERMAL_TYPE"];
        if (kv.count("IB_TEMPERATURE"))      IB_TEMPERATURE      = std::stod(kv["IB_TEMPERATURE"]);
        if (kv.count("IB_SHARP"))            IB_SHARP            = (kv["IB_SHARP"] == "true" || kv["IB_SHARP"] == "1");
        if (kv.count("IB_SMOOTH_WIDTH"))     IB_SMOOTH_WIDTH     = std::stod(kv["IB_SMOOTH_WIDTH"]);

        // --- Custom parser for expanded dynamic/piecewise IB ---
        ib_quads.clear();
        ib_polys.clear();
        ib_q_time_map.clear();
        ib_is_dynamic = false;

        // 1. Parse time checkpoints
        if (kv.count("IB_Q_TIME_MAP")) {
            std::stringstream ss(kv["IB_Q_TIME_MAP"]);
            std::string token;
            while (ss >> token) {
                size_t colon = token.find(':');
                if (colon != std::string::npos) {
                    double t_val = std::stod(token.substr(0, colon));
                    double q_val = std::stod(token.substr(colon + 1));
                    ib_q_time_map.push_back({t_val, q_val});
                }
            }
            if (ib_q_time_map.size() > 1) {
                // Check if q actually changes in time map to set dynamic flag
                for (size_t i = 1; i < ib_q_time_map.size(); ++i) {
                    if (std::abs(ib_q_time_map[i].second - ib_q_time_map[0].second) > 1e-12) {
                        ib_is_dynamic = true;
                    }
                }
            }
        }
        if (ib_q_time_map.empty()) {
            ib_q_time_map.push_back({0.0, 0.0});
        }

        // 2. Parse quadrilaterals
        int num_quads = 0;
        if (kv.count("IB_NUM_QUADS")) num_quads = std::stoi(kv["IB_NUM_QUADS"]);
        for (int i = 0; i < num_quads; ++i) {
            std::string key = "IB_QUAD_" + std::to_string(i);
            if (kv.count(key)) {
                std::stringstream ss(kv[key]);
                ImmersedBoundary::QuadShape q;
                bool ok = true;
                for (int j = 0; j < 4; ++j) {
                    if (!(ss >> q.x[j] >> q.y[j])) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    ib_quads.push_back(q);
                } else {
                    std::cerr << "Warning: Invalid coordinate count for Quad " << i << std::endl;
                }
            }
        }

        // 3. Parse polynomials/parabolas
        int num_polys = 0;
        if (kv.count("IB_NUM_POLYS")) num_polys = std::stoi(kv["IB_NUM_POLYS"]);
        for (int i = 0; i < num_polys; ++i) {
            std::string key = "IB_POLY_" + std::to_string(i);
            if (kv.count(key)) {
                std::stringstream ss(kv[key]);
                ImmersedBoundary::ParabolaShape p_shape;
                if (ss >> p_shape.dir >> p_shape.a0 >> p_shape.b0 >> p_shape.c0 
                       >> p_shape.a1 >> p_shape.b1 >> p_shape.c1 >> p_shape.side) {
                    ib_polys.push_back(p_shape);
                    // If start and end coefficients differ, it is dynamically moving
                    if (std::abs(p_shape.a0 - p_shape.a1) > 1e-12 ||
                        std::abs(p_shape.b0 - p_shape.b1) > 1e-12 ||
                        std::abs(p_shape.c0 - p_shape.c1) > 1e-12) {
                        ib_is_dynamic = true;
                    }
                } else {
                    std::cerr << "Warning: Invalid formatting for Poly " << i << std::endl;
                }
            }
        }
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

double Parameters::evaluate_ib_q(double t) const {
    if (ib_q_time_map.empty()) return 0.0;
    if (t <= ib_q_time_map.front().first) return ib_q_time_map.front().second;
    if (t >= ib_q_time_map.back().first) return ib_q_time_map.back().second;

    for (size_t i = 0; i < ib_q_time_map.size() - 1; ++i) {
        double t0 = ib_q_time_map[i].first;
        double t1 = ib_q_time_map[i+1].first;
        if (t >= t0 && t <= t1) {
            double q0 = ib_q_time_map[i].second;
            double q1 = ib_q_time_map[i+1].second;
            if (std::abs(t1 - t0) < 1e-12) return q0;
            return q0 + (t - t0) / (t1 - t0) * (q1 - q0);
        }
    }
    return 0.0;
}


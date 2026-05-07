/// @file restart.cpp
/// @brief Restart loader implementation.

#include "restart.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

static bool seek_data_array(std::ifstream& file, const std::string& name) {
    std::string line;
    std::string target = "Name=\"" + name + "\"";
    while (std::getline(file, line)) {
        if (line.find("DataArray") != std::string::npos &&
            line.find(target) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static std::vector<double> read_values(std::ifstream& file, int count) {
    std::vector<double> vals;
    vals.reserve(count);
    std::string line;
    while ((int)vals.size() < count && std::getline(file, line)) {
        if (line.find('<') != std::string::npos) {
            if (line.find("</DataArray>") != std::string::npos) break;
            continue;
        }
        std::istringstream iss(line);
        double v;
        while (iss >> v) vals.push_back(v);
    }
    return vals;
}

#include "../core/solver.hpp"

bool Restart::load_restart(const std::string& filename, std::vector<Block>& blocks, const Parameters& p) {
    if (blocks.empty()) return false;
    if (blocks.size() > 1) {
        std::cerr << "[RESTART] Error: Multi-block restart not yet implemented.\n";
        return false;
    }

    Block& b = blocks[0];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[RESTART] Error: Could not open " << filename << "\n";
        return false;
    }

    int nx = b.nx * p.N_PTS;
    int ny = b.ny * p.N_PTS;
    int total = nx * ny;

    const char* var_names[4] = {"rho", "rho_u", "rho_v", "rho_E"};

    for (int v = 0; v < 4; ++v) {
        file.clear();
        file.seekg(0);

        if (!seek_data_array(file, var_names[v])) {
            std::cerr << "[RESTART] Error: DataArray '" << var_names[v]
                      << "' not found in " << filename << "\n";
            return false;
        }

        std::vector<double> vals = read_values(file, total);
        if ((int)vals.size() != total) {
            std::cerr << "[RESTART] Error: Expected " << total
                      << " values for " << var_names[v] << "\n";
            return false;
        }

        int idx = 0;
        for (int J = 0; J < ny; ++J) {
            int ey = J / p.N_PTS, iy = J % p.N_PTS;
            for (int I = 0; I < nx; ++I) {
                int ex = I / p.N_PTS, ix = I % p.N_PTS;
                b.U(v, ey, ex, iy, ix) = vals[idx++];
            }
        }
    }

    std::cout << "[RESTART] Loaded state from " << filename << "\n";
    return true;
}

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

static bool load_single_block(const std::string& filename, Block& b, const Parameters& p) {
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
                      << " values for " << var_names[v] << " in " << filename << ", got " << vals.size() << "\n";
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
    return true;
}

bool Restart::load_restart(const std::string& filename, std::vector<Block>& blocks, const Parameters& p) {
    if (blocks.empty()) return false;

    bool is_vtm = (filename.length() >= 4 && filename.substr(filename.length() - 4) == ".vtm");

    if (is_vtm) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[RESTART] Error: Could not open " << filename << "\n";
            return false;
        }

        std::string dir = "";
        size_t slash = filename.find_last_of("/\\");
        if (slash != std::string::npos) {
            dir = filename.substr(0, slash + 1);
        }

        std::map<int, std::string> block_files;
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("<DataSet") != std::string::npos) {
                size_t idx_pos = line.find("index=\"");
                size_t file_pos = line.find("file=\"");
                if (idx_pos != std::string::npos && file_pos != std::string::npos) {
                    idx_pos += 7;
                    size_t idx_end = line.find("\"", idx_pos);
                    int idx = std::stoi(line.substr(idx_pos, idx_end - idx_pos));

                    file_pos += 6;
                    size_t file_end = line.find("\"", file_pos);
                    std::string vts_file = line.substr(file_pos, file_end - file_pos);

                    block_files[idx] = vts_file;
                }
            }
        }

        if (block_files.size() != blocks.size()) {
            std::cerr << "[RESTART] Error: VTM file contains " << block_files.size() 
                      << " blocks, but grid configuration expects " << blocks.size() << " blocks.\n";
            return false;
        }

        for (auto& b : blocks) {
            if (block_files.find(b.id) == block_files.end()) {
                std::cerr << "[RESTART] Error: Block ID " << b.id << " not found in VTM file.\n";
                return false;
            }
            std::string full_path = dir + block_files[b.id];
            if (!load_single_block(full_path, b, p)) {
                return false;
            }
        }

    } else {
        if (blocks.size() > 1) {
            std::cerr << "[RESTART] Error: Found " << blocks.size() << " blocks, but a single .vts file was provided for restart. Expected a .vtm file.\n";
            return false;
        }
        if (!load_single_block(filename, blocks[0], p)) {
            return false;
        }
    }

    std::cout << "[RESTART] Loaded state from " << filename << "\n";
    return true;
}

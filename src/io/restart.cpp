/**
 * @file restart.cpp
 * @brief Restart loader implementation for VTK UnstructuredGrid (.vtu) datasets.
 */

#include "restart.hpp"
#include "../core/solver.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>

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

bool Restart::load_restart(const std::string& filename, Solver& solver) {
    std::string actual_filename = filename;
    if (filename.length() >= 4 && filename.substr(filename.length() - 4) == ".vtm") {
        actual_filename = filename.substr(0, filename.length() - 4) + ".vtu";
        std::cout << "[RESTART] Redirecting legacy .vtm restart request to .vtu: " << actual_filename << "\n";
    }

    std::ifstream file(actual_filename);
    if (!file.is_open()) {
        std::cerr << "[RESTART] Error: Could not open checkpoint file " << actual_filename << "\n";
        return false;
    }

    const Parameters& p = solver.p;
    int npts = p.N_PTS;
    int npts2 = npts * npts;

    int total_points = 0;
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("NumberOfPoints=\"");
        if (pos != std::string::npos) {
            pos += 16;
            size_t end = line.find("\"", pos);
            total_points = std::stoi(line.substr(pos, end - pos));
            break;
        }
    }

    if (total_points <= 0) {
        std::cerr << "[RESTART] Error: Invalid total points " << total_points << " in checkpoint file.\n";
        return false;
    }

    // Read MortonID array (UInt64)
    file.clear();
    file.seekg(0);
    if (!seek_data_array(file, "MortonID")) {
        std::cerr << "[RESTART] Error: DataArray 'MortonID' not found in " << actual_filename << "\n";
        return false;
    }
    std::vector<uint64_t> file_morton_ids;
    file_morton_ids.reserve(total_points);
    while ((int)file_morton_ids.size() < total_points && std::getline(file, line)) {
        if (line.find('<') != std::string::npos) {
            if (line.find("</DataArray>") != std::string::npos) break;
            continue;
        }
        std::istringstream iss(line);
        uint64_t v;
        while (iss >> v) file_morton_ids.push_back(v);
    }
    if ((int)file_morton_ids.size() != total_points) {
        std::cerr << "[RESTART] Error: Expected " << total_points << " MortonID values, got " << file_morton_ids.size() << "\n";
        return false;
    }

    // Read rho, rho_u, rho_v, rho_E
    const char* var_names[4] = {"rho", "rho_u", "rho_v", "rho_E"};
    std::vector<std::vector<double>> state_vals(4);
    for (int v = 0; v < 4; ++v) {
        file.clear();
        file.seekg(0);
        if (!seek_data_array(file, var_names[v])) {
            std::cerr << "[RESTART] Error: DataArray '" << var_names[v] << "' not found in " << actual_filename << "\n";
            return false;
        }
        state_vals[v] = read_values(file, total_points);
        if ((int)state_vals[v].size() != total_points) {
            std::cerr << "[RESTART] Error: Expected " << total_points << " values for '" << var_names[v] << "', got " << state_vals[v].size() << "\n";
            return false;
        }
    }

    // Read Sigma if present and enabled
    std::vector<double> sigma_vals;
    if (p.ENABLE_IGR) {
        file.clear();
        file.seekg(0);
        if (seek_data_array(file, "Sigma")) {
            sigma_vals = read_values(file, total_points);
        }
    }

    // Match cell states by Morton ID
    std::unordered_map<uint64_t, Cell*> cell_map;
    for (Cell* c : solver.cells) {
        cell_map[c->morton_id] = c;
    }

    int matched_cells = 0;
    int total_cells_in_file = total_points / npts2;

    for (int c_vtu = 0; c_vtu < total_cells_in_file; ++c_vtu) {
        int point_offset = c_vtu * npts2;
        uint64_t mid = file_morton_ids[point_offset];

        auto it = cell_map.find(mid);
        if (it != cell_map.end()) {
            Cell* c = it->second;
            // Load states
            for (int v = 0; v < 4; ++v) {
                for (int node = 0; node < npts2; ++node) {
                    c->U[v * npts2 + node] = state_vals[v][point_offset + node];
                }
            }
            // Load Sigma if present
            if (p.ENABLE_IGR && !sigma_vals.empty()) {
                for (int node = 0; node < npts2; ++node) {
                    c->sigma_field[node] = sigma_vals[point_offset + node];
                }
            }
            matched_cells++;
        }
    }

    std::cout << "[RESTART] Successfully matched and loaded " << matched_cells << " / " << solver.cells.size()
              << " cells from checkpoint file (total cells in file: " << total_cells_in_file << ")\n";

    if (matched_cells != (int)solver.cells.size()) {
        std::cerr << "[RESTART] Warning: Cell count mismatch. Solver has " << solver.cells.size()
                  << " cells, but only " << matched_cells << " were matched and loaded from checkpoint.\n";
    }

    return true;
}

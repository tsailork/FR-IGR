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

    // Read S_field (phantom pressure transport variable) if present and enabled.
    // If ENABLE_PPR is true but the array is absent (old checkpoint or file saved
    // without PPR), we fall back to re-computing S = rho * press from the loaded
    // conserved state, which is applied per-cell in the matching loop below.
    std::vector<double> s_field_vals;
    bool s_field_from_file = false;
    if (p.ENABLE_PPR) {
        file.clear();
        file.seekg(0);
        if (seek_data_array(file, "S_field")) {
            s_field_vals = read_values(file, total_points);
            if ((int)s_field_vals.size() == total_points) {
                s_field_from_file = true;
            } else {
                std::cerr << "[RESTART] Warning: 'S_field' array has unexpected size ("
                          << s_field_vals.size() << " vs " << total_points
                          << "); will re-compute from rho*press.\n";
                s_field_vals.clear();
            }
        } else {
            std::cout << "[RESTART] 'S_field' not found in checkpoint; "
                         "initialising from rho*press (PPR equilibrium).\n";
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
            // Load conserved state
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
            // Load or compute S_field
            if (p.ENABLE_PPR) {
                if (s_field_from_file) {
                    // Restore exact tracked phantom-pressure state from checkpoint
                    for (int node = 0; node < npts2; ++node) {
                        c->S_field[node] = s_field_vals[point_offset + node];
                    }
                } else {
                    // Fallback: initialise to local equilibrium S = rho * press
                    // so the positivity limiter does not fire on every cell at step 1
                    for (int iy = 0; iy < npts; ++iy) {
                        for (int ix = 0; ix < npts; ++ix) {
                            int node = iy * npts + ix;
                            double rho  = c->U[0 * npts2 + node];
                            double rhou = c->U[1 * npts2 + node];
                            double rhov = c->U[2 * npts2 + node];
                            double E    = c->U[3 * npts2 + node];
                            double press = (p.GAMMA - 1.0) *
                                (E - 0.5 * (rhou*rhou + rhov*rhov) /
                                           std::max(p.POS_LIMITER_EPS, rho));
                            c->S_field[node] = std::max(p.POS_LIMITER_EPS,
                                                        rho * press);
                        }
                    }
                }
            }
            matched_cells++;
        }
    }

    std::cout << "[RESTART] Successfully matched and loaded " << matched_cells << " / " << solver.cells.size()
              << " cells from checkpoint file (total cells in file: " << total_cells_in_file << ")";
    if (p.ENABLE_PPR) {
        std::cout << " | S_field: " << (s_field_from_file ? "restored from file" : "re-computed (rho*press)");
    }
    std::cout << "\n";

    if (matched_cells != (int)solver.cells.size()) {
        std::cerr << "[RESTART] Warning: Cell count mismatch. Solver has " << solver.cells.size()
                  << " cells, but only " << matched_cells << " were matched and loaded from checkpoint.\n";
    }

    return true;
}

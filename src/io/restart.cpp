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
#include <cstring>

static std::vector<unsigned char> base64_decode(const std::string& in) {
    std::vector<int> T(256, -1);
    const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) T[(unsigned char)table[i]] = i;

    std::vector<unsigned char> out;
    out.reserve(in.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1 || c == '=') break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static bool seek_data_array(std::ifstream& file, const std::string& name, bool& is_binary, std::string& type_str) {
    std::string line;
    std::string target = "Name=\"" + name + "\"";
    while (std::getline(file, line)) {
        if (line.find("DataArray") != std::string::npos &&
            line.find(target) != std::string::npos) {
            is_binary = (line.find("format=\"binary\"") != std::string::npos);
            type_str = "Float32";
            if (line.find("type=\"Float64\"") != std::string::npos) type_str = "Float64";
            else if (line.find("type=\"UInt64\"") != std::string::npos) type_str = "UInt64";
            else if (line.find("type=\"Int32\"") != std::string::npos) type_str = "Int32";
            return true;
        }
    }
    return false;
}

static std::vector<double> read_values(std::ifstream& file, int count, bool is_binary, const std::string& type_str) {
    std::vector<double> vals;
    vals.reserve(count);
    if (is_binary) {
        std::string line, b64_str;
        while (std::getline(file, line)) {
            if (line.find("</DataArray>") != std::string::npos) break;
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start != std::string::npos && line[start] != '<') {
                size_t end = line.find_last_not_of(" \t\r\n");
                b64_str += line.substr(start, end - start + 1);
            }
        }
        std::vector<unsigned char> decoded = base64_decode(b64_str);
        if (decoded.size() >= 4) {
            uint32_t num_bytes;
            std::memcpy(&num_bytes, decoded.data(), sizeof(uint32_t));
            const unsigned char* payload = decoded.data() + sizeof(uint32_t);
            size_t payload_len = decoded.size() - sizeof(uint32_t);
            
            if (type_str == "Float32") {
                size_t n = payload_len / sizeof(float);
                const float* fptr = reinterpret_cast<const float*>(payload);
                for (size_t i = 0; i < n && (int)vals.size() < count; ++i) vals.push_back(fptr[i]);
            } else if (type_str == "Float64") {
                size_t n = payload_len / sizeof(double);
                const double* dptr = reinterpret_cast<const double*>(payload);
                for (size_t i = 0; i < n && (int)vals.size() < count; ++i) vals.push_back(dptr[i]);
            } else if (type_str == "UInt64") {
                size_t n = payload_len / sizeof(uint64_t);
                const uint64_t* uptr = reinterpret_cast<const uint64_t*>(payload);
                for (size_t i = 0; i < n && (int)vals.size() < count; ++i) vals.push_back((double)uptr[i]);
            }
        }
    } else {
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

    file.clear();
    file.seekg(0);
    bool is_binary = false;
    std::string type_str;
    if (!seek_data_array(file, "MortonID", is_binary, type_str)) {
        std::cerr << "[RESTART] Error: DataArray 'MortonID' not found in " << actual_filename << "\n";
        return false;
    }
    std::vector<double> file_morton_raw = read_values(file, total_points, is_binary, type_str);
    std::vector<uint64_t> file_morton_ids;
    file_morton_ids.reserve(file_morton_raw.size());
    for (double v : file_morton_raw) file_morton_ids.push_back((uint64_t)v);

    if ((int)file_morton_ids.size() != total_points) {
        std::cerr << "[RESTART] Error: Expected " << total_points << " MortonID values, got " << file_morton_ids.size() << "\n";
        return false;
    }

    const char* var_names[4] = {"rho", "rho_u", "rho_v", "rho_E"};
    std::vector<std::vector<double>> var_data(4);

    for (int v = 0; v < 4; ++v) {
        file.clear();
        file.seekg(0);
        if (!seek_data_array(file, var_names[v], is_binary, type_str)) {
            std::cerr << "[RESTART] Error: DataArray '" << var_names[v] << "' not found in " << actual_filename << "\n";
            return false;
        }
        var_data[v] = read_values(file, total_points, is_binary, type_str);
        if ((int)var_data[v].size() != total_points) {
            std::cerr << "[RESTART] Error: Expected " << total_points << " values for '" << var_names[v] << "', got " << var_data[v].size() << "\n";
            return false;
        }
    }

    std::unordered_map<uint64_t, Cell*> cell_map;
    for (Cell* c : solver.cells) {
        cell_map[c->morton_id] = c;
    }

    int matched_cells = 0;
    int file_num_cells = total_points / npts2;

    for (int fc = 0; fc < file_num_cells; ++fc) {
        int pt_start = fc * npts2;
        uint64_t mid = file_morton_ids[pt_start];
        auto it = cell_map.find(mid);
        if (it != cell_map.end()) {
            Cell* c = it->second;
            for (int v = 0; v < 4; ++v) {
                for (int pt = 0; pt < npts2; ++pt) {
                    c->get_U(v, pt / npts, pt % npts, npts) = var_data[v][pt_start + pt];
                }
            }
            matched_cells++;
        }
    }

    std::cout << "[RESTART] Successfully matched and loaded 2D state for " << matched_cells << " / " << solver.cells.size() << " cells\n";
    return (matched_cells > 0);
}

bool Restart::load_restart(const std::string& filename, SolverDim<3>& solver) {
    std::string actual_filename = filename;
    if (filename.length() >= 4 && filename.substr(filename.length() - 4) == ".vtm") {
        actual_filename = filename.substr(0, filename.length() - 4) + ".vtu";
    }

    std::ifstream file(actual_filename);
    if (!file.is_open()) {
        std::cerr << "[RESTART] Error: Could not open checkpoint file " << actual_filename << "\n";
        return false;
    }

    const Parameters& p = solver.p;
    int npts = p.N_PTS;
    int npts3 = npts * npts * npts;

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

    file.clear();
    file.seekg(0);
    bool is_binary = false;
    std::string type_str;
    if (!seek_data_array(file, "MortonID", is_binary, type_str)) {
        std::cerr << "[RESTART] Error: DataArray 'MortonID' not found in " << actual_filename << "\n";
        return false;
    }
    std::vector<double> file_morton_raw = read_values(file, total_points, is_binary, type_str);
    std::vector<uint64_t> file_morton_ids;
    file_morton_ids.reserve(file_morton_raw.size());
    for (double v : file_morton_raw) file_morton_ids.push_back((uint64_t)v);

    if ((int)file_morton_ids.size() != total_points) {
        std::cerr << "[RESTART] Error: Expected " << total_points << " MortonID values, got " << file_morton_ids.size() << "\n";
        return false;
    }

    const char* var_names[5] = {"rho", "rho_u", "rho_v", "rho_w", "rho_E"};
    std::vector<std::vector<double>> var_data(5);

    for (int v = 0; v < 5; ++v) {
        file.clear();
        file.seekg(0);
        if (!seek_data_array(file, var_names[v], is_binary, type_str)) {
            std::cerr << "[RESTART] Error: DataArray '" << var_names[v] << "' not found in " << actual_filename << "\n";
            return false;
        }
        var_data[v] = read_values(file, total_points, is_binary, type_str);
        if ((int)var_data[v].size() != total_points) {
            std::cerr << "[RESTART] Error: Expected " << total_points << " values for '" << var_names[v] << "', got " << var_data[v].size() << "\n";
            return false;
        }
    }

    std::unordered_map<uint64_t, Cell3D*> cell_map;
    for (Cell3D* c : solver.cells) {
        cell_map[c->morton_id] = c;
    }

    int matched_cells = 0;
    int file_num_cells = total_points / npts3;

    for (int fc = 0; fc < file_num_cells; ++fc) {
        int pt_start = fc * npts3;
        uint64_t mid = file_morton_ids[pt_start];
        auto it = cell_map.find(mid);
        if (it != cell_map.end()) {
            Cell3D* c = it->second;
            for (int v = 0; v < 5; ++v) {
                for (int pt = 0; pt < npts3; ++pt) {
                    c->U[v * npts3 + pt] = var_data[v][pt_start + pt];
                }
            }
            matched_cells++;
        }
    }

    std::cout << "[RESTART] Successfully matched and loaded 3D state for " << matched_cells << " / " << solver.cells.size() << " cells\n";
    return (matched_cells > 0);
}

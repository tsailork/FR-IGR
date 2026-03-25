#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <map>

struct Parameters {
    // --- SIMULATION CONFIG ---
    int N_ELEM_X = 400;
    int N_ELEM_Y = 400;
    int P_DEG    = 0;           // Polynomial Degree
    int N_PTS    = 1;           // Points per element per dim (computed)
    double CFL   = 0.5;         // Conservative CFL for IGR
    double GAMMA = 1.4;

    // --- IGR CONSTANTS ---
    bool ENABLE_IGR = false;
    double ALPHA_SCALE = 0.5; 

    // --- DOMAIN ---
    double X_MIN = 0.0;
    double X_MAX = 1.0;
    double Y_MIN = 0.0;
    double Y_MAX = 1.0;

    // --- TIME STEPPING ---
    double T_FINAL = 0.3;
    double DT = 0.0005; // Initial/Fallback DT
    double OUTPUT_DT = 0.01; // Output interval

    void load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << filename << ". Using defaults." << std::endl;
            N_PTS = P_DEG + 1;
            return;
        }

        std::string line;
        std::map<std::string, std::string> kv;
        while (std::getline(file, line)) {
            // Remove comments
            size_t comment = line.find('#');
            if (comment != std::string::npos) line = line.substr(0, comment);

            std::stringstream ss(line);
            std::string key, eq, val;
            if (ss >> key >> eq >> val && eq == "=") {
                kv[key] = val;
            }
        }

        if (kv.count("N_ELEM_X")) N_ELEM_X = std::stoi(kv["N_ELEM_X"]);
        if (kv.count("N_ELEM_Y")) N_ELEM_Y = std::stoi(kv["N_ELEM_Y"]);
        if (kv.count("P_DEG"))    P_DEG    = std::stoi(kv["P_DEG"]);
        if (kv.count("CFL"))      CFL      = std::stod(kv["CFL"]);
        if (kv.count("GAMMA"))    GAMMA    = std::stod(kv["GAMMA"]);
        if (kv.count("ENABLE_IGR")) ENABLE_IGR = (std::stoi(kv["ENABLE_IGR"]) != 0);
        if (kv.count("ALPHA_SCALE")) ALPHA_SCALE = std::stod(kv["ALPHA_SCALE"]);
        if (kv.count("X_MIN"))    X_MIN    = std::stod(kv["X_MIN"]);
        if (kv.count("X_MAX"))    X_MAX    = std::stod(kv["X_MAX"]);
        if (kv.count("Y_MIN"))    Y_MIN    = std::stod(kv["Y_MIN"]);
        if (kv.count("Y_MAX"))    Y_MAX    = std::stod(kv["Y_MAX"]);
        if (kv.count("T_FINAL"))  T_FINAL  = std::stod(kv["T_FINAL"]);
        if (kv.count("DT"))       DT       = std::stod(kv["DT"]);
        if (kv.count("OUTPUT_DT")) OUTPUT_DT = std::stod(kv["OUTPUT_DT"]);

        N_PTS = P_DEG + 1;
        std::cout << "Parameters loaded from " << filename << std::endl;
    }
};

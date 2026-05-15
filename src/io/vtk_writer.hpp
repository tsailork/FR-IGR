/// @file vtk_writer.hpp
/// @brief VTK StructuredGrid (.vts) output writer and PVD time-series manager.

#pragma once
#include "../core/solver.hpp"
#include <string>
#include <vector>

/// Create a directory (cross-platform).
void ensure_output_directory(const std::string& path);

class VTKWriter {
public:
    explicit VTKWriter(const std::string& name = "solution");
    void write_snapshot(Solver& solver, int step, double time);
    void load_existing_pvd();
private:
    std::string base_name;
    std::vector<std::pair<double, std::string>> snapshots;
    void write_pvd();
};

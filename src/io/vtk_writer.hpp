/**
 * @file vtk_writer.hpp
 * @brief VTK StructuredGrid (.vts) file generator and ParaView PVD collection manager.
 *
 * Handles the generation of multi-block structured XML datasets for ParaView visualization.
 * Supports checkpoint files (raw double-precision solution-point state data for exact restarting) and
 * high-resolution plot files (interpolated states to visualization sub-element boundaries).
 * Utilizes VTK format serialization natively without third-party dependencies.
 */

#pragma once
#include "../core/solver.hpp"
#include <string>
#include <vector>

/**
 * @brief Helper utility to safely ensure a directory exists across Windows and POSIX targets.
 *
 * @param path Relative or absolute path to the directory
 */
void ensure_output_directory(const std::string& path);

/**
 * @class VTKWriter
 * @brief Manages spatial file output streams and ParaView time-history datasets.
 *
 * Generates XML-structured multiblock grids (`.vtm` / `.vts`) and tracks history
 * in `.pvd` timelines to maintain continuous visualizations across resumes.
 */
class VTKWriter {
public:
    /**
     * @brief Construct a VTKWriter instance with a default base naming convention.
     *
     * @param name Base name of the output series (default is "solution")
     */
    explicit VTKWriter(const std::string& name = "solution");

    /**
     * @brief Write a raw restart-capable simulation checkpoint block-by-block.
     *
     * Exports exact solution values at Gauss-Legendre coordinates to enable
     * lossless double-precision restarts.
     *
     * @param[in,out] solver The active Solver context
     * @param[in] step Integer index of the current time step
     * @param[in] time Physical simulation time associated with the step
     */
    void write_checkpoint(Solver& solver, int step, double time);
    void write_checkpoint(SolverDim<3>& solver, int step, double time);

    /**
     * @brief Write a high-resolution visualization-friendly plot snapshot.
     *
     * Interpolates polynomial states to the element borders and boundaries
     * to eliminate pixel gaps and visualize internal sub-element dynamics.
     *
     * @param[in,out] solver The active Solver context
     * @param[in] step Integer index of the current plot step
     * @param[in] time Physical simulation time associated with the step
     */
    void write_plot(Solver& solver, int step, double time);
    void write_plot(SolverDim<3>& solver, int step, double time);

    /**
     * @brief Scan and parse existing `.pvd` files to restore snapshot histories.
     *
     * Enables continuous ParaView timelines across simulation restarts.
     */
    void load_existing_pvd();

private:
    std::string base_name;  ///< Prefix directory/filename string for generated outputs.
    std::vector<std::pair<double, std::string>> plot_snapshots;  ///< Registered visualization plot snapshots (time -> filename).
    std::vector<std::pair<double, std::string>> sol_snapshots;   ///< Registered checkpoint solution snapshots (time -> filename).

    /**
     * @brief Internal helper to write/update XML-based ParaView collection files (.pvd).
     *
     * @param pvd_name Target PVD filename to write
     * @param snaps Vector of registered snapshots to output in the XML timeline
     */
    void write_pvd(const std::string& pvd_name, const std::vector<std::pair<double, std::string>>& snaps);
};

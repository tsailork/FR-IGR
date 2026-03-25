#pragma once

#include "solver.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sys/stat.h> // For mkdir

// Helper to create directory
inline void ensure_output_directory(const std::string& path) {
    #ifdef _WIN32
        _mkdir(path.c_str());
    #else 
        mkdir(path.c_str(), 0777); 
    #endif
}

class VTKWriter {
private:
    std::string base_name;
    std::vector<std::pair<double, std::string>> snapshots;

    void write_pvd() {
        std::string pvd_filename = "pv_outputs/" + base_name + ".pvd";
        std::ofstream pvd(pvd_filename);
        if (!pvd.is_open()) {
            std::cerr << "Error: Could not update PVD file " << pvd_filename << std::endl;
            return;
        }

        pvd << "<?xml version=\"1.0\"?>\n";
        pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd << "  <Collection>\n";
        
        for (const auto& snap : snapshots) {
            pvd << "    <DataSet timestep=\"" << snap.first << "\" group=\"\" part=\"0\" file=\"" << snap.second << "\"/>\n";
        }
        
        pvd << "  </Collection>\n";
        pvd << "</VTKFile>\n";
        pvd.close();
    }

public:
    VTKWriter(const std::string& name = "solution") : base_name(name) {}

    void write_snapshot(Solver& solver, int step, double time) {
        const Parameters& p = solver.p;
        // Use .vts extension for XML Structured Grid
        std::string filename = "sol_" + std::to_string(step) + ".vts";
        std::string full_path = "pv_outputs/" + filename;
        
        std::ofstream vts(full_path);
        if (!vts.is_open()) {
            std::cerr << "Error: Could not open output file " << full_path << std::endl;
            return;
        }

        // 1. Grid Dimensions
        int nx = p.N_ELEM_X * p.N_PTS;
        int ny = p.N_ELEM_Y * p.N_PTS;
        
        // Write XML Header
        vts << "<?xml version=\"1.0\"?>\n";
        vts << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        // Extent is 0 to nx-1, 0 to ny-1, 0 to 0
        vts << "  <StructuredGrid WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";
        vts << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";

        // 2. Points (Coordinates)
        vts << "      <Points>\n";
        vts << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        vts << std::fixed << std::setprecision(6);

        // Map global grid indices (J, I) to solver indices (ey, iy, ex, ix)
        for (int J = 0; J < ny; ++J) {
            int ey = J / p.N_PTS;
            int iy = J % p.N_PTS;
            for (int I = 0; I < nx; ++I) {
                int ex = I / p.N_PTS;
                int ix = I % p.N_PTS;
                
                double x = p.X_MIN + (ex + 0.5*(1+solver.basis.z[ix])) * solver.dx;
                double y = p.Y_MIN + (ey + 0.5*(1+solver.basis.z[iy])) * solver.dy;
                
                vts << x << " " << y << " 0.0\n";
            }
        }
        vts << "        </DataArray>\n";
        vts << "      </Points>\n";

        // 3. Point Data
        vts << "      <PointData Scalars=\"rho\">\n";

        // Compute auxiliary fields
        std::vector<double> Sigma_Source = solver.compute_sensor_source();

        // Helper to write a DataArray
        auto write_data_array = [&](const std::string& name, auto func) {
            vts << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
            for (int J = 0; J < ny; ++J) {
                int ey = J / p.N_PTS;
                int iy = J % p.N_PTS;
                for (int I = 0; I < nx; ++I) {
                    int ex = I / p.N_PTS;
                    int ix = I % p.N_PTS;
                    vts << func(ey, ex, iy, ix) << " ";
                }
                vts << "\n";
            }
            vts << "        </DataArray>\n";
        };

        // --- Conserved Variables ---
        write_data_array("rho", [&](int ey, int ex, int iy, int ix) { return solver.U(0, ey, ex, iy, ix); });
        write_data_array("rho_u", [&](int ey, int ex, int iy, int ix) { return solver.U(1, ey, ex, iy, ix); });
        write_data_array("rho_v", [&](int ey, int ex, int iy, int ix) { return solver.U(2, ey, ex, iy, ix); });
        write_data_array("rho_E", [&](int ey, int ex, int iy, int ix) { return solver.U(3, ey, ex, iy, ix); });

        // --- Primitives ---
        auto get_prim = [&](int ey, int ex, int iy, int ix, int var) {
            double r = solver.U(0, ey, ex, iy, ix);
            double ru = solver.U(1, ey, ex, iy, ix);
            double rv = solver.U(2, ey, ex, iy, ix);
            double E = solver.U(3, ey, ex, iy, ix);
            double u = ru / r;
            double v = rv / r;
            double P = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
            
            if (var == 0) return u;
            if (var == 1) return v;
            if (var == 2) return P;
            if (var == 3) return P/r; // Temperature
            if (var == 4) { // Mach
                double c = std::sqrt(p.GAMMA * std::abs(P) / r);
                return std::sqrt(u*u + v*v) / c;
            }
            return 0.0;
        };

        write_data_array("u", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 0); });
        write_data_array("v", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 1); });
        write_data_array("Pressure", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 2); });
        write_data_array("Temperature", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 3); });
        write_data_array("Mach", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 4); });

        // --- IGR Fields ---
        write_data_array("Sigma", [&](int ey, int ex, int iy, int ix) {
            int idx = ey * (p.N_ELEM_X * p.N_PTS * p.N_PTS) + 
                      ex * (p.N_PTS * p.N_PTS) + 
                      iy * p.N_PTS + 
                      ix;
            return solver.sigma_field[idx];
        });

        write_data_array("Sigma_Source", [&](int ey, int ex, int iy, int ix) {
            int idx = ey * (p.N_ELEM_X * p.N_PTS * p.N_PTS) + 
                      ex * (p.N_PTS * p.N_PTS) + 
                      iy * p.N_PTS + 
                      ix;
            return Sigma_Source[idx];
        });

        vts << "      </PointData>\n";
        vts << "    </Piece>\n";
        vts << "  </StructuredGrid>\n";
        vts << "</VTKFile>\n";

        vts.close();
        std::cout << "Output written: " << full_path << std::endl;

        // Register and update PVD
        snapshots.push_back({time, filename});
        write_pvd();
    }
};

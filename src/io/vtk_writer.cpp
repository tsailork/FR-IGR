/// @file vtk_writer.cpp
/// @brief VTK StructuredGrid (.vts) output writer implementation.

#include "vtk_writer.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

void ensure_output_directory(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
}

VTKWriter::VTKWriter(const std::string& name) : base_name(name) {}

void VTKWriter::write_pvd() {
    std::string pvd_filename = "pv_outputs/" + base_name + ".pvd";
    std::ofstream pvd(pvd_filename);
    if (!pvd.is_open()) {
        std::cerr << "Error: Could not update PVD file " << pvd_filename << "\n";
        return;
    }

    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvd << "  <Collection>\n";
    
    for (const auto& snap : snapshots) {
        pvd << "    <DataSet timestep=\"" << snap.first
            << "\" group=\"\" part=\"0\" file=\"" << snap.second << "\"/>\n";
    }
    
    pvd << "  </Collection>\n";
    pvd << "</VTKFile>\n";
}

void VTKWriter::write_snapshot(Solver& solver, int step, double time) {
    const Parameters& p = solver.p;
    std::string filename = "sol_" + std::to_string(step) + ".vts";
    std::string full_path = "pv_outputs/" + filename;
    
    std::ofstream vts(full_path);
    if (!vts.is_open()) {
        std::cerr << "Error: Could not open output file " << full_path << "\n";
        return;
    }

    int nx = p.N_ELEM_X * p.N_PTS;
    int ny = p.N_ELEM_Y * p.N_PTS;
    
    vts << "<?xml version=\"1.0\"?>\n";
    vts << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vts << "  <StructuredGrid WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";
    vts << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";

    // --- Points ---
    vts << "      <Points>\n";
    vts << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vts << std::fixed << std::setprecision(6);
    for (int J = 0; J < ny; ++J) {
        int ey = J / p.N_PTS, iy = J % p.N_PTS;
        for (int I = 0; I < nx; ++I) {
            int ex = I / p.N_PTS, ix = I % p.N_PTS;
            double x = p.X_MIN + (ex + 0.5*(1+solver.basis.z[ix])) * solver.dx;
            double y = p.Y_MIN + (ey + 0.5*(1+solver.basis.z[iy])) * solver.dy;
            vts << x << " " << y << " 0.0\n";
        }
    }
    vts << "        </DataArray>\n";
    vts << "      </Points>\n";

    // --- Point Data ---
    vts << "      <PointData Scalars=\"rho\">\n";

    if (p.ENABLE_IGR) solver.compute_sensor_source();

    auto write_array = [&](const std::string& name, auto getter) {
        vts << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (int J = 0; J < ny; ++J) {
            int ey = J / p.N_PTS, iy = J % p.N_PTS;
            for (int I = 0; I < nx; ++I) {
                int ex = I / p.N_PTS, ix = I % p.N_PTS;
                vts << getter(ey, ex, iy, ix) << " ";
            }
            vts << "\n";
        }
        vts << "        </DataArray>\n";
    };

    write_array("rho",   [&](int ey, int ex, int iy, int ix) { return solver.U(0, ey, ex, iy, ix); });
    write_array("rho_u", [&](int ey, int ex, int iy, int ix) { return solver.U(1, ey, ex, iy, ix); });
    write_array("rho_v", [&](int ey, int ex, int iy, int ix) { return solver.U(2, ey, ex, iy, ix); });
    write_array("rho_E", [&](int ey, int ex, int iy, int ix) { return solver.U(3, ey, ex, iy, ix); });

    auto get_prim = [&](int ey, int ex, int iy, int ix, int var) {
        double r = solver.U(0, ey, ex, iy, ix);
        double ru = solver.U(1, ey, ex, iy, ix);
        double rv = solver.U(2, ey, ex, iy, ix);
        double E = solver.U(3, ey, ex, iy, ix);
        double u = ru / r, v = rv / r;
        double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
        if (var == 0) return u;
        if (var == 1) return v;
        if (var == 2) return press;
        if (var == 3) return press/r; // Temperature
        if (var == 4) return std::sqrt(u*u + v*v) / std::sqrt(p.GAMMA * std::abs(press) / r); // Mach
        return 0.0;
    };

    write_array("u", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 0); });
    write_array("v", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 1); });
    write_array("Pressure", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 2); });
    write_array("Temperature", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 3); });
    write_array("Mach", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 4); });

    if (p.ENABLE_IGR) {
        write_array("Sigma", [&](int ey, int ex, int iy, int ix) {
            return solver.sigma_field[solver.get_flat_idx(ey, ex, iy, ix)];
        });
        write_array("Sigma_Source", [&](int ey, int ex, int iy, int ix) {
            return solver.S_buf[solver.get_flat_idx(ey, ex, iy, ix)];
        });
    }

    vts << "      </PointData>\n";
    vts << "    </Piece>\n";
    vts << "  </StructuredGrid>\n";
    vts << "</VTKFile>\n";

    std::cout << "Output written: " << full_path << "\n";
    snapshots.push_back({time, filename});
    write_pvd();
}

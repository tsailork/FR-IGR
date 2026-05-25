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

#define RGASAIR (287.43)

void ensure_output_directory(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
}

VTKWriter::VTKWriter(const std::string& name) : base_name(name) {}

void VTKWriter::write_pvd(const std::string& pvd_name, const std::vector<std::pair<double, std::string>>& snaps) {
    std::string pvd_filename = "pv_outputs/" + pvd_name;
    std::ofstream pvd(pvd_filename);
    if (!pvd.is_open()) {
        std::cerr << "Error: Could not update PVD file " << pvd_filename << "\n";
        return;
    }

    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvd << "  <Collection>\n";
    
    for (const auto& snap : snaps) {
        pvd << "    <DataSet timestep=\"" << snap.first
            << "\" group=\"\" part=\"0\" file=\"" << snap.second << "\"/>\n";
    }
    
    pvd << "  </Collection>\n";
    pvd << "</VTKFile>\n";
}

void VTKWriter::load_existing_pvd() {
    auto load_pvd_file = [&](const std::string& filename, std::vector<std::pair<double, std::string>>& snaps) {
        std::ifstream pvd(filename);
        if (!pvd.is_open()) return;

        std::string line;
        while (std::getline(pvd, line)) {
            if (line.find("<DataSet") != std::string::npos) {
                size_t t_pos = line.find("timestep=\"");
                if (t_pos == std::string::npos) continue;
                t_pos += 10;
                size_t t_end = line.find("\"", t_pos);
                double time = std::stod(line.substr(t_pos, t_end - t_pos));

                size_t f_pos = line.find("file=\"");
                if (f_pos == std::string::npos) continue;
                f_pos += 6;
                size_t f_end = line.find("\"", f_pos);
                std::string filename_str = line.substr(f_pos, f_end - f_pos);

                snaps.push_back({time, filename_str});
            }
        }
    };
    
    load_pvd_file("pv_outputs/solution.pvd", sol_snapshots);
    load_pvd_file("pv_outputs/plot.pvd", plot_snapshots);
    std::cout << "[IO] Loaded " << sol_snapshots.size() << " checkpoint snapshots and " 
              << plot_snapshots.size() << " plot snapshots from existing PVDs\n";
}

void VTKWriter::write_checkpoint(Solver& solver, int step, double time) {
    const Parameters& p = solver.p;
    std::string vtm_filename = "sol_" + std::to_string(step) + ".vtm";
    std::string vtm_path = "pv_outputs/" + vtm_filename;

    std::ofstream vtm(vtm_path);
    if (!vtm.is_open()) {
        std::cerr << "Error: Could not open VTM file " << vtm_path << "\n";
        return;
    }

    vtm << "<?xml version=\"1.0\"?>\n";
    vtm << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtm << "  <vtkMultiBlockDataSet>\n";

    if (p.ENABLE_IGR) solver.compute_sensor_source();

    for (auto& b : solver.blocks) {
        std::string block_filename = "sol_b" + std::to_string(b.id) + "_" + std::to_string(step) + ".vts";
        std::string block_path = "pv_outputs/" + block_filename;
        
        std::ofstream vts(block_path);
        if (!vts.is_open()) continue;

        int nx = b.nx * p.N_PTS;
        int ny = b.ny * p.N_PTS;

        vts << "<?xml version=\"1.0\"?>\n";
        vts << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        vts << "  <StructuredGrid WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";
        vts << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";

        vts << "      <Points>\n";
        vts << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        vts << std::fixed << std::setprecision(6);
        for (int J = 0; J < ny; ++J) {
            int ey = J / p.N_PTS, iy = J % p.N_PTS;
            for (int I = 0; I < nx; ++I) {
                int ex = I / p.N_PTS, ix = I % p.N_PTS;
                double x = b.x_min + (ex + 0.5*(1+solver.basis.z[ix])) * b.dx;
                double y = b.y_min + (ey + 0.5*(1+solver.basis.z[iy])) * b.dy;
                vts << x << " " << y << " 0.0\n";
            }
        }
        vts << "        </DataArray>\n";
        vts << "      </Points>\n";

        vts << "      <PointData Scalars=\"rho\">\n";

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

        write_array("rho",   [&](int ey, int ex, int iy, int ix) { return b.U(0, ey, ex, iy, ix); });
        write_array("rho_u", [&](int ey, int ex, int iy, int ix) { return b.U(1, ey, ex, iy, ix); });
        write_array("rho_v", [&](int ey, int ex, int iy, int ix) { return b.U(2, ey, ex, iy, ix); });
        write_array("rho_E", [&](int ey, int ex, int iy, int ix) { return b.U(3, ey, ex, iy, ix); });

        auto get_prim = [&](int ey, int ex, int iy, int ix, int var) {
            double r = b.U(0, ey, ex, iy, ix);
            double ru = b.U(1, ey, ex, iy, ix);
            double rv = b.U(2, ey, ex, iy, ix);
            double E = b.U(3, ey, ex, iy, ix);
            double u = ru / r, v = rv / r;
            double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
            if (var == 0) return u;
            if (var == 1) return v;
            if (var == 2) return press;
            if (var == 3) return press/(r*RGASAIR);
            if (var == 4) return std::sqrt(u*u + v*v) / std::sqrt(p.GAMMA * std::abs(press) / r);
            return 0.0;
        };

        write_array("u", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 0); });
        write_array("v", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 1); });
        write_array("Pressure", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 2); });
        write_array("Temperature", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 3); });
        write_array("Mach", [&](int ey, int ex, int iy, int ix) { return get_prim(ey, ex, iy, ix, 4); });

        if (p.ENABLE_IGR) {
            write_array("Sigma", [&](int ey, int ex, int iy, int ix) {
                return b.sigma_field[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
            });
            write_array("Sigma_Source", [&](int ey, int ex, int iy, int ix) {
                return b.S_buf[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
            });
        }

        if (p.ENABLE_IB) {
            write_array("phi", [&](int ey, int ex, int iy, int ix) {
                double x_pt = b.x_min + (ex + 0.5*(1+solver.basis.z[ix])) * b.dx;
                double y_pt = b.y_min + (ey + 0.5*(1+solver.basis.z[iy])) * b.dy;
                return solver.get_ib_mask(x_pt, y_pt, b.dx, b.dy);
            });
        }

        vts << "      </PointData>\n";
        vts << "    </Piece>\n";
        vts << "  </StructuredGrid>\n";
        vts << "</VTKFile>\n";

        vtm << "    <DataSet index=\"" << b.id << "\" name=\"Block_" << b.id << "\" file=\"" << block_filename << "\"/>\n";
    }

    vtm << "  </vtkMultiBlockDataSet>\n";
    vtm << "</VTKFile>\n";

    std::cout << "[Checkpoint] Output written: " << vtm_path << "\n";
    
    bool duplicate = false;
    for (auto& snap : sol_snapshots) {
        if (std::abs(snap.first - time) < 1e-12) {
            snap.second = vtm_filename;
            duplicate = true;
            break;
        }
    }
    if (!duplicate) sol_snapshots.push_back({time, vtm_filename});
    
    write_pvd("solution.pvd", sol_snapshots);
}

void VTKWriter::write_plot(Solver& solver, int step, double time) {
    const Parameters& p = solver.p;
    std::string vtm_filename = "plot_" + std::to_string(step) + ".vtm";
    std::string vtm_path = "pv_outputs/" + vtm_filename;

    std::ofstream vtm(vtm_path);
    if (!vtm.is_open()) {
        std::cerr << "Error: Could not open VTM file " << vtm_path << "\n";
        return;
    }

    vtm << "<?xml version=\"1.0\"?>\n";
    vtm << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtm << "  <vtkMultiBlockDataSet>\n";

    if (p.ENABLE_IGR) solver.compute_sensor_source();

    int n_sub = std::max(1, p.P_DEG);

    for (auto& b : solver.blocks) {
        std::string block_filename = "plot_b" + std::to_string(b.id) + "_" + std::to_string(step) + ".vts";
        std::string block_path = "pv_outputs/" + block_filename;
        
        std::ofstream vts(block_path);
        if (!vts.is_open()) continue;

        int nx = b.nx * n_sub + 1;
        int ny = b.ny * n_sub + 1;

        vts << "<?xml version=\"1.0\"?>\n";
        vts << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        vts << "  <StructuredGrid WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";
        vts << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";

        vts << "      <Points>\n";
        vts << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        vts << std::fixed << std::setprecision(6);
        for (int J = 0; J < ny; ++J) {
            int ey = J / n_sub;
            int ky = J % n_sub;
            if (ey == b.ny) {
                ey = b.ny - 1;
                ky = n_sub;
            }
            double s = -1.0 + 2.0 * ky / n_sub;
            double y = b.y_min + (ey + 0.5 * (1.0 + s)) * b.dy;

            for (int I = 0; I < nx; ++I) {
                int ex = I / n_sub;
                int kx = I % n_sub;
                if (ex == b.nx) {
                    ex = b.nx - 1;
                    kx = n_sub;
                }
                double r = -1.0 + 2.0 * kx / n_sub;
                double x = b.x_min + (ex + 0.5 * (1.0 + r)) * b.dx;
                vts << x << " " << y << " 0.0\n";
            }
        }
        vts << "        </DataArray>\n";
        vts << "      </Points>\n";

        vts << "      <PointData Scalars=\"rho\">\n";

        auto eval_lagrange = [&](int j, double x, const std::vector<double>& z) {
            int n = z.size();
            if (n == 1) return 1.0;
            double numerator = 1.0;
            double denominator = 1.0;
            for (int k = 0; k < n; ++k) {
                if (k != j) {
                    numerator *= (x - z[k]);
                    denominator *= (z[j] - z[k]);
                }
            }
            return numerator / denominator;
        };

        auto write_array = [&](const std::string& name, auto getter) {
            vts << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
            for (int J = 0; J < ny; ++J) {
                int ey = J / n_sub;
                int ky = J % n_sub;
                if (ey == b.ny) {
                    ey = b.ny - 1;
                    ky = n_sub;
                }
                double s = -1.0 + 2.0 * ky / n_sub;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int I = 0; I < nx; ++I) {
                    int ex = I / n_sub;
                    int kx = I % n_sub;
                    if (ex == b.nx) {
                        ex = b.nx - 1;
                        kx = n_sub;
                    }
                    double r = -1.0 + 2.0 * kx / n_sub;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    double val = 0.0;
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            val += getter(ey, ex, iy, ix) * wy[iy] * wx[ix];
                        }
                    }
                    vts << val << " ";
                }
                vts << "\n";
            }
            vts << "        </DataArray>\n";
        };

        write_array("rho",   [&](int ey, int ex, int iy, int ix) { return b.U(0, ey, ex, iy, ix); });
        write_array("rho_u", [&](int ey, int ex, int iy, int ix) { return b.U(1, ey, ex, iy, ix); });
        write_array("rho_v", [&](int ey, int ex, int iy, int ix) { return b.U(2, ey, ex, iy, ix); });
        write_array("rho_E", [&](int ey, int ex, int iy, int ix) { return b.U(3, ey, ex, iy, ix); });

        auto write_prim_array = [&](const std::string& name, int var) {
            vts << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
            for (int J = 0; J < ny; ++J) {
                int ey = J / n_sub;
                int ky = J % n_sub;
                if (ey == b.ny) {
                    ey = b.ny - 1;
                    ky = n_sub;
                }
                double s = -1.0 + 2.0 * ky / n_sub;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int I = 0; I < nx; ++I) {
                    int ex = I / n_sub;
                    int kx = I % n_sub;
                    if (ex == b.nx) {
                        ex = b.nx - 1;
                        kx = n_sub;
                    }
                    double r = -1.0 + 2.0 * kx / n_sub;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    double U_interp[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            double w = wy[iy] * wx[ix];
                            for (int v = 0; v < 4; ++v) {
                                U_interp[v] += b.U(v, ey, ex, iy, ix) * w;
                            }
                        }
                    }

                    double r_val = std::max(1e-14, U_interp[0]);
                    double ru = U_interp[1];
                    double rv = U_interp[2];
                    double E = U_interp[3];
                    double u = ru / r_val;
                    double v = rv / r_val;
                    double press = (p.GAMMA - 1.0) * (E - 0.5 * r_val * (u*u + v*v));
                    if (press < 1e-14) press = 1e-14;

                    double val = 0.0;
                    if (var == 0) val = u;
                    else if (var == 1) val = v;
                    else if (var == 2) val = press;
                    else if (var == 3) val = press / (RGASAIR*r_val);
                    else if (var == 4) val = std::sqrt(u*u + v*v) / std::sqrt(p.GAMMA * std::abs(press) / r_val);

                    vts << val << " ";
                }
                vts << "\n";
            }
            vts << "        </DataArray>\n";
        };

        write_prim_array("u", 0);
        write_prim_array("v", 1);
        write_prim_array("Pressure", 2);
        write_prim_array("Temperature", 3);
        write_prim_array("Mach", 4);

        if (p.ENABLE_IGR) {
            write_array("Sigma", [&](int ey, int ex, int iy, int ix) {
                return b.sigma_field[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
            });
            write_array("Sigma_Source", [&](int ey, int ex, int iy, int ix) {
                return b.S_buf[b.get_flat_idx(ey, ex, iy, ix, p.N_PTS)];
            });
        }

        if (p.ENABLE_IB) {
            write_array("phi", [&](int ey, int ex, int iy, int ix) {
                double x_pt = b.x_min + (ex + 0.5*(1+solver.basis.z[ix])) * b.dx;
                double y_pt = b.y_min + (ey + 0.5*(1+solver.basis.z[iy])) * b.dy;
                return solver.get_ib_mask(x_pt, y_pt, b.dx, b.dy);
            });
        }

        vts << "      </PointData>\n";
        vts << "    </Piece>\n";
        vts << "  </StructuredGrid>\n";
        vts << "</VTKFile>\n";

        vtm << "    <DataSet index=\"" << b.id << "\" name=\"Block_" << b.id << "\" file=\"" << block_filename << "\"/>\n";
    }

    vtm << "  </vtkMultiBlockDataSet>\n";
    vtm << "</VTKFile>\n";

    std::cout << "[Plot] Output written: " << vtm_path << "\n";
    
    bool duplicate = false;
    for (auto& snap : plot_snapshots) {
        if (std::abs(snap.first - time) < 1e-12) {
            snap.second = vtm_filename;
            duplicate = true;
            break;
        }
    }
    if (!duplicate) plot_snapshots.push_back({time, vtm_filename});
    
    write_pvd("plot.pvd", plot_snapshots);
}

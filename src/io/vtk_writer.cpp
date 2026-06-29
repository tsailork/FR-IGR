/**
 * @file vtk_writer.cpp
 * @brief VTK UnstructuredGrid (.vtu) output writer for decoupled cell structures.
 */

#include "vtk_writer.hpp"
#include "../core/solver.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>

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
    std::string vtu_filename = "sol_" + std::to_string(step) + ".vtu";
    std::string vtu_path = "pv_outputs/" + vtu_filename;

    std::ofstream vtu(vtu_path);
    if (!vtu.is_open()) {
        std::cerr << "Error: Could not open VTU file " << vtu_path << "\n";
        return;
    }

    if (p.ENABLE_IGR) solver.compute_sensor_source();

    int npts = p.N_PTS;
    int npts2 = npts * npts;
    int num_cells_total = solver.cells.size();
    int total_points = num_cells_total * npts2;
    int cells_per_element = (npts > 1) ? (npts - 1) * (npts - 1) : 1;
    int total_vtk_cells = num_cells_total * cells_per_element;

    vtu << "<?xml version=\"1.0\"?>\n";
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << total_points << "\" NumberOfCells=\"" << total_vtk_cells << "\">\n";

    // 1. Points
    vtu << "      <Points>\n";
    vtu << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vtu << std::fixed << std::setprecision(6);
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        Cell* c = solver.cells[c_idx];
        for (int iy = 0; iy < npts; ++iy) {
            for (int ix = 0; ix < npts; ++ix) {
                double x = c->x_min + 0.5 * (1.0 + solver.basis.z[ix]) * c->dx;
                double y = c->y_min + 0.5 * (1.0 + solver.basis.z[iy]) * c->dy;
                vtu << x << " " << y << " 0.0\n";
            }
        }
    }
    vtu << "        </DataArray>\n";
    vtu << "      </Points>\n";

    // 2. Cells (Connectivity)
    vtu << "      <Cells>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        int point_offset = c_idx * npts2;
        if (npts > 1) {
            for (int iy = 0; iy < npts - 1; ++iy) {
                for (int ix = 0; ix < npts - 1; ++ix) {
                    int bl = point_offset + iy * npts + ix;
                    int br = point_offset + iy * npts + ix + 1;
                    int tr = point_offset + (iy + 1) * npts + ix + 1;
                    int tl = point_offset + (iy + 1) * npts + ix;
                    vtu << bl << " " << br << " " << tr << " " << tl << "\n";
                }
            }
        } else {
            vtu << point_offset << "\n";
        }
    }
    vtu << "        </DataArray>\n";

    vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    int current_offset = 0;
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        for (int j = 0; j < cells_per_element; ++j) {
            current_offset += (npts > 1) ? 4 : 1;
            vtu << current_offset << " ";
        }
        vtu << "\n";
    }
    vtu << "        </DataArray>\n";

    vtu << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        for (int j = 0; j < cells_per_element; ++j) {
            vtu << ((npts > 1) ? "9 " : "1 ");
        }
        vtu << "\n";
    }
    vtu << "        </DataArray>\n";
    vtu << "      </Cells>\n";

    // 3. PointData
    vtu << "      <PointData Scalars=\"rho\">\n";

    // Helper lambda to write PointData arrays
    auto write_point_array = [&](const std::string& name, auto getter) {
        vtu << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
            Cell* c = solver.cells[c_idx];
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    vtu << getter(c, iy, ix) << " ";
                }
            }
            vtu << "\n";
        }
        vtu << "        </DataArray>\n";
    };

    // MortonID (UInt64)
    vtu << "        <DataArray type=\"UInt64\" Name=\"MortonID\" format=\"ascii\">\n";
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        uint64_t mid = solver.cells[c_idx]->morton_id;
        for (int j = 0; j < npts2; ++j) {
            vtu << mid << " ";
        }
        vtu << "\n";
    }
    vtu << "        </DataArray>\n";

    write_point_array("rho",   [&](Cell* c, int iy, int ix) { return c->get_U(0, iy, ix, npts); });
    write_point_array("rho_u", [&](Cell* c, int iy, int ix) { return c->get_U(1, iy, ix, npts); });
    write_point_array("rho_v", [&](Cell* c, int iy, int ix) { return c->get_U(2, iy, ix, npts); });
    write_point_array("rho_E", [&](Cell* c, int iy, int ix) { return c->get_U(3, iy, ix, npts); });

    auto get_prim = [&](Cell* c, int iy, int ix, int var) {
        double r = c->get_U(0, iy, ix, npts);
        double ru = c->get_U(1, iy, ix, npts);
        double rv = c->get_U(2, iy, ix, npts);
        double E = c->get_U(3, iy, ix, npts);
        double u = ru / r, v = rv / r;
        double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
        if (press < 1e-14) press = 1e-14;
        if (var == 0) return u;
        if (var == 1) return v;
        if (var == 2) return press;
        if (var == 3) return press/(r*RGASAIR);
        if (var == 4) return std::sqrt(u*u + v*v) / std::sqrt(p.GAMMA * std::abs(press) / r);
        return 0.0;
    };

    write_point_array("u", [&](Cell* c, int iy, int ix) { return get_prim(c, iy, ix, 0); });
    write_point_array("v", [&](Cell* c, int iy, int ix) { return get_prim(c, iy, ix, 1); });
    write_point_array("Pressure", [&](Cell* c, int iy, int ix) { return get_prim(c, iy, ix, 2); });
    write_point_array("Temperature", [&](Cell* c, int iy, int ix) { return get_prim(c, iy, ix, 3); });
    write_point_array("Mach", [&](Cell* c, int iy, int ix) { return get_prim(c, iy, ix, 4); });

    if (p.ENABLE_IGR) {
        write_point_array("Sigma", [&](Cell* c, int iy, int ix) {
            return c->sigma_field[iy * npts + ix];
        });
        write_point_array("Sigma_Source", [&](Cell* c, int iy, int ix) {
            return c->S_buf[iy * npts + ix];
        });
    }

    if (p.ENABLE_IB) {
        write_point_array("phi", [&](Cell* c, int iy, int ix) {
            double x_pt = c->x_min + 0.5 * (1.0 + solver.basis.z[ix]) * c->dx;
            double y_pt = c->y_min + 0.5 * (1.0 + solver.basis.z[iy]) * c->dy;
            return solver.get_ib_mask(x_pt, y_pt, c->dx, c->dy);
        });
    }

    vtu << "      </PointData>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";

    std::cout << "[Checkpoint] Output written: " << vtu_path << "\n";

    bool duplicate = false;
    for (auto& snap : sol_snapshots) {
        if (std::abs(snap.first - time) < 1e-12) {
            snap.second = vtu_filename;
            duplicate = true;
            break;
        }
    }
    if (!duplicate) sol_snapshots.push_back({time, vtu_filename});

    write_pvd("solution.pvd", sol_snapshots);
}

void VTKWriter::write_plot(Solver& solver, int step, double time) {
    const Parameters& p = solver.p;
    std::string vtu_filename = "plot_" + std::to_string(step) + ".vtu";
    std::string vtu_path = "pv_outputs/" + vtu_filename;

    std::ofstream vtu(vtu_path);
    if (!vtu.is_open()) {
        std::cerr << "Error: Could not open VTU file " << vtu_path << "\n";
        return;
    }

    if (p.ENABLE_IGR) solver.compute_sensor_source();

    int n_sub = std::max(1, p.P_DEG);
    int npts_local = n_sub + 1;
    int n_pts_cell = npts_local * npts_local + n_sub * n_sub;
    int n_cells_cell = 4 * n_sub * n_sub;

    int total_points = solver.cells.size() * n_pts_cell;
    int total_vtk_cells = solver.cells.size() * n_cells_cell;

    vtu << "<?xml version=\"1.0\"?>\n";
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << total_points << "\" NumberOfCells=\"" << total_vtk_cells << "\">\n";

    // 1. Points
    vtu << "      <Points>\n";
    vtu << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vtu << std::fixed << std::setprecision(6);

    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        Cell* c = solver.cells[c_idx];
        // 1.1 Original vertices
        for (int J = 0; J < npts_local; ++J) {
            double s = -1.0 + 2.0 * J / n_sub;
            double y = c->y_min + 0.5 * (1.0 + s) * c->dy;
            for (int I = 0; I < npts_local; ++I) {
                double r = -1.0 + 2.0 * I / n_sub;
                double x = c->x_min + 0.5 * (1.0 + r) * c->dx;
                vtu << x << " " << y << " 0.0\n";
            }
        }
        // 1.2 Cell centers
        for (int j = 0; j < n_sub; ++j) {
            double s = -1.0 + 2.0 * (j + 0.5) / n_sub;
            double y = c->y_min + 0.5 * (1.0 + s) * c->dy;
            for (int i = 0; i < n_sub; ++i) {
                double r = -1.0 + 2.0 * (i + 0.5) / n_sub;
                double x = c->x_min + 0.5 * (1.0 + r) * c->dx;
                vtu << x << " " << y << " 0.0\n";
            }
        }
    }
    vtu << "        </DataArray>\n";
    vtu << "      </Points>\n";

    // 2. Cells (Connectivity)
    vtu << "      <Cells>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        int po = c_idx * n_pts_cell;
        for (int j = 0; j < n_sub; ++j) {
            for (int i = 0; i < n_sub; ++i) {
                int v_bl = po + j * npts_local + i;
                int v_br = po + j * npts_local + i + 1;
                int v_tr = po + (j + 1) * npts_local + i + 1;
                int v_tl = po + (j + 1) * npts_local + i;
                int v_c  = po + npts_local * npts_local + j * n_sub + i;

                // 4 triangles per sub-cell
                vtu << v_bl << " " << v_br << " " << v_c << "\n";
                vtu << v_br << " " << v_tr << " " << v_c << "\n";
                vtu << v_tr << " " << v_tl << " " << v_c << "\n";
                vtu << v_tl << " " << v_bl << " " << v_c << "\n";
            }
        }
    }
    vtu << "        </DataArray>\n";

    vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    int current_offset = 0;
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        for (int c = 0; c < n_cells_cell; ++c) {
            current_offset += 3;
            vtu << current_offset << " ";
        }
        vtu << "\n";
    }
    vtu << "        </DataArray>\n";

    vtu << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        for (int c = 0; c < n_cells_cell; ++c) {
            vtu << "5 ";
        }
        vtu << "\n";
    }
    vtu << "        </DataArray>\n";
    vtu << "      </Cells>\n";

    // 3. PointData
    vtu << "      <PointData Scalars=\"rho\">\n";

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
        vtu << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
            Cell* c = solver.cells[c_idx];
            // 3.1 Vertices
            for (int J = 0; J < npts_local; ++J) {
                double s = -1.0 + 2.0 * J / n_sub;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int I = 0; I < npts_local; ++I) {
                    double r = -1.0 + 2.0 * I / n_sub;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    double val = 0.0;
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            val += getter(c, iy, ix) * wy[iy] * wx[ix];
                        }
                    }
                    vtu << val << " ";
                }
                vtu << "\n";
            }
            // 3.2 Centers
            for (int j = 0; j < n_sub; ++j) {
                double s = -1.0 + 2.0 * (j + 0.5) / n_sub;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int i = 0; i < n_sub; ++i) {
                    double r = -1.0 + 2.0 * (i + 0.5) / n_sub;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    double val = 0.0;
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            val += getter(c, iy, ix) * wy[iy] * wx[ix];
                        }
                    }
                    vtu << val << " ";
                }
                vtu << "\n";
            }
        }
        vtu << "        </DataArray>\n";
    };

    write_array("rho",   [&](Cell* c, int iy, int ix) { return c->get_U(0, iy, ix, p.N_PTS); });
    write_array("rho_u", [&](Cell* c, int iy, int ix) { return c->get_U(1, iy, ix, p.N_PTS); });
    write_array("rho_v", [&](Cell* c, int iy, int ix) { return c->get_U(2, iy, ix, p.N_PTS); });
    write_array("rho_E", [&](Cell* c, int iy, int ix) { return c->get_U(3, iy, ix, p.N_PTS); });

    auto write_prim_array = [&](const std::string& name, int var) {
        vtu << "        <DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
            Cell* c = solver.cells[c_idx];
            // 3.1 Vertices
            for (int J = 0; J < npts_local; ++J) {
                double s = -1.0 + 2.0 * J / n_sub;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int I = 0; I < npts_local; ++I) {
                    double r = -1.0 + 2.0 * I / n_sub;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    double U_interp[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            double w = wy[iy] * wx[ix];
                            for (int v = 0; v < 4; ++v) {
                                U_interp[v] += c->get_U(v, iy, ix, p.N_PTS) * w;
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
                    vtu << val << " ";
                }
                vtu << "\n";
            }
            // 3.2 Centers
            for (int j = 0; j < n_sub; ++j) {
                double s = -1.0 + 2.0 * (j + 0.5) / n_sub;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int i = 0; i < n_sub; ++i) {
                    double r = -1.0 + 2.0 * (i + 0.5) / n_sub;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    double U_interp[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int iy = 0; iy < p.N_PTS; ++iy) {
                        for (int ix = 0; ix < p.N_PTS; ++ix) {
                            double w = wy[iy] * wx[ix];
                            for (int v = 0; v < 4; ++v) {
                                U_interp[v] += c->get_U(v, iy, ix, p.N_PTS) * w;
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
                    vtu << val << " ";
                }
                vtu << "\n";
            }
        }
        vtu << "        </DataArray>\n";
    };

    write_prim_array("u", 0);
    write_prim_array("v", 1);
    write_prim_array("Pressure", 2);
    write_prim_array("Temperature", 3);
    write_prim_array("Mach", 4);

    if (p.ENABLE_IGR) {
        write_array("Sigma", [&](Cell* c, int iy, int ix) {
            return c->sigma_field[iy * p.N_PTS + ix];
        });
        write_array("Sigma_Source", [&](Cell* c, int iy, int ix) {
            return c->S_buf[iy * p.N_PTS + ix];
        });
    }

    if (p.ENABLE_IB) {
        write_array("phi", [&](Cell* c, int iy, int ix) {
            double x_pt = c->x_min + 0.5 * (1.0 + solver.basis.z[ix]) * c->dx;
            double y_pt = c->y_min + 0.5 * (1.0 + solver.basis.z[iy]) * c->dy;
            return solver.get_ib_mask(x_pt, y_pt, c->dx, c->dy);
        });
    }

    vtu << "      </PointData>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";

    std::cout << "[Plot] Output written: " << vtu_path << "\n";

    bool duplicate = false;
    for (auto& snap : plot_snapshots) {
        if (std::abs(snap.first - time) < 1e-12) {
            snap.second = vtu_filename;
            duplicate = true;
            break;
        }
    }
    if (!duplicate) plot_snapshots.push_back({time, vtu_filename});

    write_pvd("plot.pvd", plot_snapshots);
}

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
#include <cstring>
#include <unordered_map>

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

    if (p.OUTPUT_DIV_ND) {
        write_point_array("div_nd", [&](Cell* c, int iy, int ix) {
            double du_dx = 0.0, dv_dy = 0.0;
            for (int k = 0; k < npts; ++k) {
                double rk_x = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, k, npts));
                du_dx += solver.basis.D[ix][k] * (c->get_U(1, iy, k, npts) / rk_x);
                double rk_y = std::max(p.POS_LIMITER_EPS, c->get_U(0, k, ix, npts));
                dv_dy += solver.basis.D[iy][k] * (c->get_U(2, k, ix, npts) / rk_y);
            }
            du_dx *= (2.0 / c->dx);
            dv_dy *= (2.0 / c->dy);
            double div_u = du_dx + dv_dy;
            
            double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, npts));
            double u = c->get_U(1, iy, ix, npts) / rho;
            double v = c->get_U(2, iy, ix, npts) / rho;
            double press = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, npts) - 0.5 * rho * (u * u + v * v)));
            double a_loc = std::sqrt(p.GAMMA * press / rho);
            double h_loc = std::min(c->dx, c->dy);
            return -div_u * h_loc / (a_loc * (p.P_DEG + 1));
        });
    }

    if (p.ENABLE_IGR) {
        write_point_array("Sigma", [&](Cell* c, int iy, int ix) {
            return c->sigma_field[iy * npts + ix];
        });
        write_point_array("Sigma_Source", [&](Cell* c, int iy, int ix) {
            return c->S_buf[iy * npts + ix];
        });
    }

    if (p.ENABLE_PPR) {
        // S_field is the raw phantom-pressure state variable (S = rho * P_phan).
        // Written first so the restart loader can recover it without reading derived fields.
        write_point_array("S_field", [&](Cell* c, int iy, int ix) {
            return c->S_field[iy * npts + ix];
        });
        write_point_array("P_phan", [&](Cell* c, int iy, int ix) {
            double r = c->get_U(0, iy, ix, npts);
            double S = c->S_field[iy * npts + ix];
            return S / std::max(p.POS_LIMITER_EPS, r);
        });
        write_point_array("P_reg", [&](Cell* c, int iy, int ix) {
            double r = c->get_U(0, iy, ix, npts);
            double ru = c->get_U(1, iy, ix, npts);
            double rv = c->get_U(2, iy, ix, npts);
            double E = c->get_U(3, iy, ix, npts);
            double u = ru / r, v = rv / r;
            double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
            double S = c->S_field[iy * npts + ix];
            double p_phan = S / std::max(p.POS_LIMITER_EPS, r);
            double theta_cfl = (p.PPR_ADAPTIVE_THETA) ? c->theta_avg : p.PPR_THETA;
            return press + theta_cfl * (press - p_phan);
        });
        write_point_array("P_diff", [&](Cell* c, int iy, int ix) {
            double r = c->get_U(0, iy, ix, npts);
            double ru = c->get_U(1, iy, ix, npts);
            double rv = c->get_U(2, iy, ix, npts);
            double E = c->get_U(3, iy, ix, npts);
            double u = ru / r, v = rv / r;
            double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
            double S = c->S_field[iy * npts + ix];
            double p_phan = S / std::max(p.POS_LIMITER_EPS, r);
            return press - p_phan;
        });
        if (p.OUTPUT_ADAPTIVE_THETA) {
            write_point_array("Theta_PPR", [&](Cell* c, int iy, int ix) {
                return (p.PPR_ADAPTIVE_THETA) ? c->theta_avg : p.PPR_THETA;
            });
        }
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

    int n_sub = p.PLOT_SUB_DIVISIONS > 0 ? p.PLOT_SUB_DIVISIONS : std::max(1, p.P_DEG);
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

    if (p.OUTPUT_DIV_ND) {
        write_array("div_nd", [&](Cell* c, int iy, int ix) {
            double du_dx = 0.0, dv_dy = 0.0;
            for (int k = 0; k < p.N_PTS; ++k) {
                double rk_x = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, k, p.N_PTS));
                du_dx += solver.basis.D[ix][k] * (c->get_U(1, iy, k, p.N_PTS) / rk_x);
                double rk_y = std::max(p.POS_LIMITER_EPS, c->get_U(0, k, ix, p.N_PTS));
                dv_dy += solver.basis.D[iy][k] * (c->get_U(2, k, ix, p.N_PTS) / rk_y);
            }
            du_dx *= (2.0 / c->dx);
            dv_dy *= (2.0 / c->dy);
            double div_u = du_dx + dv_dy;
            
            double rho = std::max(p.POS_LIMITER_EPS, c->get_U(0, iy, ix, p.N_PTS));
            double u = c->get_U(1, iy, ix, p.N_PTS) / rho;
            double v = c->get_U(2, iy, ix, p.N_PTS) / rho;
            double press = std::max(p.POS_LIMITER_EPS, (p.GAMMA - 1.0) * (c->get_U(3, iy, ix, p.N_PTS) - 0.5 * rho * (u * u + v * v)));
            double a_loc = std::sqrt(p.GAMMA * press / rho);
            double h_loc = std::min(c->dx, c->dy);
            return -div_u * h_loc / (a_loc * (p.P_DEG + 1));
        });
    }

    if (p.ENABLE_IGR) {
        write_array("Sigma", [&](Cell* c, int iy, int ix) {
            return c->sigma_field[iy * p.N_PTS + ix];
        });
        write_array("Sigma_Source", [&](Cell* c, int iy, int ix) {
            return c->S_buf[iy * p.N_PTS + ix];
        });
    }

    if (p.ENABLE_PPR) {
        write_array("P_phan", [&](Cell* c, int iy, int ix) {
            double r = c->get_U(0, iy, ix, p.N_PTS);
            double S = c->S_field[iy * p.N_PTS + ix];
            return S / std::max(p.POS_LIMITER_EPS, r);
        });
        write_array("P_reg", [&](Cell* c, int iy, int ix) {
            double r = c->get_U(0, iy, ix, p.N_PTS);
            double ru = c->get_U(1, iy, ix, p.N_PTS);
            double rv = c->get_U(2, iy, ix, p.N_PTS);
            double E = c->get_U(3, iy, ix, p.N_PTS);
            double u = ru / r, v = rv / r;
            double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
            double S = c->S_field[iy * p.N_PTS + ix];
            double p_phan = S / std::max(p.POS_LIMITER_EPS, r);
            double theta_cfl = (p.PPR_ADAPTIVE_THETA) ? c->theta_avg : p.PPR_THETA;
            return press + theta_cfl * (press - p_phan);
        });
        write_array("P_diff", [&](Cell* c, int iy, int ix) {
            double r = c->get_U(0, iy, ix, p.N_PTS);
            double ru = c->get_U(1, iy, ix, p.N_PTS);
            double rv = c->get_U(2, iy, ix, p.N_PTS);
            double E = c->get_U(3, iy, ix, p.N_PTS);
            double u = ru / r, v = rv / r;
            double press = (p.GAMMA - 1.0) * (E - 0.5 * r * (u*u + v*v));
            double S = c->S_field[iy * p.N_PTS + ix];
            double p_phan = S / std::max(p.POS_LIMITER_EPS, r);
            return press - p_phan;
        });
        if (p.OUTPUT_ADAPTIVE_THETA) {
            write_array("Theta_PPR", [&](Cell* c, int iy, int ix) {
                return (p.PPR_ADAPTIVE_THETA) ? c->theta_avg : p.PPR_THETA;
            });
        }
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

static std::string base64_encode(const unsigned char* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t val = (data[i] << 16) | ((i + 1 < len ? data[i + 1] : 0) << 8) | (i + 2 < len ? data[i + 2] : 0);
        out.push_back(table[(val >> 18) & 0x3F]);
        out.push_back(table[(val >> 12) & 0x3F]);
        out.push_back(i + 1 < len ? table[(val >> 6) & 0x3F] : '=');
        out.push_back(i + 2 < len ? table[val & 0x3F] : '=');
    }
    return out;
}

template<typename T>
static void write_binary_data_array(std::ostream& os, const std::string& name, int num_components, const std::vector<T>& data) {
    uint32_t num_bytes = static_cast<uint32_t>(data.size() * sizeof(T));
    std::vector<unsigned char> raw_buf(sizeof(uint32_t) + num_bytes);
    std::memcpy(raw_buf.data(), &num_bytes, sizeof(uint32_t));
    if (num_bytes > 0) {
        std::memcpy(raw_buf.data() + sizeof(uint32_t), data.data(), num_bytes);
    }
    std::string b64 = base64_encode(raw_buf.data(), raw_buf.size());
    std::string type_str = "Float32";
    if (std::is_same_v<T, double>) type_str = "Float64";
    else if (std::is_same_v<T, float>) type_str = "Float32";
    else if (std::is_same_v<T, int32_t>) type_str = "Int32";
    else if (std::is_same_v<T, uint64_t>) type_str = "UInt64";
    else if (std::is_same_v<T, uint8_t>) type_str = "UInt8";

    os << "        <DataArray type=\"" << type_str << "\" Name=\"" << name << "\" NumberOfComponents=\"" << num_components << "\" format=\"binary\">\n";
    os << "          " << b64 << "\n";
    os << "        </DataArray>\n";
}

void VTKWriter::write_checkpoint(SolverDim<3>& solver, int step, double time) {
    const Parameters& p = solver.p;
    std::string vtu_filename = "sol_" + std::to_string(step) + ".vtu";
    std::string vtu_path = "pv_outputs/" + vtu_filename;

    std::ofstream vtu(vtu_path, std::ios::binary);
    if (!vtu.is_open()) {
        std::cerr << "Error: Could not open VTU file " << vtu_path << "\n";
        return;
    }

    int npts = p.N_PTS;
    int npts3 = npts * npts * npts;
    int num_cells_total = solver.cells.size();
    int total_points = num_cells_total * npts3;
    int cells_per_element = (npts > 1) ? (npts - 1) * (npts - 1) * (npts - 1) : 1;
    int total_vtk_cells = num_cells_total * cells_per_element;

    vtu << "<?xml version=\"1.0\"?>\n";
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << total_points << "\" NumberOfCells=\"" << total_vtk_cells << "\">\n";

    // 1. Points
    vtu << "      <Points>\n";
    std::vector<float> pts_data;
    pts_data.reserve(total_points * 3);
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        Cell3D* c = solver.cells[c_idx];
        for (int iz = 0; iz < npts; ++iz) {
            for (int iy = 0; iy < npts; ++iy) {
                for (int ix = 0; ix < npts; ++ix) {
                    float x = c->x_min + 0.5 * (1.0 + solver.basis.z[ix]) * c->dx;
                    float y = c->y_min + 0.5 * (1.0 + solver.basis.z[iy]) * c->dy;
                    float z = c->z_min + 0.5 * (1.0 + solver.basis.z[iz]) * c->dz;
                    pts_data.push_back(x);
                    pts_data.push_back(y);
                    pts_data.push_back(z);
                }
            }
        }
    }
    write_binary_data_array(vtu, "Points", 3, pts_data);
    vtu << "      </Points>\n";

    // 2. Cells (Connectivity, Offsets, Types)
    vtu << "      <Cells>\n";
    std::vector<int32_t> conn_data;
    std::vector<int32_t> offset_data;
    std::vector<uint8_t> type_data;
    conn_data.reserve(total_vtk_cells * ((npts > 1) ? 8 : 1));
    offset_data.reserve(total_vtk_cells);
    type_data.reserve(total_vtk_cells);

    int current_offset = 0;
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        int point_offset = c_idx * npts3;
        if (npts > 1) {
            for (int iz = 0; iz < npts - 1; ++iz) {
                for (int iy = 0; iy < npts - 1; ++iy) {
                    for (int ix = 0; ix < npts - 1; ++ix) {
                        int v0 = point_offset + iz * npts * npts + iy * npts + ix;
                        int v1 = point_offset + iz * npts * npts + iy * npts + ix + 1;
                        int v2 = point_offset + iz * npts * npts + (iy + 1) * npts + ix + 1;
                        int v3 = point_offset + iz * npts * npts + (iy + 1) * npts + ix;
                        int v4 = point_offset + (iz + 1) * npts * npts + iy * npts + ix;
                        int v5 = point_offset + (iz + 1) * npts * npts + iy * npts + ix + 1;
                        int v6 = point_offset + (iz + 1) * npts * npts + (iy + 1) * npts + ix + 1;
                        int v7 = point_offset + (iz + 1) * npts * npts + (iy + 1) * npts + ix;

                        conn_data.push_back(v0); conn_data.push_back(v1);
                        conn_data.push_back(v2); conn_data.push_back(v3);
                        conn_data.push_back(v4); conn_data.push_back(v5);
                        conn_data.push_back(v6); conn_data.push_back(v7);

                        current_offset += 8;
                        offset_data.push_back(current_offset);
                        type_data.push_back(12); // VTK_HEXAHEDRON
                    }
                }
            }
        } else {
            conn_data.push_back(point_offset);
            current_offset += 1;
            offset_data.push_back(current_offset);
            type_data.push_back(1); // VTK_VERTEX
        }
    }

    write_binary_data_array(vtu, "connectivity", 1, conn_data);
    write_binary_data_array(vtu, "offsets", 1, offset_data);
    write_binary_data_array(vtu, "types", 1, type_data);
    vtu << "      </Cells>\n";

    // 3. PointData
    vtu << "      <PointData Scalars=\"rho\">\n";

    // MortonID
    std::vector<uint64_t> morton_data;
    morton_data.reserve(total_points);
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        uint64_t mid = solver.cells[c_idx]->morton_id;
        for (int j = 0; j < npts3; ++j) morton_data.push_back(mid);
    }
    write_binary_data_array(vtu, "MortonID", 1, morton_data);

    // Conservative variables (rho, rho_u, rho_v, rho_w, rho_E)
    auto write_field = [&](const std::string& name, auto getter) {
        std::vector<float> f_data;
        f_data.reserve(total_points);
        for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
            Cell3D* c = solver.cells[c_idx];
            for (int iz = 0; iz < npts; ++iz) {
                for (int iy = 0; iy < npts; ++iy) {
                    for (int ix = 0; ix < npts; ++ix) {
                        f_data.push_back(static_cast<float>(getter(c, iz, iy, ix)));
                    }
                }
            }
        }
        write_binary_data_array(vtu, name, 1, f_data);
    };

    write_field("rho",   [&](Cell3D* c, int iz, int iy, int ix) { return c->U[0 * npts3 + iz * npts * npts + iy * npts + ix]; });
    write_field("rho_u", [&](Cell3D* c, int iz, int iy, int ix) { return c->U[1 * npts3 + iz * npts * npts + iy * npts + ix]; });
    write_field("rho_v", [&](Cell3D* c, int iz, int iy, int ix) { return c->U[2 * npts3 + iz * npts * npts + iy * npts + ix]; });
    write_field("rho_w", [&](Cell3D* c, int iz, int iy, int ix) { return c->U[3 * npts3 + iz * npts * npts + iy * npts + ix]; });
    write_field("rho_E", [&](Cell3D* c, int iz, int iy, int ix) { return c->U[4 * npts3 + iz * npts * npts + iy * npts + ix]; });

    vtu << "      </PointData>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";

    std::cout << "[Checkpoint] Output written (3D Binary): " << vtu_path << "\n";

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

void VTKWriter::write_plot(SolverDim<3>& solver, int step, double time) {
    const Parameters& p = solver.p;
    std::string vtu_filename = "plot_" + std::to_string(step) + ".vtu";
    std::string vtu_path = "pv_outputs/" + vtu_filename;

    std::ofstream vtu(vtu_path, std::ios::binary);
    if (!vtu.is_open()) {
        std::cerr << "Error: Could not open VTU plot file " << vtu_path << "\n";
        return;
    }

    int n_sub = p.PLOT_SUB_DIVISIONS > 0 ? p.PLOT_SUB_DIVISIONS : std::max(1, p.P_DEG);
    int npts_local = n_sub + 1;
    int n_pts_cell = npts_local * npts_local * npts_local;
    int cells_per_element = n_sub * n_sub * n_sub;
    int num_cells_total = solver.cells.size();
    int total_vtk_cells = num_cells_total * cells_per_element;

    auto eval_lagrange = [](int j, double x, const std::vector<double>& z) {
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

    struct PointKey {
        int64_t x, y, z;
        bool operator==(const PointKey& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct PointKeyHash {
        std::size_t operator()(const PointKey& k) const {
            std::size_t h1 = std::hash<int64_t>{}(k.x);
            std::size_t h2 = std::hash<int64_t>{}(k.y);
            std::size_t h3 = std::hash<int64_t>{}(k.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    std::unordered_map<PointKey, int, PointKeyHash> point_map;
    std::vector<float> pts_coords; // 3 floats per unique point
    
    int num_fields = 11 + (p.ENABLE_IGR ? 1 : 0) + (p.ENABLE_IB ? 1 : 0);
    std::vector<double> field_sums;
    std::vector<int> field_counts;
    std::vector<uint64_t> morton_ids;

    std::vector<int> cell_node_map(num_cells_total * n_pts_cell);

    int npts3 = p.N_PTS * p.N_PTS * p.N_PTS;

    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        Cell3D* c = solver.cells[c_idx];
        for (int JZ = 0; JZ < npts_local; ++JZ) {
            double t = -1.0 + 2.0 * JZ / n_sub;
            double z = c->z_min + 0.5 * (1.0 + t) * c->dz;
            std::vector<double> wz(p.N_PTS);
            for (int iz = 0; iz < p.N_PTS; ++iz) wz[iz] = eval_lagrange(iz, t, solver.basis.z);

            for (int JY = 0; JY < npts_local; ++JY) {
                double s = -1.0 + 2.0 * JY / n_sub;
                double y = c->y_min + 0.5 * (1.0 + s) * c->dy;
                std::vector<double> wy(p.N_PTS);
                for (int iy = 0; iy < p.N_PTS; ++iy) wy[iy] = eval_lagrange(iy, s, solver.basis.z);

                for (int JX = 0; JX < npts_local; ++JX) {
                    double r = -1.0 + 2.0 * JX / n_sub;
                    double x = c->x_min + 0.5 * (1.0 + r) * c->dx;
                    std::vector<double> wx(p.N_PTS);
                    for (int ix = 0; ix < p.N_PTS; ++ix) wx[ix] = eval_lagrange(ix, r, solver.basis.z);

                    PointKey key{
                        static_cast<int64_t>(std::round(x * 1e6)),
                        static_cast<int64_t>(std::round(y * 1e6)),
                        static_cast<int64_t>(std::round(z * 1e6))
                    };

                    int pid;
                    auto it = point_map.find(key);
                    if (it == point_map.end()) {
                        pid = static_cast<int>(pts_coords.size() / 3);
                        point_map[key] = pid;
                        pts_coords.push_back(static_cast<float>(x));
                        pts_coords.push_back(static_cast<float>(y));
                        pts_coords.push_back(static_cast<float>(z));
                        field_sums.resize((pid + 1) * num_fields, 0.0);
                        field_counts.push_back(0);
                        morton_ids.push_back(c->morton_id);
                    } else {
                        pid = it->second;
                    }

                    int local_v_idx = JZ * npts_local * npts_local + JY * npts_local + JX;
                    cell_node_map[c_idx * n_pts_cell + local_v_idx] = pid;

                    // Interpolate U at (r, s, t)
                    double U_interp[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    double sigma_interp = 0.0;
                    double ib_interp = 0.0;

                    for (int iz = 0; iz < p.N_PTS; ++iz) {
                        for (int iy = 0; iy < p.N_PTS; ++iy) {
                            for (int ix = 0; ix < p.N_PTS; ++ix) {
                                double w = wz[iz] * wy[iy] * wx[ix];
                                int idx = iz * p.N_PTS * p.N_PTS + iy * p.N_PTS + ix;
                                for (int v = 0; v < 5; ++v) {
                                    U_interp[v] += c->U[v * npts3 + idx] * w;
                                }
                                if (p.ENABLE_IGR && idx < (int)c->sigma_field.size()) {
                                    sigma_interp += c->sigma_field[idx] * w;
                                }
                                if (p.ENABLE_IB && idx < (int)c->ib_mask.size()) {
                                    ib_interp += c->ib_mask[idx] * w;
                                }
                            }
                        }
                    }

                    double rho   = U_interp[0];
                    double rhou  = U_interp[1];
                    double rhov  = U_interp[2];
                    double rhow  = U_interp[3];
                    double rhoE  = U_interp[4];

                    double r_safe = std::max(1e-14, rho);
                    double u = rhou / r_safe;
                    double v = rhov / r_safe;
                    double w_vel = rhow / r_safe;
                    double press = (p.GAMMA - 1.0) * (rhoE - 0.5 * r_safe * (u*u + v*v + w_vel*w_vel));
                    if (press < 1e-14) press = 1e-14;
                    double temp = press / (RGASAIR * r_safe);
                    double mach = std::sqrt(u*u + v*v + w_vel*w_vel) / std::sqrt(p.GAMMA * press / r_safe);

                    int base_f = pid * num_fields;
                    int f = 0;
                    field_sums[base_f + (f++)] += rho;
                    field_sums[base_f + (f++)] += rhou;
                    field_sums[base_f + (f++)] += rhov;
                    field_sums[base_f + (f++)] += rhow;
                    field_sums[base_f + (f++)] += rhoE;
                    field_sums[base_f + (f++)] += u;
                    field_sums[base_f + (f++)] += v;
                    field_sums[base_f + (f++)] += w_vel;
                    field_sums[base_f + (f++)] += press;
                    field_sums[base_f + (f++)] += temp;
                    field_sums[base_f + (f++)] += mach;

                    if (p.ENABLE_IGR && f < num_fields) {
                        field_sums[base_f + (f++)] += sigma_interp;
                    }
                    if (p.ENABLE_IB && f < num_fields) {
                        field_sums[base_f + (f++)] += ib_interp;
                    }

                    field_counts[pid]++;
                }
            }
        }
    }

    int unique_points = static_cast<int>(pts_coords.size() / 3);

    vtu << "<?xml version=\"1.0\"?>\n";
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << unique_points << "\" NumberOfCells=\"" << total_vtk_cells << "\">\n";

    // 1. Points
    vtu << "      <Points>\n";
    write_binary_data_array(vtu, "Points", 3, pts_coords);
    vtu << "      </Points>\n";

    // 2. Cells (Connectivity, Offsets, Types)
    vtu << "      <Cells>\n";
    std::vector<int32_t> conn_data;
    std::vector<int32_t> offset_data;
    std::vector<uint8_t> type_data;
    conn_data.reserve(total_vtk_cells * 8);
    offset_data.reserve(total_vtk_cells);
    type_data.reserve(total_vtk_cells);

    int current_offset = 0;
    for (size_t c_idx = 0; c_idx < solver.cells.size(); ++c_idx) {
        int cell_base = c_idx * n_pts_cell;
        for (int iz = 0; iz < n_sub; ++iz) {
            for (int iy = 0; iy < n_sub; ++iy) {
                for (int ix = 0; ix < n_sub; ++ix) {
                    int v0 = cell_node_map[cell_base + iz * npts_local * npts_local + iy * npts_local + ix];
                    int v1 = cell_node_map[cell_base + iz * npts_local * npts_local + iy * npts_local + ix + 1];
                    int v2 = cell_node_map[cell_base + iz * npts_local * npts_local + (iy + 1) * npts_local + ix + 1];
                    int v3 = cell_node_map[cell_base + iz * npts_local * npts_local + (iy + 1) * npts_local + ix];
                    int v4 = cell_node_map[cell_base + (iz + 1) * npts_local * npts_local + iy * npts_local + ix];
                    int v5 = cell_node_map[cell_base + (iz + 1) * npts_local * npts_local + iy * npts_local + ix + 1];
                    int v6 = cell_node_map[cell_base + (iz + 1) * npts_local * npts_local + (iy + 1) * npts_local + ix + 1];
                    int v7 = cell_node_map[cell_base + (iz + 1) * npts_local * npts_local + (iy + 1) * npts_local + ix];

                    conn_data.push_back(v0); conn_data.push_back(v1);
                    conn_data.push_back(v2); conn_data.push_back(v3);
                    conn_data.push_back(v4); conn_data.push_back(v5);
                    conn_data.push_back(v6); conn_data.push_back(v7);

                    current_offset += 8;
                    offset_data.push_back(current_offset);
                    type_data.push_back(12); // VTK_HEXAHEDRON
                }
            }
        }
    }

    write_binary_data_array(vtu, "connectivity", 1, conn_data);
    write_binary_data_array(vtu, "offsets", 1, offset_data);
    write_binary_data_array(vtu, "types", 1, type_data);
    vtu << "      </Cells>\n";

    // 3. PointData - averaged field output across continuous domain
    vtu << "      <PointData Scalars=\"rho\">\n";

    auto write_averaged_field = [&](const std::string& name, int field_idx) {
        std::vector<float> f_data;
        f_data.reserve(unique_points);
        for (int pid = 0; pid < unique_points; ++pid) {
            double avg = 0.0;
            if (pid < (int)field_counts.size() && field_counts[pid] > 0) {
                int idx = pid * num_fields + field_idx;
                if (idx < (int)field_sums.size()) {
                    avg = field_sums[idx] / field_counts[pid];
                }
            }
            f_data.push_back(static_cast<float>(avg));
        }
        write_binary_data_array(vtu, name, 1, f_data);
    };

    int f_idx = 0;
    write_averaged_field("rho",         f_idx++);
    write_averaged_field("rho_u",       f_idx++);
    write_averaged_field("rho_v",       f_idx++);
    write_averaged_field("rho_w",       f_idx++);
    write_averaged_field("rho_E",       f_idx++);
    write_averaged_field("u",           f_idx++);
    write_averaged_field("v",           f_idx++);
    write_averaged_field("w",           f_idx++);
    write_averaged_field("Pressure",    f_idx++);
    write_averaged_field("Temperature", f_idx++);
    write_averaged_field("Mach",        f_idx++);

    if (p.ENABLE_IGR) {
        write_averaged_field("Sigma", f_idx++);
    }
    if (p.ENABLE_IB) {
        write_averaged_field("ib_mask", f_idx++);
    }

    write_binary_data_array(vtu, "MortonID", 1, morton_ids);

    vtu << "      </PointData>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";

    std::cout << "[Plot] Output written (3D Continuous Binary): " << vtu_path << "\n";

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


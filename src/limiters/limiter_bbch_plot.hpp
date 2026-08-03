/**
 * @file limiter_bbch_plot.hpp
 * @brief Post-Processing BBCH Interface Smoothing Filter for VTK Plot Output Exports.
 */

#pragma once
#include "../core/cell.hpp"
#include "../core/basis.hpp"
#include "../core/parameters.hpp"
#include "../core/solver.hpp"
#include "limiter_common.hpp"
#include "limiter_bbch.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace Limiters {

/**
 * @struct SmoothBBPlotField
 * @brief Encapsulates C0 interface-aligned BB control points for all cells for VTK export.
 */
struct SmoothBBPlotField {
    std::vector<std::vector<std::vector<std::vector<double>>>> C; // [c_idx][v][j][i]
    int npts = 0;
    int P = 0;
    double M_BB2Nodal[MAX_LIM_PTS][MAX_LIM_PTS];
    double M_Nodal2BB[MAX_LIM_PTS][MAX_LIM_PTS];
    double w_BB[MAX_LIM_PTS][MAX_LIM_PTS];
    double sum_w_sq = 0.0;

    /**
     * @brief Build C0 interface-smoothed BB control points for all cells.
     */
    void build(const Solver& solver) {
        npts = solver.p.N_PTS;
        if (npts > MAX_LIM_PTS) return;
        P = npts - 1;
        const size_t num_cells = solver.cells.size();
        C.assign(num_cells, std::vector<std::vector<std::vector<double>>>(4, std::vector<std::vector<double>>(npts, std::vector<double>(npts, 0.0))));

        // 1. Build BB transformation matrices
        for (int j = 0; j < npts; ++j) {
            double t = 0.5 * (solver.basis.z[j] + 1.0);
            for (int i = 0; i < npts; ++i) {
                M_BB2Nodal[j][i] = bernstein_B(i, P, t);
            }
        }
        invert_matrix(npts, M_BB2Nodal, M_Nodal2BB);

        sum_w_sq = 0.0;
        for (int i = 0; i < npts; ++i) {
            for (int j = 0; j < npts; ++j) {
                w_BB[i][j] = 0.0;
                for (int my = 0; my < npts; ++my) {
                    for (int mx = 0; mx < npts; ++mx) {
                        double weight = 0.25 * solver.basis.w[my] * solver.basis.w[mx];
                        w_BB[i][j] += weight * M_BB2Nodal[my][j] * M_BB2Nodal[mx][i];
                    }
                }
                sum_w_sq += w_BB[i][j] * w_BB[i][j];
            }
        }

        // 2. Transform nodal states to BB control points C[c_idx][v][j][i]
        #pragma omp parallel for schedule(static)
        for (size_t c_idx = 0; c_idx < num_cells; ++c_idx) {
            Cell* c = solver.cells[c_idx];
            for (int v = 0; v < 4; ++v) {
                double tmp[MAX_LIM_PTS][MAX_LIM_PTS];
                for (int iy = 0; iy < npts; ++iy) {
                    for (int i = 0; i < npts; ++i) {
                        tmp[iy][i] = 0.0;
                        for (int ix = 0; ix < npts; ++ix) {
                            tmp[iy][i] += M_Nodal2BB[i][ix] * c->get_U(v, iy, ix, npts);
                        }
                    }
                }
                for (int j = 0; j < npts; ++j) {
                    for (int i = 0; i < npts; ++i) {
                        C[c_idx][v][j][i] = 0.0;
                        for (int iy = 0; iy < npts; ++iy) {
                            C[c_idx][v][j][i] += M_Nodal2BB[j][iy] * tmp[iy][i];
                        }
                    }
                }
            }
        }

        // 3. Perform C0 boundary control point harmonization between neighbor cells
        #pragma omp parallel for schedule(static)
        for (size_t c_idx = 0; c_idx < num_cells; ++c_idx) {
            Cell* c = solver.cells[c_idx];
            // Left face (i = 0) with neighbor 0
            if (c->neighbors[0]) {
                int n_idx = c->neighbors[0]->cell_index;
                if (n_idx >= 0 && static_cast<size_t>(n_idx) < num_cells) {
                    for (int v = 0; v < 4; ++v) {
                        for (int j = 0; j < npts; ++j) {
                            double avg = 0.5 * (C[c_idx][v][j][0] + C[n_idx][v][j][P]);
                            C[c_idx][v][j][0] = avg;
                        }
                    }
                }
            }
            // Right face (i = P) with neighbor 1
            if (c->neighbors[1]) {
                int n_idx = c->neighbors[1]->cell_index;
                if (n_idx >= 0 && static_cast<size_t>(n_idx) < num_cells) {
                    for (int v = 0; v < 4; ++v) {
                        for (int j = 0; j < npts; ++j) {
                            double avg = 0.5 * (C[c_idx][v][j][P] + C[n_idx][v][j][0]);
                            C[c_idx][v][j][P] = avg;
                        }
                    }
                }
            }
            // Bottom face (j = 0) with neighbor 2
            if (c->neighbors[2]) {
                int n_idx = c->neighbors[2]->cell_index;
                if (n_idx >= 0 && static_cast<size_t>(n_idx) < num_cells) {
                    for (int v = 0; v < 4; ++v) {
                        for (int i = 0; i < npts; ++i) {
                            double avg = 0.5 * (C[c_idx][v][0][i] + C[n_idx][v][P][i]);
                            C[c_idx][v][0][i] = avg;
                        }
                    }
                }
            }
            // Top face (j = P) with neighbor 3
            if (c->neighbors[3]) {
                int n_idx = c->neighbors[3]->cell_index;
                if (n_idx >= 0 && static_cast<size_t>(n_idx) < num_cells) {
                    for (int v = 0; v < 4; ++v) {
                        for (int i = 0; i < npts; ++i) {
                            double avg = 0.5 * (C[c_idx][v][P][i] + C[n_idx][v][0][i]);
                            C[c_idx][v][P][i] = avg;
                        }
                    }
                }
            }
        }

        // 4. Apply exact L2 conservative projection to preserve cell averages
        #pragma omp parallel for schedule(static)
        for (size_t c_idx = 0; c_idx < num_cells; ++c_idx) {
            Cell* c = solver.cells[c_idx];
            double r_avg, ru_avg, rv_avg, E_avg;
            Limiters::compute_cell_average(*c, solver.basis, r_avg, ru_avg, rv_avg, E_avg, npts);
            double U_avg[4] = { r_avg, ru_avg, rv_avg, E_avg };

            for (int v = 0; v < 4; ++v) {
                double calc_avg = 0.0;
                for (int j = 0; j < npts; ++j) {
                    for (int i = 0; i < npts; ++i) {
                        calc_avg += w_BB[i][j] * C[c_idx][v][j][i];
                    }
                }
                double delta = calc_avg - U_avg[v];
                for (int j = 0; j < npts; ++j) {
                    for (int i = 0; i < npts; ++i) {
                        C[c_idx][v][j][i] -= (delta / sum_w_sq) * w_BB[i][j];
                    }
                }
            }
        }
    }

    /**
     * @brief Evaluate smoothed variable v at reference coordinate (r, s) in [-1, 1]^2.
     */
    double eval(size_t c_idx, int v, double r, double s) const {
        if (c_idx >= C.size()) return 0.0;
        double tr = 0.5 * (r + 1.0);
        double ts = 0.5 * (s + 1.0);
        double val = 0.0;
        for (int j = 0; j < npts; ++j) {
            double B_j = bernstein_B(j, P, ts);
            for (int i = 0; i < npts; ++i) {
                double B_i = bernstein_B(i, P, tr);
                val += C[c_idx][v][j][i] * B_i * B_j;
            }
        }
        return val;
    }
};

} // namespace Limiters

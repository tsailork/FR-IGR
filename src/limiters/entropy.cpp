/**
 * @file entropy.cpp
 * @brief Implementation of the Entropy Minimum-Preservation Limiter.
 *
 * For each computational cell, the local entropy floor is computed as the minimum
 * specific entropy:
 * \f[ s = \frac{p}{\rho^\gamma} \f]
 * across the cell's own interior solution points AND all solution points in the
 * 4 face-neighboring cells (including multiblock neighbors).
 *
 * This implementation is parallelized across elements using OpenMP.
 */

#include "entropy.hpp"
#include "../core/solver.hpp"
#include "limiter_common.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Apply the entropy minimum-preserving limiter to the entire solver domain.
 *
 * Enforces specific entropy bounds to maintain positivity and stability in high-order
 * FR methods without introducing excessive numerical dissipation.
 *
 * @param[in,out] solver The active Solver context
 * @return LimiterStats Statistics containing the number of cells limited and average theta scaling factor
 */
Limiters::LimiterStats Limiters::apply_entropy_limiter(Solver &solver) {
  const Parameters &p = solver.p;
  const Basis &basis = solver.basis;

  // --- 1. Pre-calculate min entropy for every cell in every block
  std::vector<std::vector<double>> blocks_s_min;
  for (auto& b : solver.blocks) {
    std::vector<double> s_min(b.ny * b.nx);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ey = 0; ey < b.ny; ++ey) {
      for (int ex = 0; ex < b.nx; ++ex) {
        s_min[ey * b.nx + ex] = min_entropy_in_cell(b.U, p, ey, ex);
      }
    }
    blocks_s_min.push_back(std::move(s_min));
  }

  // --- 2. Apply limiting block-by-block ---
  int num_limited = 0;
  double sum_theta = 0.0;

  for (size_t bid = 0; bid < solver.blocks.size(); ++bid) {
    auto& b = solver.blocks[bid];
    auto& s_min = blocks_s_min[bid];

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:num_limited, sum_theta)
    for (int ey = 0; ey < b.ny; ++ey) {
      for (int ex = 0; ex < b.nx; ++ex) {

        // --- Gather neighbourhood entropy minimum ---
        double s_floor = s_min[ey * b.nx + ex];

        // Left
        if (ex > 0) {
          s_floor = std::min(s_floor, s_min[ey * b.nx + (ex - 1)]);
        } else {
          if (b.ni_l.id != -1) {
            int nid = b.ni_l.id;
            int nex = (b.ni_l.face == 'L') ? 0 : solver.blocks[nid].nx - 1;
            s_floor = std::min(s_floor, blocks_s_min[nid][ey * solver.blocks[nid].nx + nex]);
          } else if (b.bc_l == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[ey * b.nx + (b.nx - 1)]);
          }
        }

        // Right
        if (ex < b.nx - 1) {
          s_floor = std::min(s_floor, s_min[ey * b.nx + (ex + 1)]);
        } else {
          if (b.ni_r.id != -1) {
            int nid = b.ni_r.id;
            int nex = (b.ni_r.face == 'L') ? 0 : solver.blocks[nid].nx - 1;
            s_floor = std::min(s_floor, blocks_s_min[nid][ey * solver.blocks[nid].nx + nex]);
          } else if (b.bc_r == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[ey * b.nx + 0]);
          }
        }

        // Bottom
        if (ey > 0) {
          s_floor = std::min(s_floor, s_min[(ey - 1) * b.nx + ex]);
        } else {
          if (b.ni_b.id != -1) {
            int nid = b.ni_b.id;
            int ney = (b.ni_b.face == 'B') ? 0 : solver.blocks[nid].ny - 1;
            s_floor = std::min(s_floor, blocks_s_min[nid][ney * solver.blocks[nid].nx + ex]);
          } else if (b.bc_b == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[(b.ny - 1) * b.nx + ex]);
          }
        }

        // Top
        if (ey < b.ny - 1) {
          s_floor = std::min(s_floor, s_min[(ey + 1) * b.nx + ex]);
        } else {
          if (b.ni_t.id != -1) {
            int nid = b.ni_t.id;
            int ney = (b.ni_t.face == 'B') ? 0 : solver.blocks[nid].ny - 1;
            s_floor = std::min(s_floor, blocks_s_min[nid][ney * solver.blocks[nid].nx + ex]);
          } else if (b.bc_t == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[0 * b.nx + ex]);
          }
        }

        // Lower s_floor to avoid being overly dissipative
        s_floor -= 1.0E-4;
        if (s_floor < 1.0E-14) s_floor = 1.0E-14;

        // --- Cell average ---
        double r_avg, ru_avg, rv_avg, E_avg;
        compute_cell_average(b.U, basis, ey, ex, r_avg, ru_avg, rv_avg, E_avg);

        // --- Extrapolate face values for checking ---
        double face_pts[MAX_FACE_PTS][4];
        int n_face = extrapolate_face_values(b.U, basis, ey, ex, face_pts);

        // --- Find worst θ ---
        double theta_s = 1.0;

        // Check interior solution points
        for (int iy = 0; iy < b.U.npts; ++iy)
          for (int ix = 0; ix < b.U.npts; ++ix) {
            double s = specific_entropy(
                b.U(0, ey, ex, iy, ix), b.U(1, ey, ex, iy, ix), b.U(2, ey, ex, iy, ix),
                b.U(3, ey, ex, iy, ix), p.GAMMA);
            if (s < s_floor)
              theta_s = std::min(
                  theta_s, bisect_for_theta(
                               b.U(0, ey, ex, iy, ix), b.U(1, ey, ex, iy, ix),
                               b.U(2, ey, ex, iy, ix), b.U(3, ey, ex, iy, ix), r_avg,
                               ru_avg, rv_avg, E_avg, p.GAMMA, s_floor, false));
          }

        // Check face-extrapolated points (Zhang-Shu requirement for GL nodes)
        for (int f = 0; f < n_face; ++f) {
          double s = specific_entropy(face_pts[f][0], face_pts[f][1],
                                      face_pts[f][2], face_pts[f][3], p.GAMMA);
          if (s < s_floor)
            theta_s =
                std::min(theta_s, bisect_for_theta(face_pts[f][0], face_pts[f][1],
                                                   face_pts[f][2], face_pts[f][3],
                                                   r_avg, ru_avg, rv_avg, E_avg,
                                                   p.GAMMA, s_floor, false));
        }

        // --- Apply scaling ---
        if (theta_s < 1.0) {
          for (int iy = 0; iy < b.U.npts; ++iy)
            for (int ix = 0; ix < b.U.npts; ++ix) {
              b.U(0, ey, ex, iy, ix) =
                  theta_s * (b.U(0, ey, ex, iy, ix) - r_avg) + r_avg;
              b.U(1, ey, ex, iy, ix) =
                  theta_s * (b.U(1, ey, ex, iy, ix) - ru_avg) + ru_avg;
              b.U(2, ey, ex, iy, ix) =
                  theta_s * (b.U(2, ey, ex, iy, ix) - rv_avg) + rv_avg;
              b.U(3, ey, ex, iy, ix) =
                  theta_s * (b.U(3, ey, ex, iy, ix) - E_avg) + E_avg;
            }
          num_limited++;
          sum_theta += theta_s;
        }
      }
    }
  }

  LimiterStats stats;
  stats.num_limited = num_limited;
  stats.sum_theta = sum_theta;
  return stats;
}

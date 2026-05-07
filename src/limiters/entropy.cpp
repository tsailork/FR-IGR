/// @file entropy.cpp
/// @brief Entropy minimum-preservation limiter implementation.
///
/// For each element, the local entropy floor is the minimum specific entropy
/// s = p / ρ^γ  taken over the cell's own solution points AND all solution
/// points in the 4 face-neighbouring cells.
///
/// OpenMP: parallelised over elements.

#include "entropy.hpp"
#include "../core/solver.hpp"
#include "limiter_common.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

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
          std::string bc = b.bc_l;
          if (bc.find(':') != std::string::npos) {
            int nid = std::stoi(bc.substr(0, bc.find(':')));
            char face = bc[bc.find(':') + 1];
            if (face == 'R') s_floor = std::min(s_floor, blocks_s_min[nid][ey * solver.blocks[nid].nx + (solver.blocks[nid].nx - 1)]);
          } else if (bc == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[ey * b.nx + (b.nx - 1)]);
          }
        }

        // Right
        if (ex < b.nx - 1) {
          s_floor = std::min(s_floor, s_min[ey * b.nx + (ex + 1)]);
        } else {
          std::string bc = b.bc_r;
          if (bc.find(':') != std::string::npos) {
            int nid = std::stoi(bc.substr(0, bc.find(':')));
            char face = bc[bc.find(':') + 1];
            if (face == 'L') s_floor = std::min(s_floor, blocks_s_min[nid][ey * solver.blocks[nid].nx + 0]);
          } else if (bc == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[ey * b.nx + 0]);
          }
        }

        // Bottom
        if (ey > 0) {
          s_floor = std::min(s_floor, s_min[(ey - 1) * b.nx + ex]);
        } else {
          std::string bc = b.bc_b;
          if (bc.find(':') != std::string::npos) {
            int nid = std::stoi(bc.substr(0, bc.find(':')));
            char face = bc[bc.find(':') + 1];
            if (face == 'T') s_floor = std::min(s_floor, blocks_s_min[nid][(solver.blocks[nid].ny - 1) * solver.blocks[nid].nx + ex]);
          } else if (bc == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[(b.ny - 1) * b.nx + ex]);
          }
        }

        // Top
        if (ey < b.ny - 1) {
          s_floor = std::min(s_floor, s_min[(ey + 1) * b.nx + ex]);
        } else {
          std::string bc = b.bc_t;
          if (bc.find(':') != std::string::npos) {
            int nid = std::stoi(bc.substr(0, bc.find(':')));
            char face = bc[bc.find(':') + 1];
            if (face == 'B') s_floor = std::min(s_floor, blocks_s_min[nid][0 * solver.blocks[nid].nx + ex]);
          } else if (bc == "PERIODIC") {
            s_floor = std::min(s_floor, s_min[0 * b.nx + ex]);
          }
        }

        // Lower s_floor to avoid being overly dissipative
        s_floor -= 1.0E-4;

        // --- Cell average ---
        double r_avg, ru_avg, rv_avg, E_avg;
        compute_cell_average(b.U, basis, p, ey, ex, r_avg, ru_avg, rv_avg, E_avg);

        // --- Extrapolate face values for checking ---
        double face_pts[MAX_FACE_PTS][4];
        int n_face = extrapolate_face_values(b.U, basis, p, ey, ex, face_pts);

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

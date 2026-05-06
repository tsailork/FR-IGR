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
  State &U = solver.U;

  // --- 1. Pre-calculate min entropy for every cell to avoid race conditions
  // --- Using a buffer ensures that the neighborhood minimum is based on the
  // unmodified state of all neighbors, even in a parallel loop.
  std::vector<double> cell_s_min(p.N_ELEM_Y * p.N_ELEM_X);
#pragma omp parallel for collapse(2) schedule(static)
  for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
    for (int ex = 0; ex < p.N_ELEM_X; ++ex) {
      cell_s_min[ey * p.N_ELEM_X + ex] = min_entropy_in_cell(U, p, ey, ex);
    }
  }

// --- 2. Apply limiting element-by-element ---
  int num_limited = 0;
  double sum_theta = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:num_limited, sum_theta)
  for (int ey = 0; ey < p.N_ELEM_Y; ++ey) {
    for (int ex = 0; ex < p.N_ELEM_X; ++ex) {

      // --- Gather neighbourhood entropy minimum ---
      double s_floor = cell_s_min[ey * p.N_ELEM_X + ex];

      // Left
      if (ex > 0)
        s_floor = std::min(s_floor, cell_s_min[ey * p.N_ELEM_X + (ex - 1)]);
      else if (p.BC_L == "PERIODIC")
        s_floor =
            std::min(s_floor, cell_s_min[ey * p.N_ELEM_X + (p.N_ELEM_X - 1)]);

      // Right
      if (ex < p.N_ELEM_X - 1)
        s_floor = std::min(s_floor, cell_s_min[ey * p.N_ELEM_X + (ex + 1)]);
      else if (p.BC_R == "PERIODIC")
        s_floor = std::min(s_floor, cell_s_min[ey * p.N_ELEM_X + 0]);

      // Bottom
      if (ey > 0)
        s_floor = std::min(s_floor, cell_s_min[(ey - 1) * p.N_ELEM_X + ex]);
      else if (p.BC_B == "PERIODIC")
        s_floor =
            std::min(s_floor, cell_s_min[(p.N_ELEM_Y - 1) * p.N_ELEM_X + ex]);

      // Top
      if (ey < p.N_ELEM_Y - 1)
        s_floor = std::min(s_floor, cell_s_min[(ey + 1) * p.N_ELEM_X + ex]);
      else if (p.BC_T == "PERIODIC")
        s_floor = std::min(s_floor, cell_s_min[0 * p.N_ELEM_X + ex]);

      // Lower s_floor to avoid being overly dissipative
      s_floor -= 1.0E-4;

      // --- Cell average ---
      double r_avg, ru_avg, rv_avg, E_avg;
      compute_cell_average(U, basis, p, ey, ex, r_avg, ru_avg, rv_avg, E_avg);

      // --- Extrapolate face values for checking ---
      double face_pts[MAX_FACE_PTS][4];
      int n_face = extrapolate_face_values(U, basis, p, ey, ex, face_pts);

      // --- Find worst θ ---
      double theta_s = 1.0;

      // Check interior solution points
      for (int iy = 0; iy < p.N_PTS; ++iy)
        for (int ix = 0; ix < p.N_PTS; ++ix) {
          double s = specific_entropy(
              U(0, ey, ex, iy, ix), U(1, ey, ex, iy, ix), U(2, ey, ex, iy, ix),
              U(3, ey, ex, iy, ix), p.GAMMA);
          if (s < s_floor)
            theta_s = std::min(
                theta_s, bisect_for_theta(
                             U(0, ey, ex, iy, ix), U(1, ey, ex, iy, ix),
                             U(2, ey, ex, iy, ix), U(3, ey, ex, iy, ix), r_avg,
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
        for (int iy = 0; iy < p.N_PTS; ++iy)
          for (int ix = 0; ix < p.N_PTS; ++ix) {
            U(0, ey, ex, iy, ix) =
                theta_s * (U(0, ey, ex, iy, ix) - r_avg) + r_avg;
            U(1, ey, ex, iy, ix) =
                theta_s * (U(1, ey, ex, iy, ix) - ru_avg) + ru_avg;
            U(2, ey, ex, iy, ix) =
                theta_s * (U(2, ey, ex, iy, ix) - rv_avg) + rv_avg;
            U(3, ey, ex, iy, ix) =
                theta_s * (U(3, ey, ex, iy, ix) - E_avg) + E_avg;
          }
        num_limited++;
        sum_theta += theta_s;
      }
    }
  }
  
  LimiterStats stats;
  stats.num_limited = num_limited;
  stats.sum_theta = sum_theta;
  return stats;
}

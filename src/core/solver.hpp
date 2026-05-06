/// @file solver.hpp
/// @brief Solver class declaration — the central engine for FR-IGR.
///
/// This header is intentionally thin.  All method bodies live in separate
/// .cpp files organised by subsystem (flux/, igr/, time/, etc.).

#pragma once

#include "basis.hpp"
#include "parameters.hpp"
#include "state.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include "../limiters/limiter_common.hpp"

/// Maximum polynomial degree supported (P = 0..3  →  up to 4 points).
constexpr int MAX_PTS = 4;

class Solver {
public:
    const Parameters& p;
    State  U;      ///< Conservative variables  [ρ, ρu, ρv, E].
    State  RHS;    ///< Right-hand-side update vector.
    Basis  basis;  ///< 1-D basis (tensor-product).

    // -- Geometry --
    double dx, dy;

    // -- Entropic pressure (IGR) buffers --
    std::vector<double> sigma_field;   ///< Regularisation field Σ.
    std::vector<double> S_buf;         ///< Sensor source S.
    std::vector<double> sigma_xy_buf;  ///< ADI XY-pass result.
    std::vector<double> sigma_yx_buf;  ///< ADI YX-pass result.
    std::vector<double> sigma_RHS;     ///< Parabolic evolution RHS.
    std::vector<double> qx_buf;        ///< BR2 auxiliary gradient (x).
    std::vector<double> qy_buf;        ///< BR2 auxiliary gradient (y).

    // -- Diagnostics --
    Limiters::LimiterStats current_limiter_stats;

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------
    explicit Solver(const Parameters& params);

    // -----------------------------------------------------------------
    // Indexing helpers
    // -----------------------------------------------------------------

    /// Map (ey, ex, iy, ix) to a flat buffer index.
    inline int get_flat_idx(int ey, int ex, int iy, int ix) const {
        return ey * (p.N_ELEM_X * p.N_PTS * p.N_PTS)
             + ex * (p.N_PTS * p.N_PTS)
             + iy * p.N_PTS + ix;
    }

    /// Clamp a value to a small positive floor (avoids division by zero).
    static inline double clamp_positivity(double val) {
        return std::max(1e-10, val);
    }

    // -----------------------------------------------------------------
    // Boundary helpers  (src/boundary/)
    // -----------------------------------------------------------------
    void get_neigh_state_x(int ey, int ex, int iy, bool is_right,
                           const double* face_state, double sig_face,
                           double* neigh_state, double& sig_neigh) const;

    void get_neigh_state_y(int ey, int ex, int ix, bool is_top,
                           const double* face_state, double sig_face,
                           double* neigh_state, double& sig_neigh) const;

    // -----------------------------------------------------------------
    // Flux reconstruction  (src/flux/)
    // -----------------------------------------------------------------
    void get_flux_pointwise(int ey, int ex, int iy, int ix,
                            double* F, double* G, double sigma) const;
    void solve_riemann(const double* UL, const double* UR, double* F_comm,
                       int dir, double sigl, double sigr) const;
    void sweep_x();
    void sweep_y();

    // -----------------------------------------------------------------
    // IGR / Entropic pressure  (src/igr/)
    // -----------------------------------------------------------------
    void compute_sensor_source();
    void solve_adi_pass(const std::vector<double>& S,
                        std::vector<double>& Out, bool x_first);
    void compute_igr_parabolic_rhs();
    void compute_entropic_pressure();

    // -----------------------------------------------------------------
    // Time stepping & stability  (src/time/)
    // -----------------------------------------------------------------
    void   check_stability() const;
    double compute_dt() const;
    void   compute_rhs();
    void   step_rk3(double dt);
};

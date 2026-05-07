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

struct NeighborInfo {
    int id = -1;      ///< Neighbor block ID. -1 if none/periodic/wall.
    char face = ' ';  ///< Neighbor face ('L', 'R', 'B', 'T').
    bool is_periodic = false;
    bool is_wall = false;
    bool is_inflow = false;
    bool is_transmissive = false;
};

struct Block {
    int id;
    int nx, ny;
    double dx, dy;
    double x_min, y_min;
    std::string bc_l, bc_r, bc_b, bc_t;
    
    NeighborInfo ni_l, ni_r, ni_b, ni_t;

    State U;
    State RHS;
    std::vector<double> sigma_field;
    std::vector<double> S_buf;
    std::vector<double> sigma_xy_buf;
    std::vector<double> sigma_yx_buf;
    std::vector<double> sigma_RHS;
    std::vector<double> qx_buf;
    std::vector<double> qy_buf;

    Block(const BlockConfig& config, int npts) 
        : id(config.id), nx(config.N_ELEM_X), ny(config.N_ELEM_Y),
          x_min(config.X_MIN), y_min(config.Y_MIN),
          bc_l(config.BC_L), bc_r(config.BC_R), bc_b(config.BC_B), bc_t(config.BC_T),
          U(config.N_ELEM_X, config.N_ELEM_Y, npts),
          RHS(config.N_ELEM_X, config.N_ELEM_Y, npts) 
    {
        dx = (config.X_MAX - config.X_MIN) / nx;
        dy = (config.Y_MAX - config.Y_MIN) / ny;
        
        int n_dofs = nx * ny * npts * npts;
        sigma_field.resize(n_dofs, 0.0);
        S_buf.resize(n_dofs, 0.0);
        sigma_xy_buf.resize(n_dofs, 0.0);
        sigma_yx_buf.resize(n_dofs, 0.0);
        sigma_RHS.resize(n_dofs, 0.0);
        qx_buf.resize(n_dofs, 0.0);
        qy_buf.resize(n_dofs, 0.0);
    }

    inline int get_flat_idx(int ey, int ex, int iy, int ix, int npts) const {
        return ey * (nx * npts * npts) + ex * (npts * npts) + iy * npts + ix;
    }
};

class Solver {
public:
    const Parameters& p;
    std::vector<Block> blocks;
    Basis  basis;  ///< 1-D basis (tensor-product).

    // -- Diagnostics --
    Limiters::LimiterStats current_limiter_stats;

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------
    explicit Solver(const Parameters& params);

    // -----------------------------------------------------------------
    // Boundary helpers  (src/boundary/)
    // -----------------------------------------------------------------
    void get_neigh_state_x(const Block& b, int ey, int ex, int iy, bool is_right,
                           const double* face_state, double sig_face,
                           double* neigh_state, double& sig_neigh) const;

    void get_neigh_state_y(const Block& b, int ey, int ex, int ix, bool is_top,
                           const double* face_state, double sig_face,
                           double* neigh_state, double& sig_neigh) const;

    // -----------------------------------------------------------------
    // Flux reconstruction  (src/flux/)
    // -----------------------------------------------------------------
    void get_flux_pointwise(const Block& b, int ey, int ex, int iy, int ix,
                            double* F, double* G, double sigma) const;
    void solve_riemann(const double* UL, const double* UR, double* F_comm,
                       int dir, double sigl, double sigr) const;
    void sweep_x();
    void sweep_y();

    // -----------------------------------------------------------------
    // IGR / Entropic pressure  (src/igr/)
    // -----------------------------------------------------------------
    void compute_sensor_source();
    void solve_adi_pass(Block& b, const std::vector<double>& S,
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

/**
 * @file esdirk34.hpp
 * @brief 4-Stage 3rd-Order L-Stable ESDIRK34 Implicit Time Integrator & JFNK GMRES Engine.
 */

#pragma once

#include "../core/basis.hpp"
#include "../core/cell.hpp"
#include "../core/parameters.hpp"
#include "implicit_constants.hpp"
#include "implicit_precond.hpp"
#include "implicit_ilu.hpp"
#include <vector>
#include <functional>
#include <cmath>

namespace fr::implicit {

/**
 * @struct ESDIRK34Tableau
 * @brief 4-stage 3rd-order L-stable ESDIRK Butcher tableau.
 */
struct ESDIRK34Tableau {
    static constexpr int s = 4;
    static constexpr double gamma = 0.4358665215084590;
    static constexpr double c[4] = {0.0, 2.0 * gamma, 0.5, 1.0};
    static constexpr double A[4][4] = {
        {0.0,                   0.0,             0.0,           0.0},
        {gamma,                 gamma,           0.0,           0.0},
        {0.5 - 0.25 * gamma,   -0.25 * gamma,    gamma,         0.0},
        {0.125,                 0.125,           0.75 - gamma,  gamma}
    };
    static constexpr double b[4] = {0.125, 0.125, 0.75 - gamma, gamma};
};

/**
 * @struct ImplicitStats
 * @brief Performance and diagnostic metrics for implicit time steps.
 */
struct ImplicitStats {
    int step = 0;
    double current_cfl = 0.0;
    int total_newton_iters = 0;
    int total_gmres_iters = 0;
    int total_res_evals = 0;
    double final_stage_res = 0.0;
    double step_wall_time_ms = 0.0;
};

/**
 * @brief Evaluates physical admissibility (density and pressure positivity) on solution nodes and face extrapolations.
 */
[[nodiscard]] bool check_realizability_2d(
    const std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    double gamma_fluid,
    double pos_eps = REALIZABILITY_EPS
);

/**
 * @brief Preconditioned GMRES linear system solver: J * delta_u = -G.
 * Solves A * x = b with preconditioner M_op^{-1}.
 */
template<typename PrecondType>
bool gmres_solve(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& matvec,
    const std::vector<double>& b,
    std::vector<double>& x,
    const PrecondType& M_op,
    double rtol = DEFAULT_GMRES_TOL,
    double atol = REALIZABILITY_EPS,
    int max_iters = DEFAULT_GMRES_MAX_ITERS,
    int restart = DEFAULT_GMRES_RESTART,
    int* gmres_iters_out = nullptr
);

/**
 * @brief Direct Block-Jacobi Newton Stage Solver for G_i(U) = U - H_i - gamma*dt*R(U) = 0.
 */
[[nodiscard]] bool solve_direct_block_jacobi_stage_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    const std::vector<double>& H_i,
    double gamma_dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    const BlockJacobiPreconditioner2D& M_op,
    ImplicitStats& stats
);

/**
 * @brief Non-linear stage solver for a single ESDIRK stage using JFNK (GMRES + Preconditioner).
 */
template<typename PrecondType>
[[nodiscard]] bool solve_jfnk_stage_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    const std::vector<double>& H_i,
    double gamma_dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    const PrecondType& M_op,
    ImplicitStats& stats
);

/**
 * @brief Direct Block-ILU(0) Newton Stage Solver for G_i(U) = U - H_i - gamma*dt*R(U) = 0.
 */
[[nodiscard]] bool solve_direct_ilu_stage_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    const std::vector<double>& H_i,
    double gamma_dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    const BlockILUPreconditioner2D& M_op,
    ImplicitStats& stats
);

/**
 * @brief Full ESDIRK34 implicit time step execution on 2D mesh.
 */
bool step_esdirk34_2d(
    std::vector<CellDim<2>*>& cells,
    const Basis& basis,
    const Parameters& params,
    double dt,
    const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective,
    BlockJacobiPreconditioner2D& precond,
    BlockILUPreconditioner2D& ilu_precond,
    int& step_counter,
    ImplicitStats& stats_out
);

} // namespace fr::implicit

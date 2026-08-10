/**
 * @file implicit_integrator.hpp
 * @brief Encapsulated Manager Engine for the FR-IGR Implicit Subsystem.
 */

#pragma once

#include "../core/basis.hpp"
#include "../core/cell.hpp"
#include "../core/parameters.hpp"
#include "implicit_constants.hpp"
#include "implicit_precond.hpp"
#include "implicit_ilu.hpp"
#include "esdirk34.hpp"
#include <vector>
#include <functional>
#include <memory>

namespace fr::implicit {

/**
 * @class ImplicitIntegrator2D
 * @brief Manages implicit time integration, preconditioners, CFL adaptation, and step retries.
 *
 * Encapsulates Block-Jacobi and Block-ILU preconditioner matrices, handles adaptive CFL
 * ramping/scaling, and executes ESDIRK34 implicit steps with automatic failure recovery.
 */
class ImplicitIntegrator2D {
public:
    BlockJacobiPreconditioner2D precond_2d;
    BlockILUPreconditioner2D ilu_precond_2d;
    int step_counter = 0;
    double current_implicit_cfl = 1.0;
    ImplicitStats last_stats;

    ImplicitIntegrator2D() = default;

    /**
     * @brief Initializes the implicit integrator engine and CFL parameters.
     */
    void initialize(const Parameters& params);

    /**
     * @brief Queries the active implicit CFL multiplier.
     */
    [[nodiscard]] double get_current_cfl() const noexcept { return current_implicit_cfl; }

    /**
     * @brief Executes one full implicit time step with step retry capability on failure.
     * @return true if step succeeded, false if step failed after maximum retries.
     */
    bool step(
        std::vector<CellDim<2>*>& cells,
        const Basis& basis,
        const Parameters& params,
        double dt,
        const std::function<void(const std::vector<CellDim<2>*>&, std::vector<double>&)>& compute_R_effective
    );
};

} // namespace fr::implicit

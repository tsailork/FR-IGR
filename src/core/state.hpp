/**
 * @file state.hpp
 * @brief Storage and accessor class for the 2D Euler equations conserved variables.
 *
 * Implements a flattened, row-major and variable-major memory layout:
 * \f[ \text{Layout: } [Variable][E_y][E_x][I_y][I_x] \f]
 * Variable indexing convention:
 *  - 0: Density (\f$\rho\f$)
 *  - 1: X-momentum (\f$\rho u\f$)
 *  - 2: Y-momentum (\f$\rho v\f$)
 *  - 3: Total energy density (\f$E\f$)
 *
 * Utilizes a single contiguous std::vector to ensure hardware cache-locality and high-performance access.
 */

#pragma once
#include <vector>
#include "parameters.hpp"

/**
 * @brief Number of conserved variables in the 2D Euler system (density, x-momentum, y-momentum, total energy).
 */
constexpr int N_VARS = 4;

/**
 * @struct State
 * @brief Represents the conservative state field of a computational block.
 *
 * Manages the raw data storage and provides multi-index accessor interfaces.
 */
struct State {
    std::vector<double> data;  ///< Contiguous array of conserved variables across all degrees of freedom.
    int nx;                    ///< Number of elements along the X coordinate direction.
    int ny;                    ///< Number of elements along the Y coordinate direction.
    int npts;                  ///< Number of solution points per coordinate direction within each element.
    int n_dofs_per_var;        ///< Number of spatial degrees of freedom (solution points) per conserved variable.

    /**
     * @brief Construct a zero-initialized state representation for a given element block topology.
     *
     * @param nx Number of elements in the X-direction
     * @param ny Number of elements in the Y-direction
     * @param npts Number of solution points per direction within each element
     */
    explicit State(int nx, int ny, int npts) : nx(nx), ny(ny), npts(npts) {
        n_dofs_per_var = nx * ny * npts * npts;
        data.resize(N_VARS * n_dofs_per_var, 0.0);
    }

    /**
     * @brief Default constructor. Constructs an unallocated state representation.
     */
    State() : nx(0), ny(0), npts(0), n_dofs_per_var(0) {}

    /**
     * @brief 5-index accessor for reading/writing conserved variables.
     *
     * Maps multi-dimensional indexes into a flattened row-major/variable-major index.
     *
     * @param v Conserved variable index (0 to 3)
     * @param ey Element Y coordinate index
     * @param ex Element X coordinate index
     * @param iy Solution point Y index within the element
     * @param ix Solution point X index within the element
     * @return Reference to the specific double value
     */
    inline double& operator()(int v, int ey, int ex, int iy, int ix) {
        return data[v * n_dofs_per_var +
                    ey * (nx * npts * npts) +
                    ex * (npts * npts) +
                    iy * npts + ix];
    }

    /**
     * @brief Const version of the 5-index accessor.
     *
     * @param v Conserved variable index (0 to 3)
     * @param ey Element Y coordinate index
     * @param ex Element X coordinate index
     * @param iy Solution point Y index within the element
     * @param ix Solution point X index within the element
     * @return Read-only copy of the specific double value
     */
    inline double operator()(int v, int ey, int ex, int iy, int ix) const {
        return data[v * n_dofs_per_var +
                    ey * (nx * npts * npts) +
                    ex * (npts * npts) +
                    iy * npts + ix];
    }
};

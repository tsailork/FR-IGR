/**
 * @file state.hpp
 * @brief Storage and accessor class for the Euler equations conserved variables in 2D and 3D.
 *
 * Implements a flattened, row-major and variable-major memory layout.
 *
 * @see Solver
 * @see Cell
 */

#pragma once
#include <vector>
#include "parameters.hpp"

template<int Dim>
struct StateDim;

/**
 * @struct StateDim<2>
 * @brief Represents the conservative state field of a computational block in 2D.
 */
template<>
struct StateDim<2> {
    static constexpr int N_VARS = 4;
    std::vector<double> data;  ///< Contiguous array of conserved variables across all degrees of freedom.
    int nx;                    ///< Number of elements along the X coordinate direction.
    int ny;                    ///< Number of elements along the Y coordinate direction.
    int npts;                  ///< Number of solution points per coordinate direction within each element.
    int n_dofs_per_var;        ///< Number of spatial degrees of freedom (solution points) per conserved variable.

    /**
     * @brief Construct a zero-initialized state representation for a given element block topology.
     */
    explicit StateDim(int nx, int ny, int npts) : nx(nx), ny(ny), npts(npts) {
        n_dofs_per_var = nx * ny * npts * npts;
        data.resize(N_VARS * n_dofs_per_var, 0.0);
    }

    /**
     * @brief Default constructor. Constructs an unallocated state representation.
     */
    StateDim() : nx(0), ny(0), npts(0), n_dofs_per_var(0) {}

    /**
     * @brief 5-index accessor for reading/writing conserved variables.
     */
    inline double& operator()(int v, int ey, int ex, int iy, int ix) {
        return data[v * n_dofs_per_var +
                    ey * (nx * npts * npts) +
                    ex * (npts * npts) +
                    iy * npts + ix];
    }

    /**
     * @brief Const version of the 5-index accessor.
     */
    inline double operator()(int v, int ey, int ex, int iy, int ix) const {
        return data[v * n_dofs_per_var +
                    ey * (nx * npts * npts) +
                    ex * (npts * npts) +
                    iy * npts + ix];
    }
};

/**
 * @struct StateDim<3>
 * @brief Represents the conservative state field of a computational block in 3D.
 */
template<>
struct StateDim<3> {
    static constexpr int N_VARS = 5;
    std::vector<double> data;  ///< Contiguous array of conserved variables across all degrees of freedom.
    int nx;                    ///< Number of elements along the X coordinate direction.
    int ny;                    ///< Number of elements align the Y coordinate direction.
    int nz;                    ///< Number of elements along the Z coordinate direction.
    int npts;                  ///< Number of solution points per coordinate direction within each element.
    int n_dofs_per_var;        ///< Number of spatial degrees of freedom (solution points) per conserved variable.

    /**
     * @brief Construct a zero-initialized state representation for a given element block topology.
     */
    explicit StateDim(int nx, int ny, int nz, int npts) : nx(nx), ny(ny), nz(nz), npts(npts) {
        n_dofs_per_var = nx * ny * nz * npts * npts * npts;
        data.resize(N_VARS * n_dofs_per_var, 0.0);
    }

    /**
     * @brief Default constructor. Constructs an unallocated state representation.
     */
    StateDim() : nx(0), ny(0), nz(0), npts(0), n_dofs_per_var(0) {}

    /**
     * @brief 7-index accessor for reading/writing conserved variables in 3D.
     */
    inline double& operator()(int v, int ez, int ey, int ex, int iz, int iy, int ix) {
        return data[v * n_dofs_per_var +
                    ez * (ny * nx * npts * npts * npts) +
                    ey * (nx * npts * npts * npts) +
                    ex * (npts * npts * npts) +
                    iz * (npts * npts) +
                    iy * npts + ix];
    }

    /**
     * @brief Const version of the 7-index accessor.
     */
    inline double operator()(int v, int ez, int ey, int ex, int iz, int iy, int ix) const {
        return data[v * n_dofs_per_var +
                    ez * (ny * nx * npts * npts * npts) +
                    ey * (nx * npts * npts * npts) +
                    ex * (npts * npts * npts) +
                    iz * (npts * npts) +
                    iy * npts + ix];
    }
};

using State2D = StateDim<2>;
using State3D = StateDim<3>;
using State = State2D;

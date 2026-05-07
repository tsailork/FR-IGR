/// @file state.hpp
/// @brief Conservative variable storage for the 2D Euler equations.
///
/// Layout:  [Variable][Ey][Ex][Iy][Ix]   (row-major, variable-major)
///   Variable indices:  0 = ρ,  1 = ρu,  2 = ρv,  3 = E
///
/// The data is stored in a single flat std::vector for cache-locality.

#pragma once
#include <vector>
#include "parameters.hpp"

/// Number of conserved variables (ρ, ρu, ρv, E).
constexpr int N_VARS = 4;

struct State {
    std::vector<double> data;
    int nx, ny, npts;
    int n_dofs_per_var;

    /// Construct a zero-initialised state for a specific block dimension.
    explicit State(int nx, int ny, int npts) : nx(nx), ny(ny), npts(npts) {
        n_dofs_per_var = nx * ny * npts * npts;
        data.resize(N_VARS * n_dofs_per_var, 0.0);
    }

    /// Default constructor (for temporaries that will be assigned later).
    State() : nx(0), ny(0), npts(0), n_dofs_per_var(0) {}

    /// 5-index accessor:  U(variable, ey, ex, iy, ix).
    inline double& operator()(int v, int ey, int ex, int iy, int ix) {
        return data[v * n_dofs_per_var +
                    ey * (nx * npts * npts) +
                    ex * (npts * npts) +
                    iy * npts + ix];
    }

    /// Const overload.
    inline double operator()(int v, int ey, int ex, int iy, int ix) const {
        return data[v * n_dofs_per_var +
                    ey * (nx * npts * npts) +
                    ex * (npts * npts) +
                    iy * npts + ix];
    }
};

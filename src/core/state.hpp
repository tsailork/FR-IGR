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
    int n_dofs_per_var;
    const Parameters* p;

    /// Construct a zero-initialised state matching the given parameters.
    explicit State(const Parameters& params) : p(&params) {
        n_dofs_per_var = p->N_ELEM_Y * p->N_ELEM_X * p->N_PTS * p->N_PTS;
        data.resize(N_VARS * n_dofs_per_var, 0.0);
    }

    /// Default constructor (for temporaries that will be assigned later).
    State() : n_dofs_per_var(0), p(nullptr) {}

    /// 5-index accessor:  U(variable, ey, ex, iy, ix).
    inline double& operator()(int v, int ey, int ex, int iy, int ix) {
        return data[v * n_dofs_per_var +
                    ey * (p->N_ELEM_X * p->N_PTS * p->N_PTS) +
                    ex * (p->N_PTS * p->N_PTS) +
                    iy * p->N_PTS + ix];
    }

    /// Const overload.
    inline double operator()(int v, int ey, int ex, int iy, int ix) const {
        return data[v * n_dofs_per_var +
                    ey * (p->N_ELEM_X * p->N_PTS * p->N_PTS) +
                    ex * (p->N_PTS * p->N_PTS) +
                    iy * p->N_PTS + ix];
    }
};

#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "parameters.hpp"

// Variable Indices
// 0: rho, 1: rhou, 2: rhov, 3: E
const int N_VARS = 4; 

struct State {
    // Flattened Data: [Var][Ey][Ex][Iy][Ix]
    std::vector<double> data;
    int n_dofs_per_var;
    const Parameters* p;

    State(const Parameters& params) : p(&params) {
        n_dofs_per_var = p->N_ELEM_Y * p->N_ELEM_X * p->N_PTS * p->N_PTS;
        data.resize(N_VARS * n_dofs_per_var, 0.0);
    }

    // Default constructor for temporary states (will be assigned later)
    State() : n_dofs_per_var(0), p(nullptr) {}

    // Fast Indexer
    inline double& operator()(int v, int ey, int ex, int iy, int ix) {
        int idx = v * n_dofs_per_var + 
                  ey * (p->N_ELEM_X * p->N_PTS * p->N_PTS) + 
                  ex * (p->N_PTS * p->N_PTS) + 
                  iy * p->N_PTS + 
                  ix;
        return data[idx];
    }
    
    // Const overload
    inline double operator()(int v, int ey, int ex, int iy, int ix) const {
        int idx = v * n_dofs_per_var + 
                  ey * (p->N_ELEM_X * p->N_PTS * p->N_PTS) + 
                  ex * (p->N_PTS * p->N_PTS) + 
                  iy * p->N_PTS + 
                  ix;
        return data[idx];
    }
};

/**
 * @file entropic_pressure.cpp
 * @brief Top-level dispatch for the entropic pressure computation on decoupled Cells.
 */

#include "../core/solver.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void Solver::compute_entropic_pressure() {
    if (!p.ENABLE_IGR)
        return;

    compute_sensor_source();

    if (p.IGR_TYPE == "PARABOLIC") {
        compute_igr_parabolic_rhs();
    } else {
        std::cerr << "Error: Elliptic ADI IGR is deprecated and removed. Please configure IGR_TYPE = PARABOLIC.\n";
        std::exit(EXIT_FAILURE);
    }

    // Clamp the entropic pressure strictly to the local thermodynamic pressure
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells.size(); ++i) {
        Cell* c = cells[i];
        for (int iy = 0; iy < p.N_PTS; ++iy) {
            for (int ix = 0; ix < p.N_PTS; ++ix) {
                int idx = iy * p.N_PTS + ix;
                double rho = std::max(1e-14, c->get_U(0, iy, ix, p.N_PTS));
                double rhou = c->get_U(1, iy, ix, p.N_PTS);
                double rhov = c->get_U(2, iy, ix, p.N_PTS);
                double E = c->get_U(3, iy, ix, p.N_PTS);
                double press = (p.GAMMA - 1.0) * (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
                if (press < 1e-14) press = 1e-14;
                c->sigma_field[idx] = std::min(c->sigma_field[idx], press);
            }
        }
    }
}

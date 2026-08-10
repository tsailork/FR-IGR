#include "../doctest.h"
#include "../../src/time/implicit_ilu.hpp"
#include "../../src/core/basis.hpp"
#include "../../src/core/cell.hpp"
#include "../../src/core/parameters.hpp"
#include <vector>
#include <memory>

TEST_CASE("BlockILUPreconditioner2D - Build and Apply Test") {
    int P_deg = 1;
    Basis basis(P_deg);
    int npts = P_deg + 1;

    std::vector<CellDim<2>*> cells(4);
    std::vector<std::unique_ptr<CellDim<2>>> storage;

    for (int i = 0; i < 4; ++i) {
        storage.push_back(std::make_unique<CellDim<2>>(npts));
        cells[i] = storage.back().get();
        cells[i]->ex = i % 2;
        cells[i]->ey = i / 2;
        cells[i]->dx = 1.0;
        cells[i]->dy = 1.0;

        for (int node = 0; node < npts * npts; ++node) {
            cells[i]->U[0 * npts * npts + node] = 1.0; // rho
            cells[i]->U[1 * npts * npts + node] = 0.1; // rhou
            cells[i]->U[2 * npts * npts + node] = 0.0; // rhov
            cells[i]->U[3 * npts * npts + node] = 2.5; // E
        }
    }

    Implicit::BlockILUPreconditioner2D ilu;
    ilu.build(cells, basis, 1.4, 0.01, 0.4358665);

    CHECK(ilu.n_cells == 4);
    CHECK(ilu.block_size == 16);

    size_t total_dofs = cells.size() * ilu.block_size;
    std::vector<double> rhs(total_dofs, 1.0);
    std::vector<double> sol(total_dofs, 0.0);

    ilu.apply(rhs, sol);

    for (size_t i = 0; i < total_dofs; ++i) {
        CHECK(std::abs(sol[i]) > 0.0);
    }
}

#include "../doctest.h"
#include "../../src/core/state.hpp"

TEST_CASE("State class allocation and indexing") {
    struct GridSize { int nx, ny; };
    GridSize sizes[] = {{1, 1}, {3, 3}, {5, 2}};
    
    for (int p_deg = 0; p_deg <= 2; ++p_deg) {
        int npts = p_deg + 1;
        for (auto gs : sizes) {
            SUBCASE(("Grid " + std::to_string(gs.nx) + "x" + std::to_string(gs.ny) + " P_DEG " + std::to_string(p_deg)).c_str()) {
                State state(gs.nx, gs.ny, npts);
                
                int expected_dofs_per_var = gs.nx * gs.ny * npts * npts;
                CHECK(state.n_dofs_per_var == expected_dofs_per_var);
                CHECK(state.data.size() == 4 * expected_dofs_per_var);
                
                // Zero initialization check
                bool all_zero = true;
                for (double val : state.data) {
                    if (val != 0.0) {
                        all_zero = false;
                        break;
                    }
                }
                CHECK(all_zero);
                
                // Indexing round-trip and no aliasing
                state(0, 0, 0, 0, 0) = 1.0; // rho
                if (expected_dofs_per_var > 1) {
                    state(3, gs.ny - 1, gs.nx - 1, npts - 1, npts - 1) = 2.0; // E
                }
                
                CHECK(state(0, 0, 0, 0, 0) == 1.0);
                if (expected_dofs_per_var > 1) {
                    CHECK(state(3, gs.ny - 1, gs.nx - 1, npts - 1, npts - 1) == 2.0);
                    // Check no aliasing
                    CHECK(state(0, 0, 0, 0, 0) == 1.0); // still 1.0
                    CHECK(state(1, 0, 0, 0, 0) == 0.0); // momentum x is zero
                }
                
                // Copy semantics
                State state_copy = state;
                CHECK(state_copy.data == state.data);
                
                state_copy(0, 0, 0, 0, 0) = 5.0;
                CHECK(state_copy(0, 0, 0, 0, 0) == 5.0);
                CHECK(state(0, 0, 0, 0, 0) == 1.0); // Original unmodified
            }
        }
    }
}

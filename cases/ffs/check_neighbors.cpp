#include "src/core/parameters.hpp"
#include "src/core/solver.hpp"
#include <iostream>

int main() {
    Parameters p;
    p.load("inputs.dat");
    Solver solver(p);
    solver.initialize_cells();
    solver.setup_cell_connectivity();
    
    int missing_neighbors = 0;
    for (Cell* c : solver.cells) {
        for (int f = 0; f < 4; ++f) {
            if (!c->is_boundary[f] && c->neighbors[f] == nullptr) {
                std::cout << "Cell in block " << c->block_id << " (ex=" << c->ex << ", ey=" << c->ey << ") face " << f << " has NO neighbor and is NOT a boundary!\n";
                missing_neighbors++;
            }
        }
    }
    std::cout << "Total missing neighbors: " << missing_neighbors << "\n";
    return 0;
}

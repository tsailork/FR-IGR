/**
 * @file sfc_partitioner.hpp
 * @brief Hilbert Space-Filling Curve (SFC) dynamic load balancer for Quadtree/Octree elements.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>
#include "cell.hpp"

namespace fr::core {

/**
 * @struct SFCPartitioner
 * @brief Computes 1D Hilbert SFC orderings for 2D/3D leaf cells to optimize OpenMP thread load distribution.
 */
class SFCPartitioner {
public:
    /**
     * @brief Compute 2D Hilbert curve index for (x, y) coordinates.
     */
    [[nodiscard]] static uint64_t hilbert_2d(uint32_t x, uint32_t y, uint32_t bits = 16) noexcept {
        uint64_t d = 0;
        for (int s = bits - 1; s >= 0; --s) {
            uint32_t rx = (x >> s) & 1;
            uint32_t ry = (y >> s) & 1;
            d += (static_cast<uint64_t>(3 * rx ^ ry) << (2 * s));
            if (ry == 0) {
                if (rx == 1) {
                    x = (1U << s) - 1 - x;
                    y = (1U << s) - 1 - y;
                }
                std::swap(x, y);
            }
        }
        return d;
    }

    /**
     * @brief Sort cell pointers along 1D Hilbert space-filling curve.
     */
    static void sort_cells_2d(std::vector<Cell*>& cells) {
        std::sort(cells.begin(), cells.end(), [](const Cell* a, const Cell* b) {
            uint32_t ax = static_cast<uint32_t>(a->ex);
            uint32_t ay = static_cast<uint32_t>(a->ey);
            uint32_t bx = static_cast<uint32_t>(b->ex);
            uint32_t by = static_cast<uint32_t>(b->ey);
            return hilbert_2d(ax, ay) < hilbert_2d(bx, by);
        });
    }
};

} // namespace fr::core

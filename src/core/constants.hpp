#pragma once

#include <cstddef>
#include <cstdint>

namespace fr {
namespace constants {

    /// Universal minimum density/pressure positivity floor
    inline constexpr double POS_LIMITER_EPS = 1e-12;

    /// Epsilon floor for division-by-zero protection in wave speeds and denominators
    inline constexpr double EPS_DENOMINATOR = 1e-12;

    /// Epsilon floor for Ducros shock sensor denominator protection
    inline constexpr double EPS_DUCROS = 1e-30;

    /// Node location matching tolerance
    inline constexpr double EPS_NODE_MATCH = 1e-12;

    /// Quantitative symmetry tolerance
    inline constexpr double SYMMETRY_TOL = 1e-10;

    /// Mathematical constant PI
    inline constexpr double PI = 3.14159265358979323846;

    /// Face boundary directions across 2D/3D elements
    enum class Face : int32_t {
        Left   = 0,
        Right  = 1,
        Bottom = 2,
        Top    = 3,
        Front  = 4,
        Back   = 5
    };

    /// Spatial coordinate directions
    enum class Direction : int32_t {
        X = 0,
        Y = 1,
        Z = 2
    };

} // namespace constants
} // namespace fr

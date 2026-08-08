#include "../doctest.h"
#include "../../src/core/geometry.hpp"
#include <fstream>
#include <cmath>

TEST_CASE("fr::geometry::CadSdfEngine 3D CAD Mesh Parsing and BVH SDF Queries") {
    // 1. Generate temporary ASCII STL file of a unit cube [0, 1]^3
    std::string test_stl = "test_unit_cube.stl";
    {
        std::ofstream os(test_stl);
        os << "solid cube\n";
        // Front face (z=1)
        os << "facet normal 0 0 1\n  outer loop\n    vertex 0 0 1\n    vertex 1 0 1\n    vertex 1 1 1\n  endloop\nendfacet\n";
        os << "facet normal 0 0 1\n  outer loop\n    vertex 0 0 1\n    vertex 1 1 1\n    vertex 0 1 1\n  endloop\nendfacet\n";
        // Back face (z=0)
        os << "facet normal 0 0 -1\n  outer loop\n    vertex 0 0 0\n    vertex 1 1 0\n    vertex 1 0 0\n  endloop\nendfacet\n";
        os << "facet normal 0 0 -1\n  outer loop\n    vertex 0 0 0\n    vertex 0 1 0\n    vertex 1 1 0\n  endloop\nendfacet\n";
        // Left face (x=0)
        os << "facet normal -1 0 0\n  outer loop\n    vertex 0 0 0\n    vertex 0 0 1\n    vertex 0 1 1\n  endloop\nendfacet\n";
        os << "facet normal -1 0 0\n  outer loop\n    vertex 0 0 0\n    vertex 0 1 1\n    vertex 0 1 0\n  endloop\nendfacet\n";
        // Right face (x=1)
        os << "facet normal 1 0 0\n  outer loop\n    vertex 1 0 0\n    vertex 1 1 1\n    vertex 1 0 1\n  endloop\nendfacet\n";
        os << "facet normal 1 0 0\n  outer loop\n    vertex 1 0 0\n    vertex 1 1 0\n    vertex 1 1 1\n  endloop\nendfacet\n";
        // Top face (y=1)
        os << "facet normal 0 1 0\n  outer loop\n    vertex 0 1 0\n    vertex 0 1 1\n    vertex 1 1 1\n  endloop\nendfacet\n";
        os << "facet normal 0 1 0\n  outer loop\n    vertex 0 1 0\n    vertex 1 1 1\n    vertex 1 1 0\n  endloop\nendfacet\n";
        // Bottom face (y=0)
        os << "facet normal 0 -1 0\n  outer loop\n    vertex 0 0 0\n    vertex 1 0 1\n    vertex 0 0 1\n  endloop\nendfacet\n";
        os << "facet normal 0 -1 0\n  outer loop\n    vertex 0 0 0\n    vertex 1 0 0\n    vertex 1 0 1\n  endloop\nendfacet\n";
        os << "endsolid cube\n";
    }

    fr::geometry::CadSdfEngine engine;

    SUBCASE("ASCII STL Loading & Facet Verification") {
        bool loaded = engine.load_stl(test_stl);
        CHECK(loaded);
        CHECK(engine.get_triangle_count() == 12);
    }

    SUBCASE("SAH BVH Tree Construction & SDF Distance Queries") {
        bool loaded = engine.load_stl(test_stl);
        CHECK(loaded);

        engine.build_bvh(2); // Max 2 triangles per leaf

        // Test Point Outside (Fluid domain): (2, 0.5, 0.5) -> distance = 1.0
        double p_out[3] = {2.0, 0.5, 0.5};
        double n_out[3];
        double phi_out = engine.query_sdf(p_out, n_out);
        CHECK(phi_out == doctest::Approx(1.0).epsilon(1e-4));
        CHECK(n_out[0] == doctest::Approx(1.0).epsilon(1e-4));

        // Test Point Inside (Solid body): (0.5, 0.5, 0.5) -> negative signed distance
        double p_in[3] = {0.5, 0.5, 0.5};
        double phi_in = engine.query_sdf(p_in);
        CHECK(phi_in < 0.0);
    }

    SUBCASE("Geometric Transformation Engine (Scaling & Translation)") {
        bool loaded = engine.load_stl(test_stl);
        CHECK(loaded);

        // Scale x2, translate by (1, 1, 1) -> Cube [1, 3]^3
        engine.transform(2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        engine.build_bvh(4);

        // Center of transformed cube (2, 2, 2)
        double p_center[3] = {2.0, 2.0, 2.0};
        double phi_center = engine.query_sdf(p_center);
        CHECK(phi_center < 0.0);

        // Outside point (5, 2, 2) -> distance = 2.0
        double p_far[3] = {5.0, 2.0, 2.0};
        double phi_far = engine.query_sdf(p_far);
        CHECK(phi_far == doctest::Approx(2.0).epsilon(1e-4));
    }

    // Clean up temporary STL file
    std::remove(test_stl.c_str());
}

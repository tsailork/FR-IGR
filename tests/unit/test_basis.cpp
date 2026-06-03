#include "../doctest.h"
#include "../../src/core/basis.hpp"
#include <cmath>

TEST_CASE("Basis initialization and properties") {
    for (int p_deg = 0; p_deg <= 3; ++p_deg) {
        SUBCASE(("P_DEG = " + std::to_string(p_deg)).c_str()) {
            Basis basis(p_deg);
            int npts = p_deg + 1;
            
            CHECK(basis.z.size() == npts);
            CHECK(basis.w.size() == npts);
            CHECK(basis.l_L.size() == npts);
            CHECK(basis.l_R.size() == npts);
            CHECK(basis.D.size() == npts);
            if (npts > 0) CHECK(basis.D[0].size() == npts);
            
            // Check Gauss-Legendre nodes sum to zero (symmetry)
            double sum_z = 0.0;
            for (double z_val : basis.z) sum_z += z_val;
            CHECK(sum_z == doctest::Approx(0.0).epsilon(1e-12));

            // Check partition of unity for quadrature weights
            double sum_w = 0.0;
            for (double weight : basis.w) sum_w += weight;
            CHECK(sum_w == doctest::Approx(2.0));
            
            // Check Lagrange interpolation partition of unity at boundaries
            double sum_lL = 0.0, sum_lR = 0.0;
            for (int i = 0; i < npts; ++i) {
                sum_lL += basis.l_L[i];
                sum_lR += basis.l_R[i];
            }
            CHECK(sum_lL == doctest::Approx(1.0));
            CHECK(sum_lR == doctest::Approx(1.0));
            
            // Check exactness of differentiation matrix (D * 1 = 0)
            for (int i = 0; i < npts; ++i) {
                double sum_D = 0.0;
                for (int j = 0; j < npts; ++j) {
                    sum_D += basis.D[i][j];
                }
                CHECK(sum_D == doctest::Approx(0.0).epsilon(1e-12));
            }
            
            // Correction function endpoints/derivative signs
            if (p_deg > 0) {
                CHECK(basis.dgl.size() == npts);
                CHECK(basis.dgr.size() == npts);
                
                // The correction functions dgl and dgr should have non-zero elements
                bool dgl_nonzero = false;
                bool dgr_nonzero = false;
                for (int i = 0; i < npts; ++i) {
                    if (std::abs(basis.dgl[i]) > 1e-10) dgl_nonzero = true;
                    if (std::abs(basis.dgr[i]) > 1e-10) dgr_nonzero = true;
                }
                CHECK(dgl_nonzero);
                CHECK(dgr_nonzero);
            }
        }
    }
}

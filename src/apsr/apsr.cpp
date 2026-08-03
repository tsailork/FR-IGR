/**
 * @file apsr.cpp
 * @brief Implementation of Rectified Anisotropic Phantom Stress Relaxation (APSR-R) physics functions.
 */

#include "apsr.hpp"
#include "../ppr/ppr.hpp"
#include <iostream>

namespace APSR {

void compute_element_theta_apsr_2d(const std::vector<CellDim<2>*>& cells,
                                   const Basis& basis,
                                   const Parameters& p) {
    PPR::compute_element_theta_2d(cells, basis, p);
}

void compute_element_theta_apsr_3d(const std::vector<CellDim<3>*>& cells,
                                   const Basis& basis,
                                   const Parameters& p) {
    PPR::compute_element_theta_3d(cells, basis, p);
}

void compute_stress_tensor_2d(const std::vector<CellDim<2>*>& cells,
                              const Basis& basis,
                              const Parameters& p) {
    if (!p.ENABLE_APSR) return;
    (void)cells; (void)basis;
}

void compute_stress_tensor_3d(const std::vector<CellDim<3>*>& cells,
                              const Basis& basis,
                              const Parameters& p) {
    if (!p.ENABLE_APSR) return;
    (void)cells; (void)basis;
}

} // namespace APSR

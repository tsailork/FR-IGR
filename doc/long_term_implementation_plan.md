# Versatile CFD Solver Development Roadmap

This document outlines the strategic milestones and progress of evolving the high-order Cartesian 2D Flux Reconstruction (FR) solver.

> [!TIP]
> **Core Philosophy:** By sticking to a background Cartesian mesh, we preserve the extreme computational efficiency and simplicity of the tensor-product Flux Reconstruction (FR) sweeps. Complex geometries are handled via numerical treatments (Immersed Boundary Methods) rather than unstructured body-fitted meshes.

---
## 0 - Wishlist of smaller Features
*  Ability to restart with a different polynomial order by projecting the solution from the old pasis to the new one
*  Ability to restart with a different refinement pattern by projecting the solution from the old cells to the new cells



---

## 1. Navier-Stokes Extension (Viscous Fluxes) [COMPLETED]

Transitioning from the Euler equations to the full Navier-Stokes (NS) equations has been successfully implemented and validated.

*   **BR2 (Bassi-Rebay 2) Discretization:** A two-phase Bassi-Rebay 2 formulation is fully implemented to discretize the viscous fluxes. Common face fluxes use interface-averaged state evaluations combined with a penalty parameter for compactness and numerical stability.
*   **Sutherland's Law:** Temperature-dependent dynamic viscosity $\mu(T)$ and thermal conductivity $\kappa(T)$ are dynamically computed from the local non-dimensional temperature \(T = p/\rho\) at each solution node and element interface.
*   **Verification:** Validated via freestream preservation and Couette flow regression tests.

---

## 2. Enhanced I/O, Diagnostics, and Configuration [COMPLETED]

*   **Standardized INI Configs:** Fully implemented robust section-based configuration files (`inputs.dat`, `domain.grid`) parsing grid dimensions, boundary conditions, and solver controls.
*   **Comprehensive Point Probes:** Point probes have been fully implemented and verified for 8 key physical variables (`Density`, `XMomentum`, `YMomentum`, `Energy`, `Pressure`, `Temperature`, `Mach`, `Sigma`), exporting time-history logs.
*   **Restarts and Multiblock PVD:** Seamlessly handles multiblock datasets (`.vts`/`.vtm`) and persists timeline history in ParaView across simulation restarts.
*   **Aerodynamic Force Integration (Lift/Drag):** Fully implemented volume penalization force density integration over the solid geometry using high-order Gauss-Legendre quadrature, outputting coefficients $C_d$, $C_l$, and absolute forces history to `forces.csv`.
*   **Regularization Parameter Exposure:** Exposed all shock sensor parameters (`USE_DUCROS_SWITCH`, `USE_PRESSURE_SENSOR`, `USE_MOMENTUM_DIV`, `USE_PRESSURE_SOURCE_CAP`, `SOURCE_CAP_COEFF`) to `inputs.dat` configuration for runtime control.

---

## 3. Immersed Boundary Method (IBM) [COMPLETED / UNDER REVIEW]

*   **Volume Penalty Method (VPM):** Solid geometries are blanked out and penalized via an indicator fraction mask \(\chi\) and stiff relaxation terms.
*   **Shifted Boundary Method (SBM):** Curvilinear boundaries are mapped onto grid interfaces using 1D normal ray stencils and Lagrange reconstructions to preserve high-order boundary representation on Cartesian meshes.
*   **Acoustic & Viscous Stability Warning:** The standard Shifted Boundary Method (SBM) is not well-posed for high-order elements (\(P \ge 2\)) and high Reynolds number compressible flows. High-order boundary extrapolations can lead to severe numerical instabilities and negative pressure/density states. Either corrections must be implemented (such as localized stabilization, filtering, or robust ghost-cell formulations) or a different high-order IB method must be integrated.
*   **Reference Length Normalization:** Aerodynamic coefficients (\(C_d, C_l\)) currently default to `0.0` if `IB_CHORD` is set to zero or is invalid (e.g., \(\le 10^{-12}\)), preventing division-by-zero. While <code>IB_CHORD</code> is used generically as the reference length, future updates should introduce a distinct <code>IB_REF_LENGTH</code> input variable to explicitly distinguish between chord length (airfoils) and diameter (cylinders/spheres) in multidimensional simulations.
*   **Verification:** Verified via regression tests checking cylinder flow configurations under both VPM and SBM.

---

## 4. More Efficient Runge-Kutta Timestepping [COMPLETED]

High-order Flux Reconstruction on fine meshes imposes strict stability limits on the explicit time-step size. To improve temporal efficiency, the solver now implements time-accurate local sub-cycling:

*   **Time-Accurate Multirate Sub-cycling:** Implemented power-of-two element sub-cycling to advance coarse elements at larger, local stable time-steps, while inactive elements accumulate intermediate RK stage derivatives in a memory-efficient accumulator state, synchronizing conservatively at global steps.
*   **Higher-Stage SSP-RK Schemes [FUTURE]:** Implement multi-stage Strong Stability Preserving Runge-Kutta schemes (such as SSP-RK(10,4) or SSP-RK(14,5)) which offer significantly larger stable CFL limits per stage, reducing the overall computational cost for long-time integrations.
*   **Stabilized Runge-Kutta-Chebyshev (RKC) Methods [FUTURE]:** To alleviate the severe parabolic timestep constraint (\(\Delta t \propto h^2\)) encountered in viscous Navier-Stokes runs, implement RKC schemes. These explicit methods have a stable step size that grows quadratically with the number of stages, making them highly efficient for diffusion-dominated problems without requiring implicit matrix solves.

---

## 5. Multi-Block Cartesian Grids [COMPLETED]

*   **Connectivity and Neighbors:** Refactored the internal structures to support arbitrary block configurations. Neighbor communication, metric verification, and 1-to-1 boundary alignment validation are executed at startup.

---

## 6. Extension to 3D Problems [FUTURE]

Extending the solver to 3D Cartesian elements will allow for realistic aerodynamic simulations (e.g., flow over wings, 3D spheres).

*   **3D Sweep Operators:** Extend the 1D dimension-split sweep approach to 3D (X-sweeps, Y-sweeps, and Z-sweeps). This maintains the simplicity of the 1D tensor-product Flux Reconstruction correction functions.
*   **Bassi-Rebay 2 (BR2) in 3D:** Generalize the viscous gradient and flux divergence operators to compute 3D stress tensors and heat fluxes.
*   **3D Immersed Boundary:** Adapt the SDF/Level-Set geometry check and mask generation to 3D, supporting STL geometries or 3D analytical shapes (spheres, ellipsoids, wings).

---

## 7. Octree Adaptive Mesh Refinement (AMR) [FUTURE]

*   **Linear Tree Infrastructure:** Transition from a flat list of blocks to a linear tree data structure (sorted leaf arrays of Morton IDs) to manage the element hierarchy. Mesh connectivity queries and neighbor matching are performed via binary searches on the sorted Morton IDs (as described in Wang, Witherden, Jameson JCP 2024), avoiding high pointer-chasing overheads.
*   **OpenMP Flattening & Load Balancing:** Maintain flat lists of pointers to active leaf elements (`std::vector<Block*>`) to allow simple and efficient `#pragma omp parallel for` load distribution across threads during solver sweeps.
*   **Generalized Geometry Refinement Zones:** Extract and generalize the immersed boundary geometry engine (analytical shapes, NACA profiles, piecewise polygons) to define mesh refinement zones. Any cell that is even partially inside a defined zone (inclusive) is automatically refined to the specified target level.
*   **2:1 Refinement Limit & Cascading Splits:** Enforce a strict maximum 2:1 refinement ratio between adjacent elements. Trigger cascading splits on coarser neighbors when an element splits to preserve stencil integrity.
*   **Automatic Near-Boundary Refinement:** Automatically detect and flag elements within a specified physical distance of solid walls or immersed boundaries for local grid refinement.
*   **Multirate / Paired Runge-Kutta Methods [COMPLETED]:** Implemented time-accurate multirate sub-cycling using power-of-two temporal levels. Coarse elements update less frequently than fine elements, utilizing an SSP-RK3 integration weight accumulator and conservative synchronization at the end of the global step. This is ready to be coupled with Quadtree/Octree adaptive refinements.
*   **Regularization & ADI Deprecation Note:** The elliptic IGR formulation (which requires tridiagonal Thomas sweeps solved via ADI) is deprecated due to its incompatibility with unstructured/tree mesh topologies. Future AMR development will exclusively rely on the element-by-element parabolic IGR (BR2 based) formulation, eliminating the need to solve global implicit tridiagonal systems across non-conforming interfaces.

---

## 8. Subcell Stabilization & Invariant Domain Preserving (IDP) Methods [FUTURE]

To handle strong shocks and high-order instabilities more robustly than algebraic limiting (e.g., Zhang-Shu), explore subcell stabilization and convex blending schemes:

*   **Subcell Finite Volume Blending:** In elements flagged by shock/discontinuity sensors, blend the high-order Flux Reconstruction solver with a robust, dissipative low-order Finite Volume solver on a subcell grid (e.g., dividing the FR element into subcells matching the solution node count). This maintains high-order accuracy in smooth regions and drops down to robust Finite Volume upwinding inside shock cores.
*   **Dissipative 2-Point Fluxes:** Explore entropy-stable two-point fluxes (e.g., Tadmor-style entropy conservative fluxes with subcell numerical diffusion) to ensure thermodynamic entropy growth.
*   **Invariant Domain Preserving (IDP) Limiters:** Implement algebraic flux correction (AFC) or convex blending that mathematically guarantees invariant domains (such as maximum/minimum bounds on density, specific entropy, and velocity) while maintaining strict local conservation.

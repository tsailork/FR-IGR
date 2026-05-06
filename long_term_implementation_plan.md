# Versatile CFD Solver Development Roadmap

This document outlines the strategic milestones required to evolve the current Cartesian 2D Euler FR-IGR solver into a highly versatile, general-purpose Computational Fluid Dynamics (CFD) framework.

> [!TIP]
> **Core Philosophy:** By sticking to a background Cartesian mesh, we preserve the extreme computational efficiency and simplicity of the tensor-product Flux Reconstruction (FR) sweeps. Complex geometries will be handled via numerical treatments (Immersed Boundary) rather than unstructured body-fitted meshes.

---

## 1. Navier-Stokes Extension (Viscous Fluxes)

Transitioning from the Euler equations to the full Navier-Stokes (NS) equations is the most fundamental physics upgrade. It allows the solver to capture boundary layers, turbulence, and viscous dissipation.

*   **Implement the BR2 (Bassi-Rebay 2) Scheme:** We already use a BR2-style operator for the Parabolic IGR. We must formalize this into a dedicated viscous flux module.
    *   *Step A:* Add a "Gradient Pass" to compute the auxiliary variable $Q = \nabla U$. This requires an initial Riemann solve (typically using central fluxes) to find the common state at interfaces.
    *   *Step B:* Compute the physical viscous fluxes $F_v(U, Q)$ at the solution points.
    *   *Step C:* Perform a second Riemann solve for the viscous fluxes, incorporating the BR2 lifting penalty parameter ($\eta$) to guarantee stability and compactness.
*   **Update Boundary Conditions:** Viscous simulations require new BCs, specifically no-slip isothermal and no-slip adiabatic walls, which dictate how the velocity and temperature gradients are mirrored in the ghost states.
*   **Sutherland's Law:** Implement dynamic viscosity $\mu(T)$ and thermal conductivity $k(T)$ calculations based on local temperature to accurately model real gases.

## 2. Enhanced I/O, Diagnostics, and Configuration

For a solver to be "general use," it must be exceptionally easy for a researcher to set up a case, monitor its progress, and extract quantitative data without touching the C++ source code.

*   **Robust Configuration Parser:** Migrate from the custom key-value parser to a standardized format like JSON, YAML, or TOML (via a lightweight header-only library like `nlohmann/json`). This allows for nested parameters, arrays (e.g., specifying multiple boundary regions), and strict type-checking.
*   **Runtime Probe Data:** Implement a "probe" system where the user can specify $(X, Y)$ coordinates in the input file. The solver interpolates the high-order polynomial data to these specific points and streams the time-history (e.g., pressure traces) to lightweight CSV files during the run, independent of the heavy VTK volume snapshots.
*   **Global Diagnostics Tracking:** Add a mechanism to compute and log integral quantities at every timestep (or output interval).
    *   *Step A:* Track total mass, total energy, and maximum Mach number to monitor conservation and stability.
    *   *Step B:* Track aerodynamic forces (Lift/Drag) by integrating pressure and shear stress along designated boundaries.

## 3. Immersed Boundary Method (IBM)

Since we are committing to a Cartesian background mesh, simulating flow around complex geometries (airfoils, cylinders, vehicles) requires an Immersed Boundary Method. 

*   **Geometry Representation (Level Set):** Introduce a Signed Distance Field (SDF) or Level Set function, $\phi(x,y)$, where $\phi < 0$ is inside the body and $\phi > 0$ is the fluid. This elegantly defines the boundary location and the surface normal vectors. **(Note: We will investigate and leverage an external library to assist with SDF generation/management to save development time).**
*   **Phase 1: Penalization / Forcing Method (Prototyping):** Add a stiff source term to the RHS equations inside and immediately adjacent to the solid body to force the velocity to zero (no-slip).
    *   *Pros:* Very easy to implement within the existing RK3 loop; requires no changes to the FR sweeps. Excellent starting point to get a working prototype.
    *   *Cons:* Can reduce the time-step restriction heavily; generally drops accuracy to 1st or 2nd order near the boundary.
*   **Phase 2: High-Order Interface Modification Method (Advanced):**
    *   Instead of complex ghost-cell bookkeeping or specialized quadrature, we will aim for an approach that modifies the interface/boundary state of elements to enforce boundary conditions, leaving the interior cells alone.
    *   *Reference:* "A high-order immersed boundary method to approximate flow problems in domains with curved boundaries" by S. Colombo et al.
    *   *Pros:* Recovers high-order accuracy; sharp interface representation without the fragility of cut-cell volume quadratures.
    *   *Cons:* A much more complex operation requiring careful consideration and significant effort to implement correctly.

## 4. Multi-Block Cartesian Grids

Uniform single-grid Cartesian domains are inefficient. The first major structural step to alleviate this is transitioning to a Multi-Block architecture. This provides a massive "bang for your buck" and refines the data structures before attempting more complex tree-based meshes.

*   **Block-Structured Refinement:** Group elements into "Blocks" (e.g., $10 \times 10$ elements per block).
    *   *Step A:* Refactor the `State` data structure so that instead of one giant grid, it manages a list of `Block` objects.
    *   *Step B:* Implement block-to-block communication interfaces (MPI-ready or shared memory).
    *   *Step C:* Allow blocks to have different uniform resolutions (e.g., a fine block nested inside a coarse block). This requires conservative interpolation (prolongation/restriction) at the boundaries between blocks of different sizes.

## 5. Octree Adaptive Mesh Refinement (AMR)

Once the Multi-Block foundation is rock solid, the final frontier for a Cartesian solver is full tree-based AMR.

*   **Linear Tree Infrastructure:** Transition from a flat list of blocks to a linear tree data structure (Quadtree in 2D, Octree in 3D).
    *   *Reference:* "An efficient GPU-based h-adaptation framework via linear trees for the flux reconstruction method" by L. Wang, F. Witherden, and A. Jameson.
    *   *Goal:* Utilize linear trees to manage the hierarchy and connectivity of the mesh elements efficiently, paving the way for future GPU acceleration.
*   **Dynamic Load Balancing:** Distribute the leaves of the linear tree across OpenMP threads (or eventually MPI ranks) dynamically to prevent idle CPU cores.
*   **User-Defined AMR Regions:** Instead of dynamic sensor-driven $h$-refinement, refinement regions will be explicitly defined by the user in the input file. This ensures predictable computational loads and avoids the overhead of dynamic re-meshing. We will look into dynamic $P$-refinement (polynomial adaptation) in the future.

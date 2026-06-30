# GEMINI.md - Project Context: FR-IGR

## Project Overview
This project is a high-order numerical solver for the **2D Euler equations** (fluid dynamics) using the **Flux Reconstruction (FR)** method (also known as the Correction Procedure via Reconstruction - CPR). It is implemented in C++ and targets Cartesian grids.

The solver includes a specialized regularization technique called **Isotropic Gradient Regularization (IGR)**, also referred to as "Entropic Pressure," to handle shocks and discontinuities by adding artificial viscosity through a Helmholtz-like smoothing equation solved via **Alternating Direction Implicit (ADI)**.

### Key Technologies
- **Language:** C++17
- **Compiler:** `g++`
- **Numerical Method:** Flux Reconstruction (Tensor Product)
- **Time Stepping:** Strong Stability Preserving Runge-Kutta 3rd Order (SSP-RK3)
- **Riemann Solver:** Rusanov / Local Lax-Friedrichs (LLF)
- **Visualization:** ParaView (using `.pvd`/`.vtm` output)
- **Dependencies:** Listed in `dependencies.txt`

## Directory Structure & Key Files
The codebase is structured into modular compilation units inside the `src/` directory:
- `src/core/`: Fundamental building blocks (`parameters.hpp/cpp`, `state.hpp`, `basis.hpp/cpp`, `solver.hpp/cpp`).
- `src/flux/`: Inviscid operators (`euler_flux.cpp`, `sweep_x.cpp`, `sweep_y.cpp`).
- `src/igr/`: Artificial viscosity mechanisms (`sensor.cpp`, `adi_solver.cpp`, `parabolic.cpp`, `entropic_pressure.cpp`).
- `src/boundary/`: Centralised boundary condition evaluation (`boundary.cpp`).
- `src/limiters/`: Stabilisation routines (`positivity.cpp`, `entropy.cpp`, `limiter_common.hpp`).
- `src/time/`: Explicit time marching (`rk3.cpp`, dynamic `stability.cpp`).
- `src/io/`: Input, output, and initial states (`initial_conditions.cpp`, `restart.cpp`, `vtk_writer.cpp`).
- `src/main.cpp`: Entry point. Parses inputs and executes the time-stepping loop.
- `tests/`: Unit test suite (`test_main.cpp`).
- `inputs.dat`: Current active simulation configuration.
- `inputs_example.txt`: Comprehensive documentation and template for all available input options.
- `run.sh`: Convenience script for building, running, and plotting.

## Building and Running
The project uses a simple shell script for the entire pipeline.

### Commands
- **Build and Run:**
  ```bash
  ./run.sh
  ```
  This script dynamically reads `NUM_THREADS` from `inputs.dat`, exports `OMP_NUM_THREADS`, and executes:
  1. `make -j4 all`
  2. `./fr_solver`
  3. Optionally runs Python visualization if available, or directs the user to open VTK files in ParaView.

- **Dependencies:**
  - C++17 compatible compiler (`g++`).
  - OpenMP for multi-core parallelism.
  - ParaView for `.pvd`/`.vts` visualization.

## Development Conventions
- **Polynomial Degree:** The degree of the polynomial inside each element is controlled by `P_DEG` in `parameters.hpp`. Common values are `0` (Finite Volume-like) or `2`/`3` (High-Order).
- **Coordinate Mapping:** The solver uses a reference element $[-1, 1]$ mapped to local physical coordinates.
- **Data Layout:** The `State` class uses a flattened `std::vector` with an indexer `operator(v, ey, ex, iy, ix)` for efficient access.
- **Regularization:** The IGR (Entropic Pressure) can be toggled via `ENABLE_IGR` in `parameters.hpp`.
- **Boundary Conditions:** Transmissive (copy-out) boundary conditions are implemented at the domain edges.

## Architecture Notes
- **Tensor Product:** 2D operations are split into 1D X-sweeps and Y-sweeps, utilizing the efficiency of tensor-product basis functions on Cartesian elements.
- **Correction Functions:** The solver uses Radau-based correction functions (FR-DG variant) for computing flux divergence at element interfaces.
- **ADI Solver:** The `compute_entropic_pressure` method implements an ADI solver to handle the implicit regularization step, which involves solving tridiagonal systems using the Thomas algorithm.

---

## Technical Refinements & Enhancements (March 2026)

### 1. Symmetrized ADI Solver
The standard ADI splitting method introduces $O(DT^2)$ directional bias. In symmetric problems (e.g., 2D Riemann Configuration 3), this drives numerical asymmetry. The solver now implements **Symmetrized ADI**:
- Executes both an $XY$ splitting pass and a $YX$ splitting pass.
- Averages the resulting entropic pressure fields: $\Sigma = 0.5 (\Sigma_{XY} + \Sigma_{YX})$.
- This restores diagonal symmetry and improves stability in high-resolution runs.

### 2. Non-Uniform IGR Stencil
Gauss-Legendre quadrature points ($P=1$) are non-uniformly spaced within elements. The IGR tridiagonal solver has been refactored to use a general **3-point non-uniform stencil**:
- Accounts for physical distances ($h_L, h_R$) between solution nodes and across element boundaries.
- Uses interface-averaged density weighting for the diffusion coefficients.
- Implements corrected zero-gradient Neumann boundary conditions at the domain edges.

### 3. Dynamic CFL Stability
The simulation now utilizes a dynamic time-step calculation based on the local wave speeds ($\lambda_{max} = |u| + c$):
- $DT = 0.5 \cdot \text{CFL} \cdot \frac{h}{(2P+1) \cdot \lambda_{max}}$
- Includes a 2D stability safety factor (0.5) for Cartesian grids.
- Prevents $T_{FINAL}$ overshoot.

### 4. Sigmoid Initial Smoothing
Discontinuous initial conditions trigger immediate instabilities in high-order FR methods. The `main.cpp` now applies **Sigmoid Smoothing** to the initial quadrants:
- Smoothing width `delta` is typically $2.0 \cdot \min(dx, dy)$.
- Mitigates start-up oscillations while preserving the sharp physical profile.

### 5. Runtime Symmetry Check
The main simulation loop includes a quantitative symmetry check that mirrors the density field across the diagonal and reports any deviation exceeding $1e^{-10}$, facilitating early detection of numerical bias.

## Technical Refinements & Enhancements (May 2026)

### 1. Robust Multiblock Restart & PVD Continuity
The solver now supports restarting from complex multiblock states:
- **XML-Based Parsing:** Automatically parses `.vtm` (MultiBlockDataSet) files to map block IDs to their respective `.vts` data files.
- **PVD History Persistence:** Upon restart, the `VTKWriter` scans the existing `solution.pvd` file to load previous time-step history, ensuring a continuous timeline in ParaView without data-loss on resume.

### 2. Automated Grid & Connectivity Validation
To prevent simulation crashes due to misconfigured domains, a strict validation pass is performed at startup:
- **1-to-1 Mapping:** Ensures that every block neighbor relationship is symmetric (e.g., if Block A says B is its left neighbor, B must say A is its right neighbor).
- **Metric Verification:** Validates that physical interface lengths and element counts match across neighboring block boundaries.

### 3. Synchronized Diagnostics & Appending
Diagnostics are now state-aware across restarts:
- **Interval Sync:** The internal timers for terminal printing, residuals, and probes are synchronized with the `RESTART_TIME`, preventing immediate output floods on resume.
- **File Appending:** Diagnostic logs (`csv_outputs/residuals.csv`, `csv_outputs/probe.csv`) switch to append mode during restarts, preserving the full time-history of the simulation.

### 4. Pressure-Bounded IGR Source Term
To improve the stability of high-order IGR at strong shocks:
- **Source Capping:** The raw sensor source term ($S_{buf}$) is now optionally capped by the local pressure ($S \le C \cdot P$) *before* the Helmholtz smoothing pass.
- **Preserved Diffusion:** Unlike capping the final viscosity field, capping the *source* allows the Helmholtz operator to naturally diffuse viscosity ahead of the shock wave, providing the necessary pre-conditioning for numerical stability.

## Technical Refinements & Enhancements (June 2026)

### 1. Headless Execution Redirection & STOP Trigger
- **Buffer-free File Redirection:** Re-routes the solver's stdout and stderr streams to `out.log` inside `main.cpp`, employing non-buffered operations (`_IONBF`) to guarantee immediate diagnostic flushes.
- **Headless STOP Interrupt:** Actively scans the simulation directory for a `STOP` trigger file during each time step, removing the file and executing a graceful solver termination upon detection.

### 2. Live Runtime Dashboard (TUI)
- **Multi-Zone Monitor:** Implements a real-time command terminal user interface (`tui.py`) parsing solver execution states, warning levels, and SBM boundary diagnostics.
- **Subprocess Group Control:** Integrates non-blocking keyboard controls allowing on-demand stops, kills, or restarts (`run.sh` re-runs) from within the dashboard.

### 3. Automated Checkpoint Detection & Restart Sync
- **PVD Scan Integration:** The TUI parses the `pv_outputs/solution.pvd` file using regex, dynamically identifies the latest physical dataset file (.vtm) and its simulation timestamp, and automatically writes these to the `RESTART_FILE` and `RESTART_TIME` parameters in `inputs.dat` when a restart (`R`) is triggered.
- **Clean State Restarting:** When a clean restart (`C`) is triggered, the TUI automatically clears these restart parameters, allowing the solver to build cleanly and start a fresh simulation run from $t=0.0$.

### 4. In-Place Input Configuration Editor
- **Interactive Terminal Takeover:** Implements the `edit_inputs_dat` function to suspend the raw input terminal mode, clear the screen, launch the system's `$EDITOR` (e.g. `vim` or `nano`) in-place, and wait for the user to edit `inputs.dat`.
- **Dynamic Parameter Reloading:** Upon editor termination, the TUI re-enters non-blocking cbreak mode, re-parses the new configuration parameters from `inputs.dat` to update live variables, and triggers a full dashboard redraw.

### 5. Portable Case Management & Interactive Web GUI
- **Portable Case Management:** Created portable running directory structures (e.g. `cases/default_case/`) containing local inputs/outputs (`inputs.dat`, `domain.grid`, `pv_outputs/`, `csv_outputs/`, `out.log`) using relative paths.
- **Project Executable Resolution:** Implemented absolute-path resolving runner scripts (`run_case.sh` / `run_case.bat`) inside case directories to build and run the solver centrally.
- **Lightweight Python Web Server:** Added `gui.py` backend server executing the compiled C++ solver under WSL/Linux, parsing VTK `.vts`, `.vtu`, and `.vtm` datasets to serve grid coordinates and scalar fields as JSON.
- **Interactive Web Interface:** Designed `gui/` frontend with an HTML5 `<canvas>` rendering block element lines, color-coded boundaries, and solid objects (cylinder, NACA airfoil profiles with angle of attack).
- **GUI Control & Plotting:** Implemented two-way configuration bindings, visual boundary condition editor, point probe click-to-place tool, and live console logging alongside Canvas-drawn residual/probe charts and 2D flow field contours.

### 6. Unstructured Grid (.vtu) Plotting & Tricontourf
- **Immersed Boundary Support:** The solver blanks out points and cells inside solid boundaries when utilizing immersed boundary methods (SBM/VPM), resulting in unstructured grids. The visualizer parses `.vtu` (UnstructuredGrid) files from `plot.pvd` rather than structured `.vts` files.
- **Tricontourf Integration:** Integrated Matplotlib's `ax.tricontourf` to correctly render 2D flow field contours on unstructured meshes, preserving accurate boundary shapes without grid-reshaping failures.
- **PVD Priority Alignments:** Configured both GUI and TUI systems to check `plot.pvd` before `solution.pvd`, ensuring that visualizer playback monitors the active plot output.

### 7. Double-Ended Visualizer Caching
- **Server-Side Cache (`CONTOUR_CACHE`):** Implemented an in-memory Python dictionary cache in `gui.py` keyed by `(var_name, vtm_name, mtime)`. If a VTM file is unchanged, the server returns the cached PNG bytes in under 0.2ms, representing a 5,900x rendering speedup.
- **Browser HTTP Caching:** Modified `/api/history` to expose modification times (`mtime`) for each timestep. The contour generator sets long-term cache-control headers (`Cache-Control: public, max-age=31536000`) when `mtime` is present, enabling instant browser-side loading.

### 8. Sequential Client-Side Preloading
- **Sequential Queue:** Added a non-blocking sequential background loading queue in Javascript (`gui/app.js`) that downloads contour images for all timesteps of the active variable when history loads or when variables are switched.
- **Buttery Smooth Playback:** By loading images ahead of time into the browser cache, drag and playback operations on the timeline slider occur instantly and seamlessly without network latency.
- **Smart Timeline Polling:** Integrates with the status polling loop to dynamically append new checkpoints as the solver runs, while preserving the user's active manual playback selection.

### 9. Conditionally Exposed GUI Controls & Refined Restart
- **Method-Specific Panels:** conditionally shows/hides SBM-specific (Shifted Boundary Method) diagnostics and parameters or VPM-specific (Volume Penalization Method) penalization parameters.
- **Dynamic Polygon Tables:** Exposes coordinate tables for dynamic piecewise multi-polygons when `IB_SHAPE = MULTI`.
- **Refined Clean Restart:** Modified clean restarts to only wipe transient case outputs (`pv_outputs/*`, `csv_outputs/*`, `out.log`, `STOP`, `residuals.dat`) instead of running `make clean`, accelerating restart compilation and startup.


## Technical Refinements & Enhancements (July 2026)

### 1. Decoupled Geometry Engine
- Decoupled `Polygon`, `Circle`, and `Naca` shapes into a modular geometry engine (`src/core/geometry.hpp/cpp`).
- Implemented ray-casting point containment, Liang-Barsky line clipping segment intersection, and general AABB-polygon intersection algorithms.

### 2. Breadth-First Near-Wall Layer Tracking
- Implemented a Breadth-First Search (BFS) starting from slip/no-slip/moving/isothermal wall boundary cells to track near-wall distance in terms of grid cell layers rather than expensive Euclidean distance, controlling wall refinement via `WALL_REFINEMENT_LEVEL` and `WALL_REFINEMENT_CELLS`.

### 3. State-Conservative Tree Decomposition
- Implemented cell splitting (`split_cell`) and merging (`merge_cells`) tree operations.
- Prolongs variables and regularization fields to children using 2D tensor-product Lagrange interpolation (`P1`/`P2`), and restricts children back to parent via strictly conservative $L_2$ projections (`R1`/`R2`).
- Resolves leaf cells containing physical points dynamically using `find_leaf_cell` rather than conforming 2D lookups.

### 4. Quadtree Adaptive Mesh Refinement (AMR) & Tree Decomposition
The solver incorporates a fully conservative, dynamically-adaptable 2D quadtree cell decomposition framework:
- **Geometry Engine**: Features a decoupled geometry engine (`src/core/geometry.hpp/cpp`) that evaluates shapes (`Circle`, `Naca`, `Polygon`, `Box`, `Multi`) for spatial queries. It supports ray-casting point containment, Liang-Barsky segment clipping, and axis-aligned bounding box (AABB) intersection.
- **Near-Wall Refinement**: Performs breadth-first search (BFS) starting from wall boundaries (slip, no-slip, moving, isothermal) to compute distance in terms of grid cell layers. Controls boundary-local mesh refinement level and cell-layer width via `WALL_REFINEMENT_LEVEL` and `WALL_REFINEMENT_CELLS`.
- **Boundary-Specific Toggles**: Supports per-boundary toggle flags appended as suffixes to boundary condition strings in `domain.grid` (e.g. `:NOREFINED` or `:REFINED`) to include or exclude specific boundary faces from near-wall refinement.
- **Manual Refinement Zones**: Supports an arbitrary number of user-defined refinement regions (`NUM_REFINEMENT_ZONES`), allowing targeted refinement to specific level using shapes (CIRCLE, BOX, NACA, or POLYGON).
- **Conservative Interpolations**: Prolongs solutions and regularization fields to child cells via Lagrange tensor-product interpolation ($P_1/P_2$), and restricts children's fields back to parent cells using strictly conservative $L_2$ projections ($R_1/R_2$).
- **State Evaluation & Immersed Boundary Mapping**: Evaluates initial conditions and maps immersed boundary solid mask states directly on the leaf cells, ensuring robust, out-of-bounds-safe solver operations on arbitrarily refined grids.

## Documentation Maintenance (Agent Hook)
Whenever tasked with "updating the documentation" for a new feature or change, you **MUST** ensure all the following locations are kept perfectly synchronized with the codebase:

1. **In-Code Doxygen (`src/`)**: 
   - Add or update Javadoc-style `/** ... */` block comments for any new or modified classes, structs, and function signatures.
   - Use rigorous tags (`@brief`, `@param`, `@return`, `@see`) to maintain interconnectivity.
2. **User-Facing HTML Guide (`doc/index.html`)**:
   - Update the respective tab (Overview, Architecture, IB Methods, Configuration, Testing) when introducing major components or structural shifts.
   - Maintain the professional, highly-technical tone and use MathJax for new equations.
3. **Execution Flow Diagrams (`doc/scripts/`)**:
   - If the main time-stepping loop, pre-iterations, or initialization steps change, update `doc/scripts/generate_flow_diagram.py`.
   - Re-run the python script to regenerate the `doc/assets/solver_flow_diagram.svg` artifact.
4. **Input Parameter Definitions (`inputs_example.txt`)**:
   - Whenever a new configuration flag or numerical parameter is added to `src/core/parameters.hpp`, it **must** be documented in `inputs_example.txt` with a detailed explanation and sample value.
5. **Project Context (`GEMINI.md`)**:
   - Append major architectural paradigms, algorithmic improvements, or testing infrastructure changes to the "Technical Refinements" section of this file to ensure future agents understand the context.

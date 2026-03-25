# GEMINI.md - Project Context: FRAI3

## Project Overview
This project is a high-order numerical solver for the **2D Euler equations** (fluid dynamics) using the **Flux Reconstruction (FR)** method (also known as the Correction Procedure via Reconstruction - CPR). It is implemented in C++ and targets Cartesian grids.

The solver includes a specialized regularization technique called **Isotropic Gradient Regularization (IGR)**, also referred to as "Entropic Pressure," to handle shocks and discontinuities by adding artificial viscosity through a Helmholtz-like smoothing equation solved via **Alternating Direction Implicit (ADI)**.

### Key Technologies
- **Language:** C++17
- **Compiler:** `g++`
- **Numerical Method:** Flux Reconstruction (Tensor Product)
- **Time Stepping:** Strong Stability Preserving Runge-Kutta 3rd Order (SSP-RK3)
- **Riemann Solver:** Rusanov / Local Lax-Friedrichs (LLF)
- **Visualization:** Python (using `plot2d.py`)

## Directory Structure & Key Files
- `main.cpp`: Entry point. Sets up initial conditions (e.g., 2D Riemann Problem Configuration 3) and executes the time-stepping loop.
- `solver.hpp`: Contains the `Solver` class. Implements RHS assembly (X/Y sweeps), Riemann solvers, and the IGR/Entropic Pressure mechanism.
- `parameters.hpp`: Global simulation parameters including grid resolution (`N_ELEM_X`, `N_ELEM_Y`), polynomial degree (`P_DEG`), and physics constants (`GAMMA`).
- `state.hpp`: Defines the `State` data structure for conservative variables ($\rho, \rho u, \rho v, E$).
- `basis.hpp`: Handles basis functions (Lagrange polynomials), derivative matrices, and FR correction polynomial derivatives.
- `run.sh`: Convenience script for building, running, and plotting.
- `plot2d.py`: Python script to visualize the resulting `solution_2d.csv`.

## Building and Running
The project uses a simple shell script for the entire pipeline.

### Commands
- **Build and Run:**
  ```bash
  ./run.sh
  ```
  This script executes:
  1. `g++ -std=c++17 -g main.cpp -o fr_solver`
  2. `./fr_solver`
  3. `python plot2d.py`

- **Dependencies:**
  - C++17 compatible compiler (`g++`).
  - Python 3 with `pandas` and `matplotlib` (for plotting).

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

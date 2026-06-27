# FR-IGR: High-Order Euler Solver with Isotropic Gradient Regularization

FR-IGR is a high-order numerical solver for the 2D Euler equations using the **Flux Reconstruction (FR)** method on Cartesian multiblock grids. It features **Isotropic Gradient Regularization (IGR)** for robust shock capturing and is parallelized using OpenMP.

This project has also been an exploration of AI-LLM assisted code generation. LLM-based coding tools have been used extinsively in the creation of this codebase.

## Features
- **High-Order Accuracy**: Supports arbitrary polynomial degrees (P=1, P=2, P=3, etc.) using Radau-based correction functions.
- **Multiblock Support**: Supports complex domain topologies with automated 1-to-1 face connectivity validation.
- **IGR Regularization**: Advanced artificial viscosity using Helmholtz-smoothed gradient sensors.
- **Robustness**: Includes Zhang-Shu positivity-preserving limiters and entropy-based stabilization.
- **Continuity**: Robust restart system with full VTK timeline (PVD) and diagnostic history preservation.

## Quick Start
1. **Requirements**: `g++` (C++17), `make`, and `OpenMP`. See [dependencies.txt](dependencies.txt) for more details.
2. **Build and Run**: 
   ```bash
   ./run.sh
   ```
3. **Visualization**: Open `pv_outputs/solution.pvd` in **ParaView**.

## Documentation
- [GEMINI.md](GEMINI.md): Detailed technical history, architecture notes, and feature descriptions.
- [dependencies.txt](dependencies.txt): Software and library requirements.
- [inputs_example.txt](inputs_example.txt): Comprehensive guide to all simulation parameters.
- (.html) [High level overview](doc/index.html) of the methods used in this code.
- (.html) [Doxygen](doc/doxygen/html/index.html) documentation pages. 

---

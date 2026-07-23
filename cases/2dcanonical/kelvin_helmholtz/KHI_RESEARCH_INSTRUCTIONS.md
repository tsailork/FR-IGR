# MISSION: Kelvin-Helmholtz Instability (KHI) Resolution & Viscosity Study

You are tasked with executing a systematic computational investigation of the **2D Kelvin-Helmholtz Instability (KHI)** using the high-order C++ Flux Reconstruction solver (`FR-IGR`).

---

## 1. Study Parameters & Constraints

- **Test Case**: Kelvin-Helmholtz Instability (`cases/2dcanonical/kelvin_helmholtz/`)
- **Methods to Compare (ONLY 2 METHODS)**:
  1. **Navier-Stokes (`navier_stokes`)**: Pure physical viscosity without artificial viscosity (`ENABLE_NS=true`, `ENABLE_PPR=false`, `ENABLE_IGR=false`).
  2. **Adaptive PPR (`ppr_adaptive`)**: Navier-Stokes with Phantom Pressure Relaxation artificial viscosity (`ENABLE_NS=true`, `ENABLE_PPR=true`, `ENABLE_IGR=false`).
  - *STRICT RULE*: Do **NOT** run pure Euler (inviscid) or IGR (parabolic entropic pressure) cases.
- **Parameter Matrix (16 Cases Total)**:
  - **Polynomial Degrees**: $P=2$ (3rd-order) and $P=3$ (4th-order).
  - **Grid Resolutions**: $1\times$ ($64 \times 64$ elements) and $2\times$ ($128 \times 128$ elements).
  - **Reynolds Numbers**: Normal $Re = 10000.0$ and Double $Re = 20000.0$.
  - **Physical Duration**: $T_{\text{FINAL}} = 5.0$, `OUTPUT_INTERVAL = 0.10` (50 VTK snapshots per run).

---

## 2. Automated Execution Procedure

Execute the dedicated automated investigation script located inside the case directory:

```bash
python3 cases/2dcanonical/kelvin_helmholtz/run_khi_investigation.py
```

This script will automatically:
1. Recompile the C++ solver (`bin/fr_solver`).
2. Construct all 16 study variant directories with exact parameters:
   - `run_navier_stokes_64x64_P2_Re10000`
   - `run_navier_stokes_64x64_P2_Re20000`
   - `run_navier_stokes_64x64_P3_Re10000`
   - `run_navier_stokes_64x64_P3_Re20000`
   - `run_navier_stokes_128x128_P2_Re10000`
   - `run_navier_stokes_128x128_P2_Re20000`
   - `run_navier_stokes_128x128_P3_Re10000`
   - `run_navier_stokes_128x128_P3_Re20000`
   - `run_ppr_adaptive_64x64_P2_Re10000`
   - `run_ppr_adaptive_64x64_P2_Re20000`
   - `run_ppr_adaptive_64x64_P3_Re10000`
   - `run_ppr_adaptive_64x64_P3_Re20000`
   - `run_ppr_adaptive_128x128_P2_Re10000`
   - `run_ppr_adaptive_128x128_P2_Re20000`
   - `run_ppr_adaptive_128x128_P3_Re10000`
   - `run_ppr_adaptive_128x128_P3_Re20000`
3. Execute C++ time integration for each run using OpenMP multi-threading.
4. Execute Python analytics (`spectrum_2d.py`, `enstrophy.py`, `ppr_localization.py`).
5. Generate publication-quality multi-scheme comparative overlay figures inside `summary_plots/`.

---

## 3. Post-Processing & Deliverables to Report

Upon completion of execution and analytics, analyze the generated CSV data (`csv_outputs/`) and plot figures (`summary_plots/`) to produce a comprehensive report covering:

1. **Vortex Roll-up & Secondary Pairing**:
   - Compare $P=2$ vs $P=3$ and resolution ($64\times64$ vs $128\times128$) impact on the 3-cycle shear layer perturbation and secondary pairing at $t \approx 3.5 - 5.0$.
2. **Enstrophy & Kinetic Energy Dissipation Rates**:
   - Compare peak enstrophy $\Omega_{\max}$ and dissipation rates $d\Omega/dt, dK/dt$ between Navier-Stokes and Adaptive PPR at $Re=10000$ vs Double $Re=20000$.
3. **Spectral Energy Cascade**:
   - Evaluate $E(k, t)$ slopes at $t=2.0$ and $t=5.0$ and verify how PPR artificial viscosity dampens high-wavenumber numerical oscillations without over-damping physical scales.
4. **PPR Artificial Viscosity Selectivity**:
   - Examine spatial cross-correlation $C_{\Sigma, |\omega|}$ to verify localized activation at intense vorticity cores and shear interfaces.
5. **Key Conclusions**:
   - Assess the overall performance and suitability of the PPR method for large eddy simulations of turbulent shear flows.

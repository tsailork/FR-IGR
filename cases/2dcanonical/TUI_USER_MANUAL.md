# 2D Canonical Turbulence Study Manager — User Manual & Reference Guide

Welcome to the **FR-IGR 2D Canonical Turbulence Study Manager (`manage_study.py`)**. This interactive Terminal User Interface (TUI) is designed specifically for computational fluid dynamics (CFD) researchers investigating the interaction of the **Phantom Pressure Relaxation (PPR / IGR)** artificial viscosity method with 2D turbulent flow structures, enstrophy cascades, and shock waves.

---

## 1. Directory Architecture & Naming Schema

The framework organizes canonical benchmark test cases inside `cases/2dcanonical/` into isolated subdirectories:

```
cases/2dcanonical/
├── manage_study.py                  <-- Interactive TUI Research Application (Main Entry Point)
├── TUI_USER_MANUAL.md               <-- User Manual & Reference Guide
├── kelvin_helmholtz/
│   ├── default/                      <-- PROTECTED Baseline Configuration Template
│   │   ├── domain.grid
│   │   └── inputs.dat
│   ├── run_ppr_adaptive_64x64_P2/    <-- Study Variant Run Subfolder
│   │   ├── domain.grid
│   │   ├── inputs.dat
│   │   ├── pv_outputs/               <-- VTK Output Datasets (.vtu)
│   │   └── csv_outputs/              <-- Time-series & Spectral CSVs
│   ├── run_navier_stokes_64x64_P2/
│   ├── run_no_regularization_64x64_P2/
│   ├── run_igr_parabolic_64x64_P2/
│   └── run_ref_pos_entropy_256x256_P3/ <-- Ultra-High Resolution Reference DNS
├── shock_vortex/
├── decaying_turbulence/
├── richtmyer_meshkov/
└── orszag_tang/
```

### Folder Rules & Conventions:
1. **`default/` Subfolder (Protected Template)**:
   - Holds the master baseline `inputs.dat` and `domain.grid` for a canonical testcase.
   - **Safety Protection**: The TUI enforces a strict guard that prevents cleaning or deleting the `default/` template folder.
2. **`run_<scheme>_<Nx>x<Ny>_P<deg>` Naming Schema**:
   - Encodes scheme variant (`ppr_adaptive`, `navier_stokes`, `no_regularization`, `igr_parabolic`, `ref_pos_entropy`), spatial mesh resolution ($N_x \times N_y$), and polynomial degree ($P_{\text{DEG}}$).

---

## 2. Keybind Reference & Detailed Functionality

| Keybind | Function | Description & Intended Research Workflow |
| :---: | :--- | :--- |
| `[1-5]` | **Toggle Case Selection** | Toggles selection checkmark `[x]` for individual canonical cases (`[1]` Kelvin-Helmholtz, `[2]` Shock-Vortex, `[3]` Decaying Turbulence, `[4]` Richtmyer-Meshkov, `[5]` Orszag-Tang). |
| `[A]` | **Select / Deselect All** | Multi-selects or deselects all 5 canonical cases in one keypress. |
| `[R]` | **Run Selected Variants** | Prepares and executes C++ simulations for all selected cases across the 4 standard scheme variants (`ppr_adaptive`, `navier_stokes`, `no_regularization`, `igr_parabolic`). |
| `[D]` | **Propagate Default Inputs** | **Default Input Propagation**: Reads modified parameters from `default/inputs.dat` and updates existing subcase directories (`run_*`). **Cleans dataset outputs** (`pv_outputs/*`, `csv_outputs/*`, `out.log`) while **preserving directory structures**. |
| `[C]` | **Clean & Delete** | Opens interactive clean sub-menu:<br>• `Option 1`: Wipes dataset outputs (`pv_outputs/*`, `csv_outputs/*`, `out.log`) for selected runs.<br>• `Option 2`: Deletes selected variant run folders entirely.<br>*(Template `default/` folder is strictly protected)*. |
| `[H]` | **Construct Case** | Interactive case construction wizard (all use Navier-Stokes):<br>• `1`: **Positivity Only** (PPR=false, default POS_LIMITER_EPS).<br>• `2`: **Positivity + Entropy** (PPR=false, ENTROPY_LIMITER_EPS=1e-3).<br>• `3`: **Positivity + PPR** (default PPR setup).<br>Prompts for resolution multiplier factor $M$ ($1\times, 2\times, 4\times, 10\times$) and scales $N_x, N_y$. **Creates variant directories in `NOT RUN` state** for user selection and execution via `[R]` or `[S]`. |
| `[E]` | **Edit Configuration** | Launches system `$EDITOR` (defaulting to `vim`, with fallback to `nano`/`vi`) in-place to edit `default/inputs.dat`. Temporarily suspends raw terminal mode before spawning `vim` to prevent terminal corruption. |
| `[P]` | **Run Diagnostic Analytics** | Invokes Python diagnostic scripts (`spectrum_2d.py`, `enstrophy.py`, `vortex_stats.py`, `ppr_localization.py`) on completed run datasets. |
| `[V]` | **Multi-Scheme Comparative Plots** | Runs `comparative_plots.py` to generate research-style publication figures overlaying PPR vs. Reference DNS vs. IGR vs. Navier-Stokes. |
| `[S]` | **Run All Sweeps** | One-key batch execution of all 5 canonical cases across all scheme variants and diagnostics. |
| `[M]` | **User Manual** | Opens this `TUI_USER_MANUAL.md` document in the text editor. |
| `[Q]` | **Quit TUI** | Gracefully exits the TUI application. |

---

## 3. Detailed Workflows for Researchers

### Workflow 1: Tuning Baseline Parameters & Propagating Changes
1. Press **`[E]`** to open `default/inputs.dat` in your system text editor.
2. Modify target parameters (e.g. change `T_FINAL = 2.0` or `RE = 10000.0`). Save and exit the editor.
3. Press **`[D]`** to propagate these baseline input changes to all existing subcases (`run_*`). The TUI will update the parameters in each subcase's `inputs.dat` and automatically clean previous dataset outputs while keeping all variant directory structures intact!

### Workflow 2: Constructing Custom Case Variants & Resolution Sweeps
1. Select your target case(s) in the TUI (e.g. press `[1]` for `kelvin_helmholtz`).
2. Press **`[H]`** to launch the interactive Case Construction Wizard.
3. Select your method (Option `1` Positivity Only, Option `2` Positivity + Entropy `1e-3`, Option `3` Positivity + PPR) and resolution multiplier $M$ (e.g. `2` for $2\times$ mesh resolution).
4. The TUI creates the subcase directory in `NOT RUN` state. You can inspect/edit its configuration (`[E]`) and execute it using **`[R]`** (Run Selected Variants) or **`[S]`** (Run All Sweeps).

### Workflow 3: Research Diagnostics & Publication Graphics
1. After running simulations, press **`[P]`** to execute the diagnostic suite:
   - `spectrum_2d.py`: 2D FFT energy spectra $E(k)$ and Helmholtz solenoidal ($\boldsymbol{u}_s$) / dilatational ($\boldsymbol{u}_c$) decomposition.
   - `enstrophy.py`: Kinetic energy $K(t)$ and enstrophy $\Omega(t)$ decay rates.
   - `vortex_stats.py`: Vortex core circulation $\Gamma(t)$ and quadrupole acoustic sound pressure $p'$.
   - `ppr_localization.py`: Spatial cross-correlations $C_{\Sigma, |\omega|}$ and $C_{\Sigma, [\nabla \cdot \boldsymbol{u}]_-}$.
2. Press **`[V]`** to generate publication-quality comparative plots (`300 DPI`, LaTeX typography) saved inside `summary_plots/`.

---

## 4. Edge Cases & Defensive Safeguards

* **Grid Connectivity Protection**: When grid resolution ($N_x \times N_y$) is modified, the TUI preserves physical domain bounds ($X_{\text{MIN}}, X_{\text{MAX}}, Y_{\text{MIN}}, Y_{\text{MAX}}$) and boundary condition strings (`0:R`, `INFLOW_SUPERSONIC:...`), preventing grid topology corruption.
* **Template Protection Guard**: The `default/` template folder cannot be deleted or wiped by the clean menu.
* **Crashed Run Recovery**: If a run fails due to extreme CFL or physical divergence, the TUI tags its status as `CRASHED` in red text and displays the error log location (`out.log`).
* **OpenMP Parallelism**: The TUI automatically detects CPU core availability and exports `OMP_NUM_THREADS` appropriately for standard vs. reference DNS runs.

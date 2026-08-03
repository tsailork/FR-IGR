#!/usr/bin/env python3
"""
Enstrophy & Kinetic Energy Temporal Decay Diagnostic Module.

Extracts time-series histories of:
1. Total Kinetic Energy: K(t) = 0.5 * integral(rho * (u^2 + v^2) dx dy)
2. Total Enstrophy: Omega(t) = 0.5 * integral(|vorticity|^2 dx dy)
3. Kinetic Energy Dissipation Rate: dK/dt
4. Enstrophy Dissipation Rate: dOmega/dt

Usage:
    python3 python_utilities/enstrophy.py --case cases/2dcanonical/decaying_turbulence
"""

import os
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from vtk_utils import load_vtk_dataset, resample_to_uniform_grid


def compute_vorticity_2d(u_grid, v_grid, dx, dy):
    """
    Computes 2D z-vorticity matrix w_z = dv/dx - du/dy using 2nd-order central differences.

    Parameters:
        u_grid (ndarray): (Ny, Nx) X-velocity matrix.
        v_grid (ndarray): (Ny, Nx) Y-velocity matrix.
        dx (float): Grid spacing along X.
        dy (float): Grid spacing along Y.

    Returns:
        vorticity (ndarray): (Ny, Nx) z-vorticity matrix.
    """
    dv_dx = np.gradient(v_grid, dx, axis=1)
    du_dy = np.gradient(u_grid, dy, axis=0)
    return dv_dx - du_dy


def analyze_case_enstrophy_history(case_dir, grid_res=(256, 256), output_dir=None):
    """
    Scans all VTK plot files in a case directory, calculates K(t) and Omega(t), and exports plots/CSV.

    Parameters:
        case_dir (str): Path to target case directory.
        grid_res (tuple): Interpolation grid dimensions (Nx, Ny).
        output_dir (str): Output directory for plots and CSV files.
    """
    pv_dir = os.path.join(case_dir, "pv_outputs")
    vtk_files = glob.glob(os.path.join(pv_dir, "plot_*.vtu"))
    if not vtk_files:
        vtk_files = glob.glob(os.path.join(pv_dir, "*.vtu"))

    if not vtk_files:
        raise FileNotFoundError(f"No VTK datasets found in: {pv_dir}")

    # Sort files chronologically by timestep index
    def extract_index(f):
        nums = re.findall(r'\d+', os.path.basename(f))
        return int(nums[-1]) if nums else 0

    vtk_files.sort(key=extract_index)

    time_list = []
    energy_list = []
    enstrophy_list = []
    max_vorticity_list = []

    print(f"[ENSTROPHY] Analyzing {len(vtk_files)} timesteps in {case_dir}...")

    for step_idx, file_path in enumerate(vtk_files):
        coords, fields = load_vtk_dataset(file_path)

        if 'u' not in fields or 'v' not in fields:
            continue

        # Resample onto uniform grid for domain integration
        X, Y, u_grid = resample_to_uniform_grid(coords, fields['u'], resolution=grid_res, method='cubic')
        _, _, v_grid = resample_to_uniform_grid(coords, fields['v'], resolution=grid_res, method='cubic')
        
        if 'rho' in fields:
            _, _, rho_grid = resample_to_uniform_grid(coords, fields['rho'], resolution=grid_res, method='cubic')
        else:
            rho_grid = np.ones_like(u_grid)

        Lx = np.max(X) - np.min(X)
        Ly = np.max(Y) - np.min(Y)
        Nx, Ny = grid_res
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        cell_area = dx * dy

        # Compute vorticity & enstrophy
        w_z = compute_vorticity_2d(u_grid, v_grid, dx, dy)
        
        K_t = 0.5 * np.sum(rho_grid * (u_grid**2 + v_grid**2)) * cell_area
        Omega_t = 0.5 * np.sum(w_z**2) * cell_area
        w_max = np.max(np.abs(w_z))

        # Approximate time from filename index or step count
        t_val = step_idx * 0.01

        time_list.append(t_val)
        energy_list.append(K_t)
        enstrophy_list.append(Omega_t)
        max_vorticity_list.append(w_max)

    t_arr = np.array(time_list)
    K_arr = np.array(energy_list)
    Omega_arr = np.array(enstrophy_list)
    w_max_arr = np.array(max_vorticity_list)

    # Compute dissipation rates via finite differences
    dK_dt = np.gradient(K_arr, t_arr) if len(t_arr) > 1 else np.zeros_like(K_arr)
    dOmega_dt = np.gradient(Omega_arr, t_arr) if len(t_arr) > 1 else np.zeros_like(Omega_arr)

    if output_dir is None:
        output_dir = os.path.join(case_dir, "csv_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Export CSV data
    csv_file = os.path.join(output_dir, "enstrophy_history.csv")
    out_matrix = np.column_stack((t_arr, K_arr, Omega_arr, w_max_arr, dK_dt, dOmega_dt))
    np.savetxt(csv_file, out_matrix, delimiter=",", 
               header="time,kinetic_energy,enstrophy,max_vorticity,dK_dt,dOmega_dt", comments="")
    print(f"[ENSTROPHY] Exported enstrophy history to: {csv_file}")

    # Plot Enstrophy and Kinetic Energy Decay Curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True)

    ax1.plot(t_arr, K_arr, 'b-o', linewidth=2, label=r'Kinetic Energy $K(t)$')
    ax1.set_ylabel(r'Kinetic Energy $K(t)$', fontsize=12)
    ax1.set_title(f'Kinetic Energy & Enstrophy Temporal Evolution\nCase: {os.path.basename(case_dir)}', fontsize=12)
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(t_arr, Omega_arr, 'r-s', linewidth=2, label=r'Enstrophy $\Omega(t)$')
    ax2.set_xlabel(r'Time $t$', fontsize=12)
    ax2.set_ylabel(r'Enstrophy $\Omega(t)$', fontsize=12)
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plot_file = os.path.join(output_dir, "enstrophy_history.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[ENSTROPHY] Generated history plot: {plot_file}")

    return csv_file, plot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enstrophy & Kinetic Energy Decay Utility")
    parser.add_argument("--case", type=str, required=True, help="Path to target case directory")
    parser.add_argument("--grid", type=int, default=256, help="Interpolation grid resolution (Nx=Ny)")
    args = parser.parse_args()

    analyze_case_enstrophy_history(args.case, grid_res=(args.grid, args.grid))

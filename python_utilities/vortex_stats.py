#!/usr/bin/env python3
"""
Vortex Core Tracking & Shock-Vortex Interaction Diagnostic Module.

Quantitative tracking of vortex core dynamics post-collision with shock waves:
1. Vortex Core Center Localization: (x_c, y_c) = argmax |omega(x, y)|
2. Vortex Circulation History: Gamma(R, t) = integral_{r <= R} omega(x, y) dx dy
3. Peak Vorticity Decay: omega_max(t) = max |omega(x, y)|
4. Acoustic Sound Pressure Perturbation Field: p'(x, y) = p(x, y) - p_mean

Usage:
    python3 python_utilities/vortex_stats.py --case cases/2dcanonical/shock_vortex
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from vtk_utils import load_vtk_dataset, resample_to_uniform_grid, find_latest_vtk_file
from enstrophy import compute_vorticity_2d


def analyze_vortex_core(coords, fields, grid_res=(256, 256), r_core=0.5):
    """
    Computes vortex core metrics including center location, circulation Gamma(R), and acoustic pressure.

    Parameters:
        coords (ndarray): (N, 2) array of (x, y) coordinates.
        fields (dict): PointData fields from VTK file.
        grid_res (tuple): Interpolation grid dimensions (Nx, Ny).
        r_core (float): Radius threshold for circulation evaluation.

    Returns:
        vortex_info (dict): Dictionary containing core center, max vorticity, circulation, and grids.
    """
    X, Y, u_grid = resample_to_uniform_grid(coords, fields['u'], resolution=grid_res, method='cubic')
    _, _, v_grid = resample_to_uniform_grid(coords, fields['v'], resolution=grid_res, method='cubic')
    
    if 'Pressure' in fields:
        _, _, p_grid = resample_to_uniform_grid(coords, fields['Pressure'], resolution=grid_res, method='cubic')
    else:
        p_grid = np.zeros_like(u_grid)

    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Nx, Ny = grid_res
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    cell_area = dx * dy

    # Compute vorticity matrix
    w_z = compute_vorticity_2d(u_grid, v_grid, dx, dy)

    # Locate vortex core center
    max_idx = np.unravel_index(np.argmax(np.abs(w_z)), w_z.shape)
    x_c = X[max_idx]
    y_c = Y[max_idx]
    w_max = w_z[max_idx]

    # Compute circulation Gamma within core radius R
    R_grid = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    core_mask = R_grid <= r_core
    gamma_core = np.sum(w_z[core_mask]) * cell_area

    # Acoustic pressure perturbation field p'
    p_mean = np.mean(p_grid)
    p_prime = p_grid - p_mean

    return {
        'x_c': x_c,
        'y_c': y_c,
        'w_max': w_max,
        'gamma_core': gamma_core,
        'X': X,
        'Y': Y,
        'w_z': w_z,
        'p_prime': p_prime
    }


def process_shock_vortex_case(case_dir, grid_res=(256, 256), output_dir=None):
    """
    Runs vortex core diagnostics on the latest dataset of a shock-vortex interaction case.

    Parameters:
        case_dir (str): Path to target case directory.
        grid_res (tuple): Interpolation grid dimensions (Nx, Ny).
        output_dir (str): Directory for output plots and CSV files.
    """
    pv_dir = os.path.join(case_dir, "pv_outputs")
    vtk_file = find_latest_vtk_file(pv_dir, prefix="plot_")

    print(f"[VORTEX STATS] Processing dataset: {vtk_file}")
    coords, fields = load_vtk_dataset(vtk_file)

    vort_info = analyze_vortex_core(coords, fields, grid_res=grid_res, r_core=0.4)

    if output_dir is None:
        output_dir = os.path.join(case_dir, "csv_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Print summary statistics to stdout
    print(f"[VORTEX STATS] Core Center: ({vort_info['x_c']:.4f}, {vort_info['y_c']:.4f})")
    print(f"[VORTEX STATS] Peak Vorticity w_max: {vort_info['w_max']:.4f}")
    print(f"[VORTEX STATS] Circulation Gamma(r=0.4): {vort_info['gamma_core']:.6f}")

    # Export metrics to CSV
    csv_file = os.path.join(output_dir, "vortex_core_metrics.csv")
    with open(csv_file, "w") as f:
        f.write("x_center,y_center,w_max,gamma_core\n")
        f.write(f"{vort_info['x_c']},{vort_info['y_c']},{vort_info['w_max']},{vort_info['gamma_core']}\n")

    # Contour Plot: Vorticity & Acoustic Pressure Field
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Subplot 1: Vorticity Contours
    c1 = ax1.contourf(vort_info['X'], vort_info['Y'], vort_info['w_z'], levels=40, cmap='RdBu_r')
    ax1.plot(vort_info['x_c'], vort_info['y_c'], 'ko', markersize=6, label='Vortex Center')
    fig.colorbar(c1, ax=ax1, label=r'Vorticity $\omega_z$')
    ax1.set_title(r'Vorticity Field $\omega_z$', fontsize=12)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.legend(loc='upper right')

    # Subplot 2: Acoustic Sound Pressure Perturbations p'
    c2 = ax2.contourf(vort_info['X'], vort_info['Y'], vort_info['p_prime'], levels=40, cmap='seismic')
    fig.colorbar(c2, ax=ax2, label=r'Pressure Fluctuation $p\'$')
    ax2.set_title(r'Acoustic Sound Field $p\'$', fontsize=12)
    ax2.set_xlabel('X', fontsize=10)

    plot_file = os.path.join(output_dir, "vortex_core_analysis.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[VORTEX STATS] Generated core plot: {plot_file}")

    return csv_file, plot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vortex Core & Acoustic Wave Diagnostic Utility")
    parser.add_argument("--case", type=str, required=True, help="Path to target case directory")
    parser.add_argument("--grid", type=int, default=256, help="Interpolation grid resolution (Nx=Ny)")
    args = parser.parse_args()

    process_shock_vortex_case(args.case, grid_res=(args.grid, args.grid))

#!/usr/bin/env python3
"""
PPR Regularization Spatial Selectivity & Mask Correlation Module.

Evaluates how selectively the Phantom Pressure Relaxation (PPR / IGR) entropic 
pressure field Sigma(x, y) activates at shocks versus smooth turbulent vortices.

Normalized Spatial Cross-Correlation Metrics:
1. C_{Sigma, |omega|} : Correlation between regularizer viscosity Sigma and local vorticity |omega|.
   TARGET: Near ZERO (0.0). High correlation indicates unphysical damping of vortices.
2. C_{Sigma, [div u]_-}: Correlation between regularizer viscosity Sigma and compressive velocity divergence.
   TARGET: Near ONE (1.0). High correlation confirms strict shock-capturing localization.

Usage:
    python3 python_utilities/ppr_localization.py --case cases/2dcanonical/shock_vortex
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from vtk_utils import load_vtk_dataset, resample_to_uniform_grid, find_latest_vtk_file
from enstrophy import compute_vorticity_2d


def compute_divergence_2d(u_grid, v_grid, dx, dy):
    """
    Computes 2D velocity divergence div_u = du/dx + dv/dy using 2nd-order central differences.
    """
    du_dx = np.gradient(u_grid, dx, axis=1)
    dv_dy = np.gradient(v_grid, dy, axis=0)
    return du_dx + dv_dy


def evaluate_ppr_spatial_selectivity(coords, fields, grid_res=(256, 256)):
    """
    Calculates spatial cross-correlations between PPR entropic pressure Sigma, vorticity |omega|, 
    and compressive divergence [div u]_.

    Parameters:
        coords (ndarray): Nodal coordinates.
        fields (dict): PointData arrays from VTK dataset.
        grid_res (tuple): Grid interpolation resolution (Nx, Ny).

    Returns:
        results (dict): Cross-correlation values and 2D spatial matrices.
    """
    X, Y, u_grid = resample_to_uniform_grid(coords, fields['u'], resolution=grid_res, method='cubic')
    _, _, v_grid = resample_to_uniform_grid(coords, fields['v'], resolution=grid_res, method='cubic')

    if 'Sigma' in fields:
        _, _, sigma_grid = resample_to_uniform_grid(coords, fields['Sigma'], resolution=grid_res, method='cubic')
    else:
        sigma_grid = np.zeros_like(u_grid)

    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Nx, Ny = grid_res
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # Compute vorticity and divergence
    w_z = compute_vorticity_2d(u_grid, v_grid, dx, dy)
    vort_mag = np.abs(w_z)

    div_u = compute_divergence_2d(u_grid, v_grid, dx, dy)
    compressive_div = np.maximum(0.0, -div_u) # Pure compression (shocks)

    # Compute normalized cross-correlations
    sigma_norm = np.linalg.norm(sigma_grid)
    vort_norm  = np.linalg.norm(vort_mag)
    comp_norm  = np.linalg.norm(compressive_div)

    corr_vort = 0.0
    if sigma_norm > 1e-12 and vort_norm > 1e-12:
        corr_vort = np.sum(sigma_grid * vort_mag) / (sigma_norm * vort_norm)

    corr_comp = 0.0
    if sigma_norm > 1e-12 and comp_norm > 1e-12:
        corr_comp = np.sum(sigma_grid * compressive_div) / (sigma_norm * comp_norm)

    return {
        'corr_vort': corr_vort,
        'corr_comp': corr_comp,
        'X': X,
        'Y': Y,
        'sigma': sigma_grid,
        'vort_mag': vort_mag,
        'compressive_div': compressive_div
    }


def analyze_ppr_case_localization(case_dir, grid_res=(256, 256), output_dir=None):
    """
    Runs spatial correlation diagnostics on a case dataset and generates comparison plots.

    Parameters:
        case_dir (str): Path to target case directory.
        grid_res (tuple): Interpolation grid dimensions (Nx, Ny).
        output_dir (str): Output directory for plots and CSV metrics.
    """
    pv_dir = os.path.join(case_dir, "pv_outputs")
    vtk_file = find_latest_vtk_file(pv_dir, prefix="sol_")

    print(f"[PPR LOCALIZATION] Processing dataset: {vtk_file}")
    coords, fields = load_vtk_dataset(vtk_file)

    res = evaluate_ppr_spatial_selectivity(coords, fields, grid_res=grid_res)

    print(f"[PPR LOCALIZATION] Spatial Correlation C_(Sigma, |omega|) [Target ~ 0.0]: {res['corr_vort']:.6f}")
    print(f"[PPR LOCALIZATION] Spatial Correlation C_(Sigma, [div u]_-) [Target ~ 1.0]: {res['corr_comp']:.6f}")

    if output_dir is None:
        output_dir = os.path.join(case_dir, "csv_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Export metrics to CSV
    csv_file = os.path.join(output_dir, "ppr_correlation_metrics.csv")
    with open(csv_file, "w") as f:
        f.write("corr_vorticity,corr_compressive_div\n")
        f.write(f"{res['corr_vort']},{res['corr_comp']}\n")

    # Spatial Comparison Contour Plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # Subplot 1: PPR Entropic Pressure Sigma
    c1 = ax1.contourf(res['X'], res['Y'], res['sigma'], levels=30, cmap='inferno')
    fig.colorbar(c1, ax=ax1, label=r'PPR Viscosity $\Sigma$')
    ax1.set_title(r'PPR Field $\Sigma(x, y)$', fontsize=12)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)

    # Subplot 2: Vorticity Magnitude |omega|
    c2 = ax2.contourf(res['X'], res['Y'], res['vort_mag'], levels=30, cmap='viridis')
    fig.colorbar(c2, ax=ax2, label=r'Vorticity $|\omega|$')
    ax2.set_title(r'Vorticity Magnitude $|\omega|$' + f'\n' + fr'$C_{{(\Sigma, |\omega|)}} = {res["corr_vort"]:.4f}$', fontsize=11)
    ax2.set_xlabel('X', fontsize=10)

    # Subplot 3: Compressive Divergence [div u]_
    c3 = ax3.contourf(res['X'], res['Y'], res['compressive_div'], levels=30, cmap='magma')
    fig.colorbar(c3, ax=ax3, label=r'Compression $[\nabla \cdot u]_-$')
    ax3.set_title(r'Compressive Divergence $[\nabla \cdot u]_-$' + f'\n' + fr'$C_{{(\Sigma, [\nabla \cdot u]_-)}} = {res["corr_comp"]:.4f}$', fontsize=11)
    ax3.set_xlabel('X', fontsize=10)

    plot_file = os.path.join(output_dir, "ppr_spatial_localization.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[PPR LOCALIZATION] Generated localization plot: {plot_file}")

    return csv_file, plot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPR Regularization Spatial Selectivity Diagnostic Utility")
    parser.add_argument("--case", type=str, required=True, help="Path to target case directory")
    parser.add_argument("--grid", type=int, default=256, help="Interpolation grid resolution (Nx=Ny)")
    args = parser.parse_args()

    analyze_ppr_case_localization(args.case, grid_res=(args.grid, args.grid))

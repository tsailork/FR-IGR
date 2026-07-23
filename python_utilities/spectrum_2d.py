#!/usr/bin/env python3
"""
2D Turbulent Kinetic Energy Spectrum & Helmholtz Decomposition Module.

Performs 2D Fast Fourier Transform (FFT) spectral analysis on fluid velocity fields, 
decomposes the velocity vector into Solenoidal (rotational, div-free) and Dilatational 
(compressible, curl-free) modes via Helmholtz decomposition, and computes shell-averaged 
1D kinetic energy spectra E(k), E_s(k), and E_c(k).

Theoretical Power Law Benchmarks:
- k^-3 : Kraichnan 2D enstrophy direct cascade scaling law.
- k^-5/3: Kolmogorov 3D energy cascade scaling law.
- k^-2 : Kadomtsev-Petviashvili / shocklet discontinuity energy spectrum.

Usage:
    python3 python_utilities/spectrum_2d.py --case cases/2dcanonical/decaying_turbulence --grid 256
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from vtk_utils import load_vtk_dataset, resample_to_uniform_grid, find_latest_vtk_file


def helmholtz_decomposition_2d(u_grid, v_grid, Lx=1.0, Ly=1.0):
    """
    Performs exact 2D Helmholtz decomposition of velocity field into solenoidal and dilatational modes.

    u = u_solenoidal + u_dilatational
    where div(u_solenoidal) = 0 and curl(u_dilatational) = 0.

    Parameters:
        u_grid (ndarray): (Ny, Nx) matrix of X-velocity components.
        v_grid (ndarray): (Ny, Nx) matrix of Y-velocity components.
        Lx (float): Physical domain length along X.
        Ly (float): Physical domain length along Y.

    Returns:
        u_s, v_s (ndarray): Solenoidal velocity components in physical space.
        u_c, v_c (ndarray): Dilatational velocity components in physical space.
        u_hat, v_hat (ndarray): Full velocity Fourier transform matrices.
        u_s_hat, v_s_hat (ndarray): Solenoidal velocity Fourier transform matrices.
        u_c_hat, v_c_hat (ndarray): Dilatational velocity Fourier transform matrices.
        kx, ky, k_mag (ndarray): Wavenumber coordinate matrices.
    """
    Ny, Nx = u_grid.shape

    # 2D Fast Fourier Transforms
    u_hat = np.fft.fft2(u_grid) / (Nx * Ny)
    v_hat = np.fft.fft2(v_grid) / (Nx * Ny)

    # Physical wavenumber grids (rad/m)
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    Kx, Ky = np.meshgrid(kx_1d, ky_1d)
    K_sq = Kx**2 + Ky**2
    K_mag = np.sqrt(K_sq)

    # Allocate spectral arrays for solenoidal and dilatational modes
    u_c_hat = np.zeros_like(u_hat, dtype=complex)
    v_c_hat = np.zeros_like(v_hat, dtype=complex)
    u_s_hat = np.zeros_like(u_hat, dtype=complex)
    v_s_hat = np.zeros_like(v_hat, dtype=complex)

    # Non-zero wavenumber modes: Projection onto wave vector k = (Kx, Ky)
    mask = K_sq > 1e-12
    dot_k_u = Kx[mask] * u_hat[mask] + Ky[mask] * v_hat[mask]

    # Dilatational mode projection (parallel to k): u_c_hat = (k . u_hat) * k / |k|^2
    u_c_hat[mask] = (dot_k_u / K_sq[mask]) * Kx[mask]
    v_c_hat[mask] = (dot_k_u / K_sq[mask]) * Ky[mask]

    # Solenoidal mode projection (perpendicular to k): u_s_hat = u_hat - u_c_hat
    u_s_hat[mask] = u_hat[mask] - u_c_hat[mask]
    v_s_hat[mask] = v_hat[mask] - v_c_hat[mask]

    # DC mode (k = 0): Assign to solenoidal component
    u_s_hat[0, 0] = u_hat[0, 0]
    v_s_hat[0, 0] = v_hat[0, 0]

    # Inverse FFT back to physical space
    u_s = np.real(np.fft.ifft2(u_s_hat * (Nx * Ny)))
    v_s = np.real(np.fft.ifft2(v_s_hat * (Nx * Ny)))
    u_c = np.real(np.fft.ifft2(u_c_hat * (Nx * Ny)))
    v_c = np.real(np.fft.ifft2(v_c_hat * (Nx * Ny)))

    return (u_s, v_s), (u_c, v_c), (u_hat, v_hat), (u_s_hat, v_s_hat), (u_c_hat, v_c_hat), (Kx, Ky, K_mag)


def compute_shell_averaged_spectrum_2d(u_hat, v_hat, Kx, Ky, Lx=1.0, Ly=1.0):
    """
    Computes 1D shell-averaged kinetic energy spectrum E(k) from 2D Fourier transforms.

    E_2D(kx, ky) = 0.5 * (|u_hat|^2 + |v_hat|^2)
    E(k) = sum_{k - 0.5 <= |k'| < k + 0.5} E_2D(k')

    Parameters:
        u_hat, v_hat (ndarray): 2D Fourier coefficients of velocity field.
        Kx, Ky (ndarray): Wavenumber meshgrid matrices.
        Lx, Ly (float): Physical domain lengths.

    Returns:
        k_bins (ndarray): 1D integer wavenumber bin array.
        E_k (ndarray): 1D kinetic energy spectral density array.
    """
    Ny, Nx = u_hat.shape
    dk_x = 2.0 * np.pi / Lx
    dk_y = 2.0 * np.pi / Ly
    dk = min(dk_x, dk_y)

    K_mag = np.sqrt(Kx**2 + Ky**2) / dk
    E_2D = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    k_max = int(min(Nx, Ny) // 2)
    k_bins = np.arange(1, k_max)
    E_k = np.zeros_like(k_bins, dtype=float)

    for i, k in enumerate(k_bins):
        shell_mask = (K_mag >= (k - 0.5)) & (K_mag < (k + 0.5))
        E_k[i] = np.sum(E_2D[shell_mask])

    return k_bins, E_k


def analyze_case_energy_spectrum(case_dir, grid_res=(256, 256), output_dir=None):
    """
    Performs full 2D spectral energy analysis and Helmholtz decomposition on a case dataset.

    Parameters:
        case_dir (str): Path to case directory (e.g. 'cases/2dcanonical/decaying_turbulence').
        grid_res (tuple): Interpolation grid dimensions (Nx, Ny).
        output_dir (str): Output directory for plots and CSV spectrum data.
    """
    pv_outputs_dir = os.path.join(case_dir, "pv_outputs")
    vtk_file = find_latest_vtk_file(pv_outputs_dir, prefix="plot_")

    print(f"[SPECTRUM 2D] Processing file: {vtk_file}")
    coords, fields = load_vtk_dataset(vtk_file)

    if 'u' not in fields or 'v' not in fields:
        raise KeyError(f"Velocity components 'u' and 'v' missing from VTK dataset: {vtk_file}")

    # Resample onto uniform Cartesian grid for FFT
    X, Y, u_grid = resample_to_uniform_grid(coords, fields['u'], resolution=grid_res, method='cubic')
    _, _, v_grid = resample_to_uniform_grid(coords, fields['v'], resolution=grid_res, method='cubic')

    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)

    # Perform Helmholtz Decomposition
    (u_s, v_s), (u_c, v_c), (u_hat, v_hat), (u_s_hat, v_s_hat), (u_c_hat, v_c_hat), (Kx, Ky, K_mag) = \
        helmholtz_decomposition_2d(u_grid, v_grid, Lx=Lx, Ly=Ly)

    # Compute 1D Shell-Averaged Spectra
    k_bins, E_total = compute_shell_averaged_spectrum_2d(u_hat, v_hat, Kx, Ky, Lx=Lx, Ly=Ly)
    _, E_solenoidal = compute_shell_averaged_spectrum_2d(u_s_hat, v_s_hat, Kx, Ky, Lx=Lx, Ly=Ly)
    _, E_dilatational = compute_shell_averaged_spectrum_2d(u_c_hat, v_c_hat, Kx, Ky, Lx=Lx, Ly=Ly)

    if output_dir is None:
        output_dir = os.path.join(case_dir, "csv_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Export CSV numerical data
    csv_file = os.path.join(output_dir, "energy_spectrum_2d.csv")
    data_matrix = np.column_stack((k_bins, E_total, E_solenoidal, E_dilatational))
    np.savetxt(csv_file, data_matrix, delimiter=",", header="wavenumber_k,E_total,E_solenoidal,E_dilatational", comments="")
    print(f"[SPECTRUM 2D] Exported numerical spectrum data to: {csv_file}")

    # Plot 1D Energy Spectra with Theoretical Cascades
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.loglog(k_bins, E_total, 'k-o', linewidth=2, label=r'Total Kinetic Energy $E(k)$')
    ax.loglog(k_bins, E_solenoidal, 'b--s', linewidth=1.5, label=r'Solenoidal $E_s(k)$ (Vortical)')
    ax.loglog(k_bins, E_dilatational, 'r-.^', linewidth=1.5, label=r'Dilatational $E_c(k)$ (Compressible)')

    # Add theoretical reference power law lines
    k_ref = k_bins[(k_bins >= 3) & (k_bins <= max(4, len(k_bins)//3))]
    if len(k_ref) > 1:
        # Kraichnan 2D enstrophy cascade k^-3
        C_k3 = E_total[k_bins == k_ref[0]][0] * (k_ref[0]**3)
        ax.loglog(k_ref, C_k3 * (k_ref**(-3.0)), 'g:', linewidth=2, label=r'Kraichnan $k^{-3}$ (Enstrophy Cascade)')
        
        # Shocklet / Discontinuity k^-2
        C_k2 = E_total[k_bins == k_ref[0]][0] * (k_ref[0]**2)
        ax.loglog(k_ref, C_k2 * (k_ref**(-2.0)), 'm:', linewidth=2, label=r'Shocklet $k^{-2}$')

    ax.set_xlabel(r'Wavenumber $k$', fontsize=12)
    ax.set_ylabel(r'Energy Density $E(k)$', fontsize=12)
    ax.set_title(f'2D Turbulent Energy Spectrum & Helmholtz Decomposition\nCase: {os.path.basename(case_dir)}', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=10, loc='best')

    plot_file = os.path.join(output_dir, "energy_spectrum_2d.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[SPECTRUM 2D] Generated spectrum plot: {plot_file}")

    return csv_file, plot_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Energy Spectrum & Helmholtz Decomposition Utility")
    parser.add_argument("--case", type=str, required=True, help="Path to target case directory")
    parser.add_argument("--grid", type=int, default=256, help="Interpolation grid resolution (Nx=Ny)")
    args = parser.parse_args()

    analyze_case_energy_spectrum(args.case, grid_res=(args.grid, args.grid))

#!/usr/bin/env python3
"""
Publication-Quality Comparative Analysis & Multi-Scheme Plotting Utility.

Generates research-style comparative plots comparing:
1. Reference Ultra-High Resolution Solution (10x mesh + Positivity/Entropy limiters)
2. PPR (Phantom Pressure Relaxation with adaptive theta)
3. IGR (Isotropic Gradient Regularization)
4. Navier-Stokes Baseline (Physical viscosity)
5. Un-regularized Euler Baseline

Graphics & Styling:
- Publication standards (LaTeX fonts, 300 DPI, color-blind friendly palettes).
- Overlaid energy spectra E(k), enstrophy decay Omega(t), and vortex core circulation Gamma(t).

Usage:
    python3 python_utilities/comparative_plots.py --case_dir cases/2dcanonical/decaying_turbulence
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from vtk_utils import load_vtk_dataset, resample_to_uniform_grid, find_latest_vtk_file
from spectrum_2d import helmholtz_decomposition_2d, compute_shell_averaged_spectrum_2d

# Apply professional publication plot formatting settings
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern", "DejaVu Serif", "Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "lines.markersize": 5
})

SCHEME_STYLE = {
    "runs_reference_10x": {"label": "Reference DNS (10x Grid)", "color": "#000000", "ls": "-", "marker": "o"},
    "runs_ppr_adaptive":  {"label": "PPR (Adaptive \u03b8)", "color": "#d62728", "ls": "-", "marker": "s"},
    "runs_igr_parabolic": {"label": "IGR (Parabolic)", "color": "#1f77b4", "ls": "--", "marker": "^"},
    "runs_navier_stokes": {"label": "Navier-Stokes (Physical)", "color": "#2ca02c", "ls": "-.", "marker": "d"},
    "runs_no_regularization": {"label": "Un-regularized Euler", "color": "#ff7f0e", "ls": ":", "marker": "x"}
}


def get_case_style(vname, case_path):
    if "kelvin_helmholtz" in case_path.lower():
        parts = vname.split("_")
        
        # Identify method
        if "ppr" in parts:
            method = "ppr_adaptive"
        else:
            method = "navier_stokes"
            
        # Find resolution
        res = "64x64" if "64x64" in parts else "128x128"
        
        # Find P
        p_deg = "P2" if "P2" in parts else "P3"
        
        # Find Re
        re_val = "Re10000" if "Re10000" in parts else "Re20000"
        
        # Set color based on (resolution, p_deg) -> 4 distinct colors (RGB + Orange)
        color_map = {
            ("64x64", "P2"): "#1f77b4",   # Royal Blue
            ("64x64", "P3"): "#ff7f0e",   # Bright Orange
            ("128x128", "P2"): "#2ca02c", # Forest Green
            ("128x128", "P3"): "#d62728"  # Crimson Red
        }
        color = color_map.get((res, p_deg), "#7f7f7f")
        
        # PPR is solid lines (-), NS is dashed lines (--)
        ls = "-" if method == "ppr_adaptive" else "--"
        
        # Re10k is circle (o), Re20k is square (s)
        marker = "o" if re_val == "Re10000" else "s"
        
        # Set nice labels
        method_label = "PPR" if method == "ppr_adaptive" else "NS"
        re_label = "Re10k" if re_val == "Re10000" else "Re20k"
        label = f"{method_label} | {res} | {p_deg} | {re_label}"
        
        return {
            "label": label,
            "color": color,
            "ls": ls,
            "marker": marker
        }
    else:
        return SCHEME_STYLE.get(vname, {"label": vname, "color": "#7f7f7f", "ls": "-", "marker": ""})


def plot_comparative_energy_spectra(case_path, grid_res=(256, 256), output_dir=None):
    """
    Plots overlaid energy spectra E(k) across all scheme variants for a single benchmark case.
    """
    variant_dirs = [d for d in glob.glob(os.path.join(case_path, "case_*")) + glob.glob(os.path.join(case_path, "run_*")) if os.path.isdir(d)]
    if not variant_dirs:
        print(f"[COMPARATIVE PLOTS] No variant runs found in: {case_path}")
        return

    is_khi = "kelvin_helmholtz" in case_path.lower()
    if is_khi:
        # Sort variant_dirs to group entries sharing the same color next to each other
        def sort_key(d):
            vname = os.path.basename(d)
            parts = vname.split("_")
            res = "64x64" if "64x64" in parts else "128x128"
            p_deg = "P2" if "P2" in parts else "P3"
            method = "ppr_adaptive" if "ppr" in parts else "navier_stokes"
            re_val = "Re10000" if "Re10000" in parts else "Re20000"
            
            res_order = 0 if res == "64x64" else 1
            p_order = 0 if p_deg == "P2" else 1
            method_order = 0 if method == "ppr_adaptive" else 1
            re_order = 0 if re_val == "Re10000" else 1
            return (res_order, p_order, method_order, re_order)
        variant_dirs.sort(key=sort_key)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for vdir in variant_dirs:
        vname = os.path.basename(vdir)
        style = get_case_style(vname, case_path)

        pv_dir = os.path.join(vdir, "pv_outputs")
        if not os.path.exists(pv_dir):
            continue

        try:
            vtk_file = find_latest_vtk_file(pv_dir, prefix="plot_")
            coords, fields = load_vtk_dataset(vtk_file)
            X, Y, u_grid = resample_to_uniform_grid(coords, fields['u'], resolution=grid_res)
            _, _, v_grid = resample_to_uniform_grid(coords, fields['v'], resolution=grid_res)

            Lx = np.max(X) - np.min(X)
            Ly = np.max(Y) - np.min(Y)

            _, _, (u_hat, v_hat), _, _, (Kx, Ky, _) = helmholtz_decomposition_2d(u_grid, v_grid, Lx=Lx, Ly=Ly)
            k_bins, E_total = compute_shell_averaged_spectrum_2d(u_hat, v_hat, Kx, Ky, Lx=Lx, Ly=Ly)

            ax.loglog(k_bins, E_total, label=style['label'], color=style['color'], 
                      linestyle=style['ls'], marker=style['marker'], markevery=max(1, len(k_bins)//15))
        except Exception as e:
            print(f"[WARNING] Could not process variant {vname}: {e}")

    # Theoretical Power Law Overlays
    if 'k_bins' in locals() and len(k_bins) > 5:
        k_ref = k_bins[(k_bins >= 3) & (k_bins <= max(4, len(k_bins)//3))]
        C_k3 = E_total[k_bins == k_ref[0]][0] * (k_ref[0]**3)
        ax.loglog(k_ref, C_k3 * (k_ref**(-3.0)), 'k:', linewidth=2, label=r'Kraichnan $k^{-3}$')

    ax.set_xlabel(r'Wavenumber $k$', fontsize=12)
    ax.set_ylabel(r'Energy Density $E(k)$', fontsize=12)
    ax.set_title(f'Comparative Energy Spectrum: {os.path.basename(case_path)}', fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    if is_khi:
        ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.95)
    else:
        ax.legend(fontsize=10, loc='best', framealpha=0.9)

    if output_dir is None:
        output_dir = os.path.join(case_path, "summary_plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_file = os.path.join(output_dir, "comparative_energy_spectrum.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[COMPARATIVE PLOTS] Saved comparative energy spectrum to: {plot_file}")


def plot_comparative_enstrophy_history(case_path, output_dir=None):
    """
    Plots overlaid Enstrophy Omega(t) and Kinetic Energy K(t) decay curves across variants.
    """
    variant_dirs = [d for d in glob.glob(os.path.join(case_path, "*")) if os.path.isdir(d) and os.path.basename(d) not in ["summary_plots", "pv_outputs", "csv_outputs", "anims", "__pycache__"]]
    if not variant_dirs:
        return

    is_khi = "kelvin_helmholtz" in case_path.lower()
    if is_khi:
        # Sort variant_dirs to group entries sharing the same color next to each other
        def sort_key(d):
            vname = os.path.basename(d)
            parts = vname.split("_")
            res = "64x64" if "64x64" in parts else "128x128"
            p_deg = "P2" if "P2" in parts else "P3"
            method = "ppr_adaptive" if "ppr" in parts else "navier_stokes"
            re_val = "Re10000" if "Re10000" in parts else "Re20000"
            
            res_order = 0 if res == "64x64" else 1
            p_order = 0 if p_deg == "P2" else 1
            method_order = 0 if method == "ppr_adaptive" else 1
            re_order = 0 if re_val == "Re10000" else 1
            return (res_order, p_order, method_order, re_order)
        variant_dirs.sort(key=sort_key)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=300, sharex=True)

    default_k_limits = None
    default_om_limits = None

    for vdir in variant_dirs:
        vname = os.path.basename(vdir)
        csv_file = os.path.join(vdir, "csv_outputs", "enstrophy_history.csv")
        if os.path.exists(csv_file):
            try:
                data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
                K_norm = data[:, 1] / data[0, 1]
                Omega_norm = data[:, 2] / data[0, 2]

                if vname == "default" or "navier_stokes" in vname or default_k_limits is None:
                    k_min, k_max = np.nanmin(K_norm), np.nanmax(K_norm)
                    om_min, om_max = np.nanmin(Omega_norm), np.nanmax(Omega_norm)
                    k_margin = max(0.1, 0.25 * (k_max - k_min))
                    om_margin = max(0.2, 0.25 * (om_max - om_min))
                    default_k_limits = (max(0.0, k_min - k_margin), k_max + k_margin)
                    default_om_limits = (max(0.0, om_min - om_margin), om_max + om_margin)
            except Exception:
                pass

    is_khi = "kelvin_helmholtz" in case_path.lower()

    for vdir in variant_dirs:
        vname = os.path.basename(vdir)
        style = get_case_style(vname, case_path)

        csv_file = os.path.join(vdir, "csv_outputs", "enstrophy_history.csv")
        if not os.path.exists(csv_file):
            continue

        try:
            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            t_arr = data[:, 0]
            K_arr = data[:, 1]
            Omega_arr = data[:, 2]

            ax1.plot(t_arr, K_arr / K_arr[0], label=style['label'], color=style['color'], 
                     linestyle=style['ls'], marker=style['marker'], markevery=max(1, len(t_arr)//10))
            ax2.plot(t_arr, Omega_arr / Omega_arr[0], label=style['label'], color=style['color'], 
                     linestyle=style['ls'], marker=style['marker'], markevery=max(1, len(t_arr)//10))
        except Exception as e:
            print(f"[WARNING] Could not read CSV for {vname}: {e}")

    ax1.set_ylabel(r'Normalized Kinetic Energy $K(t)/K_0$', fontsize=11)
    ax1.set_title(f'Comparative Decay History: {os.path.basename(case_path)}', fontsize=13)
    ax1.grid(True, ls="--", alpha=0.4)

    if is_khi:
        ax1.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.95)
    else:
        ax1.legend(fontsize=9, loc='best')

    if not is_khi and default_k_limits:
        ax1.set_ylim(default_k_limits)

    ax2.set_xlabel(r'Time $t$', fontsize=11)
    ax2.set_ylabel(r'Normalized Enstrophy $\Omega(t)/\Omega_0$', fontsize=11)
    ax2.grid(True, ls="--", alpha=0.4)

    if not is_khi and default_om_limits:
        ax2.set_ylim(default_om_limits)

    if output_dir is None:
        output_dir = os.path.join(case_path, "summary_plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_file = os.path.join(output_dir, "comparative_enstrophy_history.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[COMPARATIVE PLOTS] Saved comparative enstrophy history to: {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparative Multi-Scheme Plotting Utility")
    parser.add_argument("--case_dir", type=str, required=True, help="Path to canonical case directory")
    args = parser.parse_args()

    plot_comparative_energy_spectra(args.case_dir)
    plot_comparative_enstrophy_history(args.case_dir)

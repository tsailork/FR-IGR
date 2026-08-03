#!/usr/bin/env python3
"""
Automated Diagnostic Runner & Multi-Case Colorful Comparative Plotter
====================================================================
Runs turbulence decay and enstrophy diagnostics across target test problems:
- Decaying Turbulence (decaying_turbulence: default, pos-ent, pos-only)
- Orszag-Tang Vortex (orszag_tang: default, pos-ent, pos-only)

Generates publication-quality comparative coplots with distinct vibrant colors.
"""

import os
import sys
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_UTILS = os.path.join(PROJECT_DIR, "python_utilities")
CANONICAL_DIR = os.path.join(PROJECT_DIR, "cases", "2dcanonical")

# Curated vibrant publication color palette (NO GREY!)
COLOR_PALETTE = [
    "#1f77b4",  # Royal Blue
    "#d62728",  # Crimson Red
    "#2ca02c",  # Forest Green
    "#ff7f0e",  # Bright Orange
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#17becf"   # Teal
]

MARKERS = ["o", "s", "^", "d", "v", "D", "p", "P"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

CASE_LABELS = {
    "default":  "Default (Baseline)",
    "pos-ent":  "Positivity + Entropy Limiter",
    "pos-only": "Positivity Limiter Only"
}


def run_case_diagnostics(case_dir, grid=256):
    """Executes spectrum_2d.py and enstrophy.py for a single case directory if needed."""
    pv_dir = os.path.join(case_dir, "pv_outputs")
    csv_dir = os.path.join(case_dir, "csv_outputs")
    os.makedirs(csv_dir, exist_ok=True)

    vtu_files = glob.glob(os.path.join(pv_dir, "*.vtu"))
    if not vtu_files:
        print(f"[SKIP] No VTU files in {pv_dir}")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHON_UTILS
    env["PYTHONUNBUFFERED"] = "1"

    spec_csv = os.path.join(csv_dir, "energy_spectrum_2d.csv")
    if not os.path.exists(spec_csv):
        print(f"[RUN SPECTRUM] Analyzing 2D spectrum for {os.path.basename(case_dir)}...", flush=True)
        subprocess.run(["python3", "-u", os.path.join(PYTHON_UTILS, "spectrum_2d.py"), "--case", case_dir, "--grid", str(grid)], env=env)
    else:
        print(f"[SKIP SPECTRUM] {os.path.basename(case_dir)} spectrum already analyzed.", flush=True)

    enst_csv = os.path.join(csv_dir, "enstrophy_history.csv")
    if not os.path.exists(enst_csv):
        print(f"[RUN ENSTROPHY] Analyzing enstrophy history for {os.path.basename(case_dir)}...", flush=True)
        subprocess.run(["python3", "-u", os.path.join(PYTHON_UTILS, "enstrophy.py"), "--case", case_dir, "--grid", str(grid)], env=env)
    else:
        print(f"[SKIP ENSTROPHY] {os.path.basename(case_dir)} enstrophy history already analyzed.", flush=True)


def generate_coplots(problem_dir):
    """Generates comparative coplots comparing cases in problem_dir with distinct colors."""
    problem_name = os.path.basename(problem_dir)
    print(f"\n[COPLOTS] Generating comparative plots for problem: {problem_name}", flush=True)

    subdirs = sorted([d for d in glob.glob(os.path.join(problem_dir, "*")) 
                      if os.path.isdir(d) and os.path.basename(d) not in ["summary_plots", "pv_outputs", "csv_outputs", "anims", "__pycache__"]])

    if not subdirs:
        print(f"[WARNING] No case subdirectories found in {problem_dir}")
        return

    summary_dir = os.path.join(problem_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    # Styling settings for publication quality
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
        "lines.linewidth": 2.0,
        "lines.markersize": 5
    })

    # -------------------------------------------------------------------------
    # 1. Comparative Energy Spectrum E(k) Coplot
    # -------------------------------------------------------------------------
    fig_spec, ax_spec = plt.subplots(figsize=(8, 6), dpi=300)
    has_spec_data = False

    for idx, sdir in enumerate(subdirs):
        cname = os.path.basename(sdir)
        label = CASE_LABELS.get(cname, cname)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        ls    = LINESTYLES[idx % len(LINESTYLES)]
        mk    = MARKERS[idx % len(MARKERS)]

        spec_csv = os.path.join(sdir, "csv_outputs", "energy_spectrum_2d.csv")
        if os.path.exists(spec_csv):
            try:
                data = np.genfromtxt(spec_csv, delimiter=",", skip_header=1)
                k_bins = data[:, 0]
                E_total = data[:, 1]
                
                # Filter non-zero E
                valid = (k_bins > 0) & (E_total > 0)
                if np.any(valid):
                    has_spec_data = True
                    k_valid = k_bins[valid]
                    E_valid = E_total[valid]
                    ax_spec.loglog(k_valid, E_valid, label=label, color=color,
                                  linestyle=ls, marker=mk, markevery=max(1, len(k_valid)//12))
            except Exception as e:
                print(f"[WARNING] Failed to load spectrum for {cname}: {e}")

    if has_spec_data:
        # Theoretical Power Law overlays
        try:
            k_ref = k_valid[(k_valid >= 3) & (k_valid <= max(4, len(k_valid)//3))]
            if len(k_ref) > 1:
                C_k3 = E_valid[k_valid == k_ref[0]][0] * (k_ref[0]**3)
                ax_spec.loglog(k_ref, C_k3 * (k_ref**(-3.0)), 'k:', linewidth=2, label=r'Kraichnan $k^{-3}$')
                C_k53 = E_valid[k_valid == k_ref[0]][0] * (k_ref[0]**(5/3))
                ax_spec.loglog(k_ref, C_k53 * (k_ref**(-5/3)), 'k--', linewidth=1.5, alpha=0.7, label=r'Kolmogorov $k^{-5/3}$')
        except Exception:
            pass

        ax_spec.set_xlabel(r'Wavenumber $k$', fontsize=12)
        ax_spec.set_ylabel(r'Energy Density $E(k)$', fontsize=12)
        ax_spec.set_title(f'Comparative Energy Spectrum ({problem_name.replace("_", " ").title()})', fontsize=13)
        ax_spec.grid(True, which="both", ls="--", alpha=0.4)
        ax_spec.legend(fontsize=10, loc='best', framealpha=0.95)

        spec_plot_path = os.path.join(summary_dir, "comparative_energy_spectrum.png")
        fig_spec.savefig(spec_plot_path, bbox_inches='tight')
        plt.close(fig_spec)
        print(f"[SAVED COPLOT] {spec_plot_path}", flush=True)
    else:
        plt.close(fig_spec)

    # -------------------------------------------------------------------------
    # 2. Comparative Kinetic Energy & Enstrophy History Coplots
    # -------------------------------------------------------------------------
    fig_hist, (ax_k, ax_om) = plt.subplots(2, 1, figsize=(8, 8), dpi=300, sharex=True)
    has_hist_data = False

    # First pass: find default case (or first valid case) to establish physical plot bounds
    default_k_limits = None
    default_om_limits = None

    for sdir in subdirs:
        cname = os.path.basename(sdir)
        enst_csv = os.path.join(sdir, "csv_outputs", "enstrophy_history.csv")
        if os.path.exists(enst_csv):
            try:
                data = np.genfromtxt(enst_csv, delimiter=",", skip_header=1)
                K_norm = data[:, 1] / data[0, 1]
                Omega_norm = data[:, 2] / data[0, 2]

                if cname == "default" or default_k_limits is None:
                    k_min, k_max = np.nanmin(K_norm), np.nanmax(K_norm)
                    om_min, om_max = np.nanmin(Omega_norm), np.nanmax(Omega_norm)
                    
                    # Add 25% buffer
                    k_margin = max(0.1, 0.25 * (k_max - k_min))
                    om_margin = max(0.2, 0.25 * (om_max - om_min))

                    default_k_limits = (max(0.0, k_min - k_margin), k_max + k_margin)
                    default_om_limits = (max(0.0, om_min - om_margin), om_max + om_margin)
            except Exception:
                pass

    for idx, sdir in enumerate(subdirs):
        cname = os.path.basename(sdir)
        label = CASE_LABELS.get(cname, cname)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        ls    = LINESTYLES[idx % len(LINESTYLES)]
        mk    = MARKERS[idx % len(MARKERS)]

        enst_csv = os.path.join(sdir, "csv_outputs", "enstrophy_history.csv")
        if os.path.exists(enst_csv):
            try:
                data = np.genfromtxt(enst_csv, delimiter=",", skip_header=1)
                t_arr = data[:, 0]
                K_arr = data[:, 1]
                Omega_arr = data[:, 2]

                has_hist_data = True
                mark_step = max(1, len(t_arr)//10)

                ax_k.plot(t_arr, K_arr / K_arr[0], label=label, color=color,
                          linestyle=ls, marker=mk, markevery=mark_step)
                ax_om.plot(t_arr, Omega_arr / Omega_arr[0], label=label, color=color,
                           linestyle=ls, marker=mk, markevery=mark_step)
            except Exception as e:
                print(f"[WARNING] Failed to load enstrophy for {cname}: {e}")

    if has_hist_data:
        ax_k.set_ylabel(r'Normalized Kinetic Energy $K(t)/K_0$', fontsize=11)
        ax_k.set_title(f'Comparative Decay History ({problem_name.replace("_", " ").title()})', fontsize=13)
        ax_k.grid(True, ls="--", alpha=0.4)
        ax_k.legend(fontsize=10, loc='best', framealpha=0.95)

        if default_k_limits:
            ax_k.set_ylim(default_k_limits)

        ax_om.set_xlabel(r'Time $t$', fontsize=11)
        ax_om.set_ylabel(r'Normalized Enstrophy $\Omega(t)/\Omega_0$', fontsize=11)
        ax_om.grid(True, ls="--", alpha=0.4)
        ax_om.legend(fontsize=10, loc='best', framealpha=0.95)

        if default_om_limits:
            ax_om.set_ylim(default_om_limits)

        hist_plot_path = os.path.join(summary_dir, "comparative_decay_history.png")
        fig_hist.savefig(hist_plot_path, bbox_inches='tight')
        plt.close(fig_hist)
        print(f"[SAVED COPLOT] {hist_plot_path}", flush=True)
    else:
        plt.close(fig_hist)


def main():
    problems = ["decaying_turbulence", "orszag_tang"]

    for prob in problems:
        pdir = os.path.join(CANONICAL_DIR, prob)
        if not os.path.exists(pdir):
            print(f"[ERROR] Problem directory does not exist: {pdir}")
            continue

        print(f"\n=======================================================")
        print(f"PROCESSING PROBLEM: {prob}")
        print(f"=======================================================")

        cases = sorted([d for d in glob.glob(os.path.join(pdir, "*"))
                        if os.path.isdir(d) and os.path.basename(d) not in ["summary_plots", "pv_outputs", "csv_outputs", "anims", "__pycache__"]])

        for cdir in cases:
            run_case_diagnostics(cdir, grid=256)

        generate_coplots(pdir)

    print("\n[COMPLETE] All diagnostics executed and comparative coplots generated!")


if __name__ == "__main__":
    main()

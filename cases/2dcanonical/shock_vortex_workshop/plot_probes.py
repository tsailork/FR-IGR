#!/usr/bin/env python3
"""
Python Line Probe Analysis & Plotting Script for Modern Shock-Vortex Interaction Benchmark.
AIAA High-Fidelity CFD Verification Workshop / HiOCFD Case CI2.

Reads probe CSVs at t = 0.7 for:
1. Longitudinal line: y = 0.4 (x in [0, 2.0])
2. Transverse post-shock line: x = 0.52 (y in [0, 1.0])
3. Transverse core line: x = 1.05 (y in [0, 1.0])

Outputs individual high-resolution plots and a combined multi-panel summary figure.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_probe_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[Warning] Probe file not found: {filepath}")
        return None
    data = np.genfromtxt(filepath, delimiter=',', names=True, skip_header=0)
    return data

def plot_all(d_y04, d_x052, d_x105, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    
    # Set plot styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    # --- Figure 1: Combined 3-Panel Summary ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # Panel 1: y = 0.4
    if d_y04 is not None:
        ax1 = axes[0]
        color1 = 'navy'
        ax1.plot(d_y04['x'], d_y04['rho'], color=color1, linewidth=1.8, label=r'Density $\rho$')
        ax1.set_xlabel(r'Position $x$')
        ax1.set_ylabel(r'Density $\rho$', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title(r'Longitudinal Probe ($y = 0.4, t = 0.7$)')
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax1_twin = ax1.twinx()
        color2 = 'crimson'
        ax1_twin.plot(d_y04['x'], d_y04['press'], color=color2, linewidth=1.4, linestyle='--', label=r'Pressure $p$')
        ax1_twin.set_ylabel(r'Pressure $p$', color=color2)
        ax1_twin.tick_params(axis='y', labelcolor=color2)

    # Panel 2: x = 0.52
    if d_x052 is not None:
        ax2 = axes[1]
        ax2.plot(d_x052['y'], d_x052['rho'], color='darkgreen', linewidth=1.8, label=r'Density $\rho$')
        ax2.set_xlabel(r'Position $y$')
        ax2.set_ylabel(r'Density $\rho$')
        ax2.set_title(r'Post-Shock Transverse Probe ($x = 0.52, t = 0.7$)')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right')

    # Panel 3: x = 1.05
    if d_x105 is not None:
        ax3 = axes[2]
        ax3.plot(d_x105['y'], d_x105['rho'], color='darkorange', linewidth=1.8, label=r'Density $\rho$')
        ax3.set_xlabel(r'Position $y$')
        ax3.set_ylabel(r'Density $\rho$')
        ax3.set_title(r'Vortex Core Transverse Probe ($x = 1.05, t = 0.7$)')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper right')

    plt.suptitle('AIAA High-Fidelity CFD Workshop: Shock-Vortex Interaction (Case CI2)', y=1.02, fontweight='bold')
    plt.tight_layout()
    summary_png = os.path.join(out_dir, "workshop_shock_vortex_summary.png")
    plt.savefig(summary_png, bbox_inches='tight')
    print(f"[Saved] {summary_png}")
    plt.close()

    # --- Individual Figures ---
    # 1. y = 0.4
    if d_y04 is not None:
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
        color = 'navy'
        ax1.set_xlabel('x-coordinate')
        ax1.set_ylabel('Density (ρ)', color=color)
        ax1.plot(d_y04['x'], d_y04['rho'], color=color, linewidth=1.8, label='FR-IGR Density')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        color = 'crimson'
        ax2.set_ylabel('Pressure (p)', color=color)
        ax2.plot(d_y04['x'], d_y04['press'], color=color, linewidth=1.5, linestyle='--', label='FR-IGR Pressure')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('AIAA Workshop Shock-Vortex: Line Probe y = 0.4 (t = 0.7)')
        fig.tight_layout()
        fn = os.path.join(out_dir, "probe_line_y04.png")
        plt.savefig(fn)
        print(f"[Saved] {fn}")
        plt.close()

    # 2. x = 0.52
    if d_x052 is not None:
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(d_x052['y'], d_x052['rho'], 'g-', linewidth=1.8, label='FR-IGR Density')
        plt.xlabel('y-coordinate')
        plt.ylabel('Density (ρ)')
        plt.title('AIAA Workshop Shock-Vortex: Probe x = 0.52 (t = 0.7)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(out_dir, "probe_line_x052.png")
        plt.savefig(fn)
        print(f"[Saved] {fn}")
        plt.close()

    # 3. x = 1.05
    if d_x105 is not None:
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(d_x105['y'], d_x105['rho'], 'r-', linewidth=1.8, label='FR-IGR Density')
        plt.xlabel('y-coordinate')
        plt.ylabel('Density (ρ)')
        plt.title('AIAA Workshop Shock-Vortex: Probe x = 1.05 (t = 0.7)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(out_dir, "probe_line_x105.png")
        plt.savefig(fn)
        print(f"[Saved] {fn}")
        plt.close()

def main():
    csv_dir = "csv_outputs"
    print("=== Shock-Vortex Interaction Line Probe Plotter ===")
    
    file_y04  = os.path.join(csv_dir, "probe_y04.csv") if os.path.exists(os.path.join(csv_dir, "probe_y04.csv")) else "probe_y04.csv"
    file_x052 = os.path.join(csv_dir, "probe_x052.csv") if os.path.exists(os.path.join(csv_dir, "probe_x052.csv")) else "probe_x052.csv"
    file_x105 = os.path.join(csv_dir, "probe_x105.csv") if os.path.exists(os.path.join(csv_dir, "probe_x105.csv")) else "probe_x105.csv"

    d_y04  = load_probe_csv(file_y04)
    d_x052 = load_probe_csv(file_x052)
    d_x105 = load_probe_csv(file_x105)

    plot_all(d_y04, d_x052, d_x105, out_dir=".")

if __name__ == "__main__":
    main()

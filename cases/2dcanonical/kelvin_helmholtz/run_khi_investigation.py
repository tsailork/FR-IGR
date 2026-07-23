#!/usr/bin/env python3
"""
Kelvin-Helmholtz Instability (KHI) Dedicated Study Automation Script
==============================================================================
Runs the full 16-case matrix for KHI investigating:
- 2 Methods: Navier-Stokes (NS) vs Adaptive PPR (PPR)
- 2 Polynomial Degrees: P=2 (3rd order) vs P=3 (4th order)
- 2 Resolutions: 1x (64x64) vs 2x (128x128)
- 2 Reynolds Numbers: Normal Re (10000) vs Double Re (20000)

After executing simulations, runs the Python diagnostic suite (spectra, enstrophy,
PPR localization) and generates publication-quality comparison overlay plots.
"""

import os
import sys
import shutil
import glob
import subprocess
import time

CANONICAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR   = os.path.dirname(os.path.dirname(CANONICAL_DIR))
BIN_SOLVER    = os.path.join(PROJECT_DIR, "bin", "fr_solver")
PYTHON_UTILS  = os.path.join(PROJECT_DIR, "python_utilities")
KHI_DIR       = os.path.join(CANONICAL_DIR, "kelvin_helmholtz")
DEFAULT_DIR   = os.path.join(KHI_DIR, "default")

METHODS = [
    ("navier_stokes", {"ENABLE_NS": "true", "ENABLE_PPR": "false", "ENABLE_IGR": "false", "ENABLE_POS_LIMITER": "true", "ENABLE_ENTROPY_LIMITER": "false"}),
    ("ppr_adaptive",  {"ENABLE_NS": "true", "ENABLE_PPR": "true",  "ENABLE_IGR": "false", "ENABLE_POS_LIMITER": "true", "ENABLE_ENTROPY_LIMITER": "false"})
]

P_DEGS      = [2, 3]
RESOLUTIONS = [(64, 64, "1x"), (128, 128, "2x")]
REYNOLDS    = [10000.0, 20000.0]


def parse_ini(file_path):
    kv = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()
    return kv


def update_ini(file_path, target_key, new_value):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    found = False
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            k, _ = stripped.split("=", 1)
            if k.strip() == target_key:
                new_lines.append(f"{target_key} = {new_value}\n")
                found = True
                continue
        new_lines.append(line)
    if not found:
        new_lines.append(f"{target_key} = {new_value}\n")
    with open(file_path, "w") as f:
        f.writelines(new_lines)


def build_solver():
    print("[BUILD] Compiling C++ FR-IGR Solver...", flush=True)
    res = subprocess.run(["make", "-C", PROJECT_DIR, "-j12", "all"], capture_output=True, text=True)
    if res.returncode == 0:
        print("[BUILD] Solver executable is ready.", flush=True)
        return True
    else:
        print(f"[BUILD ERROR] {res.stderr}", flush=True)
        return False


def setup_case(method_name, custom_params, nx, ny, p_deg, re_val):
    folder_name = f"case_{method_name}_{nx}x{ny}_P{p_deg}_Re{int(re_val)}"
    run_path = os.path.join(KHI_DIR, folder_name)

    os.makedirs(os.path.join(run_path, "pv_outputs"), exist_ok=True)
    os.makedirs(os.path.join(run_path, "csv_outputs"), exist_ok=True)

    shutil.copy(os.path.join(DEFAULT_DIR, "domain.grid"), os.path.join(run_path, "domain.grid"))
    shutil.copy(os.path.join(DEFAULT_DIR, "inputs.dat"),  os.path.join(run_path, "inputs.dat"))

    # Update grid resolution
    update_ini(os.path.join(run_path, "domain.grid"), "N_ELEM_X", str(nx))
    update_ini(os.path.join(run_path, "domain.grid"), "N_ELEM_Y", str(ny))

    # Update inputs.dat
    inp_file = os.path.join(run_path, "inputs.dat")
    update_ini(inp_file, "P_DEG", str(p_deg))
    update_ini(inp_file, "RE", str(re_val))
    update_ini(inp_file, "CFL", "0.6")
    update_ini(inp_file, "T_FINAL", "5.0")
    update_ini(inp_file, "OUTPUT_INTERVAL", "0.10")
    update_ini(inp_file, "NUM_THREADS", "16")

    for k, v in custom_params.items():
        update_ini(inp_file, k, str(v))

    return run_path


def run_single_case(rpath, num_threads=16):
    pv_count = len(glob.glob(os.path.join(rpath, "pv_outputs", "*.vtu")))
    csv_file = os.path.join(rpath, "csv_outputs", "enstrophy_history.csv")

    if pv_count >= 50 and os.path.exists(csv_file):
        print(f"[SKIP RUN & DIAGNOSTICS] {os.path.basename(rpath)} already completed with {pv_count} datasets.", flush=True)
        return rpath

    print(f"\n[RUNNING] {os.path.basename(rpath)} with {num_threads} threads...", flush=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    inp_file = os.path.join(rpath, "inputs.dat")
    update_ini(inp_file, "NUM_THREADS", str(num_threads))

    log_path = os.path.join(rpath, "out.log")
    with open(log_path, "w") as log_f:
        p = subprocess.Popen([BIN_SOLVER], cwd=rpath, stdout=log_f, stderr=log_f, env=env)
        p.wait()

    pv_count = len(glob.glob(os.path.join(rpath, "pv_outputs", "*.vtu")))
    print(f"[RUN COMPLETE] {os.path.basename(rpath)} produced {pv_count} datasets.", flush=True)

    run_diagnostics(rpath)
    return rpath


def run_diagnostics(run_path):
    print(f"[DIAGNOSTICS] Analyzing {os.path.basename(run_path)}...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHON_UTILS
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.run(["python3", "-u", os.path.join(PYTHON_UTILS, "spectrum_2d.py"), "--case", run_path, "--grid", "256"], env=env)
    subprocess.run(["python3", "-u", os.path.join(PYTHON_UTILS, "enstrophy.py"), "--case", run_path, "--grid", "256"], env=env)
    if "ppr" in run_path:
        subprocess.run(["python3", "-u", os.path.join(PYTHON_UTILS, "ppr_localization.py"), "--case", run_path, "--grid", "256"], env=env)


def generate_comparisons():
    print("\n[PLOTTING] Generating publication comparison overlay plots...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHON_UTILS
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(["python3", "-u", os.path.join(PYTHON_UTILS, "comparative_plots.py"), "--case_dir", KHI_DIR], env=env)


def main():
    if not build_solver():
        sys.exit(1)

    all_runs = []
    for method_name, custom_params in METHODS:
        for p_deg in P_DEGS:
            for nx, ny, res_tag in RESOLUTIONS:
                for re_val in REYNOLDS:
                    rpath = setup_case(method_name, custom_params, nx, ny, p_deg, re_val)
                    all_runs.append(rpath)

    print(f"\n[MATRIX READY] Created {len(all_runs)} KHI investigation cases.", flush=True)

    for rpath in all_runs:
        run_single_case(rpath, num_threads=16)

    generate_comparisons()
    print("\n[INVESTIGATION COMPLETE] All 16 cases simulated, analyzed, and plotted successfully!", flush=True)


if __name__ == "__main__":
    main()

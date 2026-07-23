#!/usr/bin/env python3
"""
FR-IGR 2D Canonical Turbulence Study Manager (TUI Application)
==============================================================================
Interactive research dashboard for configuring, managing, executing, and analyzing
2D canonical turbulence and shock-capturing benchmark test cases.

Key Features:
- Restructured case subfolders: default/ (template) alongside run_<scheme>_<res>_P<deg>/
- Selective execution, interactive case construction wizard ([H])
- Default config propagation to study subcases with automatic output cleaning ([D])
- Python diagnostic processing ([P]) & publication-quality comparative plotting ([V])
- Defensive safety guards protecting template folders and grid connectivities
- Clean ANSI formatting, status badges, resolution metadata, and terminal state preservation
"""

import sys
import os
import glob
import re
import time
import subprocess
import select
import termios
import tty
import shutil
from datetime import datetime

# Path calculations
CANONICAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(os.path.dirname(CANONICAL_DIR))
BIN_SOLVER    = os.path.join(PROJECT_DIR, "bin", "fr_solver")
PYTHON_UTILS  = os.path.join(PROJECT_DIR, "python_utilities")

# ANSI Terminal Formatting
CLR_RESET   = "\033[0m"
CLR_BOLD    = "\033[1m"
CLR_RED     = "\033[31m"
CLR_GREEN   = "\033[32m"
CLR_YELLOW  = "\033[33m"
CLR_BLUE    = "\033[34m"
CLR_MAGENTA = "\033[35m"
CLR_CYAN    = "\033[36m"
CLR_WHITE   = "\033[37m"
CLR_GRAY    = "\033[90m"

BG_BLUE     = "\033[44m"
BG_DARK     = "\033[40m"

CANONICAL_CASES = [
    "kelvin_helmholtz",
    "shock_vortex",
    "decaying_turbulence",
    "richtmyer_meshkov",
    "orszag_tang"
]

SCHEMES = [
    ("ppr_adaptive",      "PPR Adaptive (theta)",  True,  False, True,  False, False),
    ("navier_stokes",     "Navier-Stokes (NS)",    False, False, True,  False, False),
    ("no_regularization", "Un-regularized Euler",   False, False, False, False, False),
    ("igr_parabolic",    "IGR Parabolic",         False, True,  True,  False, False)
]


class NonBlockingTerminal:
    """Context manager for non-blocking TTY character input."""
    def __enter__(self):
        self.is_tty = sys.stdin.isatty()
        if self.is_tty:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        if self.is_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_char(self):
        if self.is_tty and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def parse_ini_file(file_path):
    """Parses a standard INI configuration file into a key-value dictionary."""
    kv = {}
    if not os.path.exists(file_path):
        return kv
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                kv[key.strip()] = val.strip()
    return kv


def update_ini_parameter(file_path, target_key, new_value):
    """Updates or appends a key-value parameter in an INI configuration file."""
    if not os.path.exists(file_path):
        return
    lines = []
    found = False
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#") and not stripped.startswith(";"):
            key, _ = stripped.split("=", 1)
            if key.strip() == target_key:
                new_lines.append(f"{target_key} = {new_value}\n")
                found = True
                continue
        new_lines.append(line)

    if not found:
        new_lines.append(f"{target_key} = {new_value}\n")

    with open(file_path, "w") as f:
        f.writelines(new_lines)


def get_case_variants(case_name):
    """Scans a case folder and lists all study variant run directories (run_*)."""
    case_path = os.path.join(CANONICAL_DIR, case_name)
    if not os.path.exists(case_path):
        return []
    
    entries = os.listdir(case_path)
    run_dirs = [e for e in entries if os.path.isdir(os.path.join(case_path, e)) and e.startswith("run_")]
    run_dirs.sort()
    return run_dirs


def get_run_status(run_path):
    """Determines the execution status and dataset count of a study variant directory."""
    pv_dir = os.path.join(run_path, "pv_outputs")
    log_file = os.path.join(run_path, "out.log")

    dataset_count = 0
    if os.path.exists(pv_dir):
        dataset_count = len(glob.glob(os.path.join(pv_dir, "*.vtu")))

    if not os.path.exists(log_file):
        return "NOT RUN", CLR_GRAY, dataset_count

    with open(log_file, "r") as f:
        content = f.read()

    if "Simulation complete." in content:
        return "COMPLETED", CLR_BOLD + CLR_GREEN, dataset_count
    elif "nonphysical" in content.lower() or "fatal" in content.lower() or "error" in content.lower():
        return "CRASHED", CLR_BOLD + CLR_RED, dataset_count
    elif dataset_count > 0:
        return "PARTIAL", CLR_YELLOW, dataset_count
    else:
        return "INITIALIZED", CLR_CYAN, dataset_count


def get_run_resolution_str(run_path):
    """Retrieves resolution and P_DEG string (e.g., '64x64 P2') for a run directory."""
    grid_file = os.path.join(run_path, "domain.grid")
    inp_file  = os.path.join(run_path, "inputs.dat")

    grid_kv = parse_ini_file(grid_file)
    inp_kv  = parse_ini_file(inp_file)

    nx  = grid_kv.get("N_ELEM_X", "?")
    ny  = grid_kv.get("N_ELEM_Y", "?")
    deg = inp_kv.get("P_DEG", "?")
    return f"{nx}x{ny} P{deg}"


def clean_run_directory(run_path, delete_folder=False):
    """
    Cleans output datasets from a study variant folder or deletes the folder entirely.
    SAFETY GUARD: Explicitly prevents modifying/deleting the 'default/' template folder.
    """
    if os.path.basename(run_path) == "default":
        print(f"{CLR_RED}[SAFETY GUARD] Refusing to clean/delete protected template folder 'default/'!{CLR_RESET}")
        return

    if delete_folder:
        shutil.rmtree(run_path, ignore_errors=True)
        print(f"{CLR_YELLOW}[CLEAN] Deleted study directory: {os.path.basename(run_path)}{CLR_RESET}")
    else:
        pv_dir = os.path.join(run_path, "pv_outputs")
        csv_dir = os.path.join(run_path, "csv_outputs")
        log_file = os.path.join(run_path, "out.log")
        stop_file = os.path.join(run_path, "STOP")
        res_file = os.path.join(run_path, "residuals.dat")

        if os.path.exists(pv_dir):
            for f in glob.glob(os.path.join(pv_dir, "*")): os.remove(f)
        if os.path.exists(csv_dir):
            for f in glob.glob(os.path.join(csv_dir, "*")): os.remove(f)
        if os.path.exists(log_file): os.remove(log_file)
        if os.path.exists(stop_file): os.remove(stop_file)
        if os.path.exists(res_file): os.remove(res_file)

        print(f"{CLR_GREEN}[CLEAN] Reset outputs for: {os.path.basename(run_path)}{CLR_RESET}")


def propagate_default_inputs(case_name):
    """
    Propagates changes in default/inputs.dat to all existing study variants for a case.
    Triggers a clean of sub-case dataset outputs while preserving sub-case directory structures.
    """
    default_dir = os.path.join(CANONICAL_DIR, case_name, "default")
    default_inputs = os.path.join(default_dir, "inputs.dat")

    if not os.path.exists(default_inputs):
        print(f"{CLR_RED}Default inputs.dat missing in {default_dir}{CLR_RESET}")
        return

    def_kv = parse_ini_file(default_inputs)
    variants = get_case_variants(case_name)

    print(f"\n{CLR_CYAN}[PROPAGATE] Propagating default inputs to {len(variants)} study variants in '{case_name}'...{CLR_RESET}")

    for vname in variants:
        vpath = os.path.join(CANONICAL_DIR, case_name, vname)
        v_inputs = os.path.join(vpath, "inputs.dat")

        # Update key physics/solver parameters while maintaining variant scheme toggles
        for key in ["T_FINAL", "CFL", "GAMMA", "RE", "PR", "NUM_THREADS", "PRINT_INTERVAL", "OUTPUT_INTERVAL"]:
            if key in def_kv:
                update_ini_parameter(v_inputs, key, def_kv[key])

        # Clean output datasets for subcase without deleting directory
        clean_run_directory(vpath, delete_folder=False)

    print(f"{CLR_GREEN}[PROPAGATE] Successfully updated & cleaned {len(variants)} subcase directories.{CLR_RESET}")


def ensure_solver_built():
    """Compiles the C++ solver executable if missing or out-of-date."""
    print(f"{CLR_CYAN}[BUILD] Compiling C++ FR-IGR Solver...{CLR_RESET}")
    res = subprocess.run(["make", "-C", PROJECT_DIR, "-j12", "all"], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"{CLR_GREEN}[BUILD] Solver executable is up to date: {BIN_SOLVER}{CLR_RESET}")
        return True
    else:
        print(f"{CLR_RED}[BUILD FATAL] Solver build failed:\n{res.stderr}{CLR_RESET}")
        return False


def setup_variant_folder(case_name, scheme_id, nx=None, ny=None, p_deg=None, is_ref=False, custom_params=None):
    """
    Creates and initializes a specific study variant run folder from default templates.
    Enforces strict grid scaling and naming conventions.
    """
    default_dir = os.path.join(CANONICAL_DIR, case_name, "default")
    def_grid = os.path.join(default_dir, "domain.grid")
    def_inp  = os.path.join(default_dir, "inputs.dat")

    if not os.path.exists(def_grid) or not os.path.exists(def_inp):
        print(f"{CLR_RED}[ERROR] Default template files missing in {default_dir}{CLR_RESET}")
        return None

    def_grid_kv = parse_ini_file(def_grid)
    def_inp_kv  = parse_ini_file(def_inp)

    # Determine resolution and polynomial degree
    res_x = nx if nx is not None else int(def_grid_kv.get("N_ELEM_X", 64))
    res_y = ny if ny is not None else int(def_grid_kv.get("N_ELEM_Y", 64))
    deg   = p_deg if p_deg is not None else int(def_inp_kv.get("P_DEG", 2))

    if is_ref:
        folder_name = f"run_ref_pos_entropy_{res_x}x{res_y}_P{deg}"
    else:
        folder_name = f"run_{scheme_id}_{res_x}x{res_y}_P{deg}"

    run_path = os.path.join(CANONICAL_DIR, case_name, folder_name)
    os.makedirs(os.path.join(run_path, "pv_outputs"), exist_ok=True)
    os.makedirs(os.path.join(run_path, "csv_outputs"), exist_ok=True)

    # Copy baseline templates
    shutil.copy(def_grid, os.path.join(run_path, "domain.grid"))
    shutil.copy(def_inp,  os.path.join(run_path, "inputs.dat"))

    # Update resolution in domain.grid
    update_ini_parameter(os.path.join(run_path, "domain.grid"), "N_ELEM_X", str(res_x))
    update_ini_parameter(os.path.join(run_path, "domain.grid"), "N_ELEM_Y", str(res_y))

    # Update scheme toggles in inputs.dat
    inp_file = os.path.join(run_path, "inputs.dat")
    update_ini_parameter(inp_file, "P_DEG", str(deg))

    if custom_params:
        for k, v in custom_params.items():
            update_ini_parameter(inp_file, k, str(v))
    elif is_ref:
        update_ini_parameter(inp_file, "ENABLE_PPR", "false")
        update_ini_parameter(inp_file, "ENABLE_IGR", "false")
        update_ini_parameter(inp_file, "ENABLE_NS",  "true")
        update_ini_parameter(inp_file, "ENABLE_POS_LIMITER", "true")
        update_ini_parameter(inp_file, "ENABLE_ENTROPY_LIMITER", "true")
    else:
        # Match scheme specification
        for s_id, s_name, ppr, igr, ns, pos, ent in SCHEMES:
            if s_id == scheme_id:
                update_ini_parameter(inp_file, "ENABLE_PPR", "true" if ppr else "false")
                update_ini_parameter(inp_file, "ENABLE_IGR", "true" if igr else "false")
                update_ini_parameter(inp_file, "ENABLE_NS",  "true" if ns else "false")
                break

    return run_path


def execute_variant_run(run_path, threads=12):
    """Executes C++ time integration for a specific variant run folder."""
    if not ensure_solver_built():
        return False

    vname = os.path.basename(run_path)
    print(f"\n{CLR_CYAN}[RUNNING] Executing simulation for: {vname} (Threads: {threads})...{CLR_RESET}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    log_path = os.path.join(run_path, "out.log")
    with open(log_path, "w") as log_f:
        p = subprocess.Popen([BIN_SOLVER], cwd=run_path, stdout=log_f, stderr=log_f, env=env)
        p.wait()

    status, clr, count = get_run_status(run_path)
    print(f"{clr}[RUN COMPLETE] Status: {status} | Outputs: {count} datasets in pv_outputs/{CLR_RESET}")
    return status == "COMPLETED"


def run_python_diagnostics(run_path, grid_res=256):
    """Triggers the Python diagnostic processing suite on a completed variant run."""
    vname = os.path.basename(run_path)
    print(f"\n{CLR_MAGENTA}[DIAGNOSTICS] Processing Python analytics for: {vname}...{CLR_RESET}")

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHON_UTILS

    # 1. 2D Spectrum & Helmholtz Decomposition
    subprocess.run(["python3", os.path.join(PYTHON_UTILS, "spectrum_2d.py"), "--case", run_path, "--grid", str(grid_res)], env=env)
    
    # 2. Enstrophy History
    subprocess.run(["python3", os.path.join(PYTHON_UTILS, "enstrophy.py"), "--case", run_path, "--grid", str(grid_res)], env=env)

    # 3. Vortex Stats if applicable
    if "shock_vortex" in run_path:
        subprocess.run(["python3", os.path.join(PYTHON_UTILS, "vortex_stats.py"), "--case", run_path, "--grid", str(grid_res)], env=env)

    # 4. PPR Spatial Localization
    if "ppr" in run_path or "igr" in run_path:
        subprocess.run(["python3", os.path.join(PYTHON_UTILS, "ppr_localization.py"), "--case", run_path, "--grid", str(grid_res)], env=env)

    print(f"{CLR_GREEN}[DIAGNOSTICS COMPLETE] Analytics generated in csv_outputs/{CLR_RESET}")


def launch_system_editor(file_path, old_settings=None):
    """
    Launches system $EDITOR (defaulting to vim) in-place.
    Restores normal cooked terminal settings before spawning the editor,
    and re-enables non-blocking cbreak mode upon exit to prevent crashes.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 1. Restore normal cooked terminal settings
    if old_settings is not None and sys.stdin.isatty():
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    # 2. Clear terminal screen
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

    # 3. Choose the text editor (default to vim)
    editor = os.environ.get("EDITOR", "vim")
    print(f"Opening {os.path.basename(file_path)} in {editor}...")
    time.sleep(0.3)

    try:
        subprocess.run([editor, file_path])
    except Exception as e:
        print(f"Error launching editor '{editor}': {e}")
        fallback_editors = ["vim", "nano", "vi"] if editor != "vim" else ["nano", "vi"]
        for ed in fallback_editors:
            try:
                print(f"Falling back to {ed}...")
                subprocess.run([ed, file_path])
                break
            except Exception:
                continue

    # 4. Re-enable non-blocking cbreak mode
    if sys.stdin.isatty():
        try:
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            pass

    # 5. Clear screen again to cleanly redraw the TUI
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


class StudyManagerTUI:
    def __init__(self):
        self.selected_cases = {case: True for case in CANONICAL_CASES}
        self.active_message = ""

    def render_dashboard(self):
        """Renders the main interactive TUI dashboard with strict column widths and status badges."""
        os.system("clear")
        print(f"{CLR_BOLD}{BG_BLUE}{CLR_WHITE} FR-IGR 2D CANONICAL TURBULENCE STUDY MANAGER {CLR_RESET}")
        print(f"{CLR_GRAY}Location: {CANONICAL_DIR}{CLR_RESET}")
        
        solver_status = f"{CLR_GREEN}READY{CLR_RESET}" if os.path.exists(BIN_SOLVER) else f"{CLR_RED}NOT BUILT{CLR_RESET}"
        sel_count = sum(1 for c in self.selected_cases.values() if c)
        print(f"C++ Solver Executable: {solver_status} | Selected Cases: {CLR_CYAN}{sel_count}/{len(CANONICAL_CASES)}{CLR_RESET}\n")

        print(f"{CLR_BOLD}{'SEL':<4} {'CANONICAL CASE':<22} {'VARIANT RUN DIRECTORIES':<36} {'RES/P':<10} {'STATUS':<12} {'DATASETS':<10}{CLR_RESET}")
        print("─" * 94)

        for idx, case in enumerate(CANONICAL_CASES, 1):
            is_sel = "[x]" if self.selected_cases[case] else "[ ]"
            sel_clr = CLR_GREEN if self.selected_cases[case] else CLR_GRAY
            
            variants = get_case_variants(case)
            if not variants:
                print(f"{sel_clr}[{idx}] {is_sel} {case:<20} {CLR_GRAY}(No run variants set up){CLR_RESET}")
            else:
                max_show = 5
                show_variants = variants[:max_show]
                for v_i, vname in enumerate(show_variants):
                    vpath = os.path.join(CANONICAL_DIR, case, vname)
                    status, st_clr, count = get_run_status(vpath)
                    res_str = get_run_resolution_str(vpath)
                    
                    if v_i == 0:
                        prefix = f"{sel_clr}[{idx}] {is_sel} {case:<20}{CLR_RESET}"
                    else:
                        prefix = f"    {'':<22}"
                    
                    print(f"{prefix} {vname:<36} {res_str:<10} {st_clr}{status:<12}{CLR_RESET} {count:<10}")

                if len(variants) > max_show:
                    print(f"    {'':<22} {CLR_GRAY}... +{len(variants) - max_show} more variant directories ...{CLR_RESET}")

        print("─" * 94)
        print(f"{CLR_BOLD}KEYBINDINGS:{CLR_RESET}")
        print(f" [{CLR_CYAN}1-5{CLR_RESET}] Toggle Case        [{CLR_CYAN}A{CLR_RESET}] Select/Deselect All  [{CLR_CYAN}R{CLR_RESET}] Run Selected Variants")
        print(f" [{CLR_CYAN}D{CLR_RESET}] Propagate Defaults [{CLR_CYAN}H{CLR_RESET}] Construct Case       [{CLR_CYAN}C{CLR_RESET}] Clean / Delete")
        print(f" [{CLR_CYAN}P{CLR_RESET}] Diagnostic Analytics[{CLR_CYAN}V{CLR_RESET}] Comparative Plots   [{CLR_CYAN}E{CLR_RESET}] Edit inputs.dat")
        print(f" [{CLR_CYAN}S{CLR_RESET}] Run All Sweeps      [{CLR_CYAN}M{CLR_RESET}] User Manual          [{CLR_CYAN}Q{CLR_RESET}] Quit TUI")
        print("─" * 94)

        if self.active_message:
            print(f"{CLR_YELLOW}{self.active_message}{CLR_RESET}\n")
            self.active_message = ""

    def run_loop(self):
        """Main non-blocking event loop."""
        with NonBlockingTerminal() as term:
            while True:
                self.render_dashboard()
                print(f"{CLR_BOLD}Enter Option > {CLR_RESET}", end="", flush=True)

                ch = None
                while ch is None:
                    ch = term.get_char()
                    time.sleep(0.05)

                ch_upper = ch.upper()

                if ch in ['1', '2', '3', '4', '5']:
                    case_name = CANONICAL_CASES[int(ch) - 1]
                    self.selected_cases[case_name] = not self.selected_cases[case_name]

                elif ch_upper == 'A':
                    all_val = not all(self.selected_cases.values())
                    for c in CANONICAL_CASES:
                        self.selected_cases[c] = all_val

                elif ch_upper == 'D':
                    # Propagate default inputs to subcases
                    for case, sel in self.selected_cases.items():
                        if sel:
                            propagate_default_inputs(case)
                    time.sleep(1.5)

                elif ch_upper == 'R':
                    # Prepare & Run selected variants
                    ensure_solver_built()
                    for case, sel in self.selected_cases.items():
                        if sel:
                            for s_id, _, _, _, _, _, _ in SCHEMES:
                                rpath = setup_variant_folder(case, s_id)
                                if rpath:
                                    execute_variant_run(rpath)
                    time.sleep(1.5)

                elif ch_upper == 'H':
                    # Interactive Case Construction
                    print(f"\n{CLR_BOLD}{CLR_CYAN}=== CONSTRUCT CASE VARIANT (Navier-Stokes) ==={CLR_RESET}")
                    print("Select Method Option:")
                    print(f" [{CLR_CYAN}1{CLR_RESET}] Positivity Only (default POS_LIMITER_EPS, PPR=false)")
                    print(f" [{CLR_CYAN}2{CLR_RESET}] Positivity + Entropy Limiter (ENTROPY_LIMITER_EPS=1e-3, PPR=false)")
                    print(f" [{CLR_CYAN}3{CLR_RESET}] Positivity + PPR (default PPR case setup)")

                    m_choice = input("\nEnter Method Choice (1-3) > ").strip()
                    if m_choice not in ['1', '2', '3']:
                        print(f"{CLR_RED}Invalid method choice.{CLR_RESET}")
                        time.sleep(1.0)
                        continue

                    try:
                        res_fac_str = input("Enter Resolution Multiplier Factor on Nx, Ny (e.g., 1, 2, 4, 10) [Default: 2] > ").strip()
                        res_fac = float(res_fac_str) if res_fac_str else 2.0
                    except ValueError:
                        res_fac = 2.0

                    if m_choice == '1':
                        tag = "positivity"
                        c_params = {
                            "ENABLE_NS": "true",
                            "ENABLE_PPR": "false",
                            "ENABLE_IGR": "false",
                            "ENABLE_POS_LIMITER": "true",
                            "ENABLE_ENTROPY_LIMITER": "false"
                        }
                    elif m_choice == '2':
                        tag = "positivity_entropy"
                        c_params = {
                            "ENABLE_NS": "true",
                            "ENABLE_PPR": "false",
                            "ENABLE_IGR": "false",
                            "ENABLE_POS_LIMITER": "true",
                            "ENABLE_ENTROPY_LIMITER": "true",
                            "ENTROPY_LIMITER_EPS": "1e-3"
                        }
                    else: # '3'
                        tag = "ppr_adaptive"
                        c_params = {
                            "ENABLE_NS": "true",
                            "ENABLE_PPR": "true",
                            "ENABLE_IGR": "false",
                            "ENABLE_POS_LIMITER": "true",
                            "ENABLE_ENTROPY_LIMITER": "false"
                        }

                    for case, sel in self.selected_cases.items():
                        if sel:
                            def_grid = parse_ini_file(os.path.join(CANONICAL_DIR, case, "default", "domain.grid"))
                            nx = int(round(int(def_grid.get("N_ELEM_X", 64)) * res_fac))
                            ny = int(round(int(def_grid.get("N_ELEM_Y", 64)) * res_fac))
                            
                            rpath = setup_variant_folder(case, tag, nx=nx, ny=ny, custom_params=c_params)
                            if rpath:
                                print(f"{CLR_GREEN}[CONSTRUCT] Created case variant: {os.path.basename(rpath)} (Ready to run via [R]){CLR_RESET}")
                    time.sleep(1.2)

                elif ch_upper == 'C':
                    # Clean Menu
                    print(f"\n{CLR_YELLOW}CLEAN OPTIONS:{CLR_RESET} [1] Clean Dataset Outputs  [2] Delete Variant Folders  [Cancel: Any other key]")
                    opt = input("Select Clean Mode > ").strip()
                    if opt == '1':
                        for case, sel in self.selected_cases.items():
                            if sel:
                                for vname in get_case_variants(case):
                                    clean_run_directory(os.path.join(CANONICAL_DIR, case, vname), delete_folder=False)
                    elif opt == '2':
                        for case, sel in self.selected_cases.items():
                            if sel:
                                for vname in get_case_variants(case):
                                    clean_run_directory(os.path.join(CANONICAL_DIR, case, vname), delete_folder=True)
                    time.sleep(1.5)

                elif ch_upper == 'P':
                    # Run Python Analytics
                    for case, sel in self.selected_cases.items():
                        if sel:
                            for vname in get_case_variants(case):
                                run_python_diagnostics(os.path.join(CANONICAL_DIR, case, vname))
                    time.sleep(1.5)

                elif ch_upper == 'V':
                    # Run Comparative Plotting
                    print(f"\n{CLR_MAGENTA}[PLOT] Generating Multi-Scheme Comparative Plots...{CLR_RESET}")
                    env = os.environ.copy()
                    env["PYTHONPATH"] = PYTHON_UTILS
                    for case, sel in self.selected_cases.items():
                        if sel:
                            cpath = os.path.join(CANONICAL_DIR, case)
                            subprocess.run(["python3", os.path.join(PYTHON_UTILS, "comparative_plots.py"), "--case_dir", cpath], env=env)
                    time.sleep(1.5)

                elif ch_upper == 'E':
                    # Edit default inputs.dat for selected case
                    sel_cases = [c for c, sel in self.selected_cases.items() if sel]
                    if sel_cases:
                        target_case = sel_cases[0]
                        def_inp = os.path.join(CANONICAL_DIR, target_case, "default", "inputs.dat")
                        old_st = term.old_settings if hasattr(term, 'old_settings') else None
                        launch_system_editor(def_inp, old_settings=old_st)

                elif ch_upper == 'S':
                    # Run All Sweeps
                    ensure_solver_built()
                    for case in CANONICAL_CASES:
                        for s_id, _, _, _, _, _, _ in SCHEMES:
                            rpath = setup_variant_folder(case, s_id)
                            if rpath:
                                execute_variant_run(rpath)
                                run_python_diagnostics(rpath)
                    time.sleep(1.5)

                elif ch_upper == 'M':
                    # Open User Manual
                    manual_path = os.path.join(CANONICAL_DIR, "TUI_USER_MANUAL.md")
                    if os.path.exists(manual_path):
                        old_st = term.old_settings if hasattr(term, 'old_settings') else None
                        launch_system_editor(manual_path, old_settings=old_st)

                elif ch_upper == 'Q':
                    print(f"\n{CLR_GREEN}Exiting TUI Study Manager. Goodbye!{CLR_RESET}")
                    sys.exit(0)


if __name__ == "__main__":
    tui = StudyManagerTUI()
    tui.run_loop()

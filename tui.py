#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import re
import select
import termios
import tty
import signal
from datetime import datetime

# ANSI Colors
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
BG_BLACK    = "\033[40m"

# Regex patterns for parsing log
step_pattern = re.compile(
    r"\[Step\s+(\d+)\]\s+t:\s+([\d\.]+)\s+\|\s+([\d\.]+)%\s+\|\s+ETA:\s+([\d\.-]+)s\s+\|\s+L2_Sum:\s+([0-9eE\.\+-]+)(?:\s+\|\s+Lim:\s+(\d+)\s+\(avg_th:\s+([\d\.]+)\))?"
)
warning_pattern = re.compile(r"warning|\[warning\]", re.IGNORECASE)
sbm_diag_pattern = re.compile(
    r"\[SBM DIAGNOSTICS\]\s+Max\s+Lebesgue:\s+([0-9eE\.\+-]+)\s+\|\s+Limiter\s+Triggers:\s+(\d+)\s+\|\s+Max\s+D/L:\s+([0-9eE\.\+-]+)\s+\|\s+Max\s+D/dL:\s+([0-9eE\.\+-]+)"
)

class NonBlockingInput:
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

def parse_inputs_dat():
    params = {}
    if not os.path.exists("inputs.dat"):
        return params
    try:
        with open("inputs.dat", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    params[key.strip()] = val.strip()
    except Exception:
        pass
    return params

def visual_len(s):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))

def pad_ansi(text, width, align='left'):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    vis_len = len(ansi_escape.sub('', text))
    if vis_len >= width:
        return text
    pad_len = width - vis_len
    if align == 'left':
        return text + ' ' * pad_len
    elif align == 'right':
        return ' ' * pad_len + text
    else: # center
        left_pad = pad_len // 2
        right_pad = pad_len - left_pad
        return ' ' * left_pad + text + ' ' * right_pad

def make_panel(title, content_lines, width=78):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    border_top = f"┌─ {CLR_BOLD}{CLR_CYAN}{title}{CLR_RESET} " + "─" * (width - len(title) - 5) + "┐"
    border_bottom = "└" + "─" * (width - 2) + "┘"
    
    panel_lines = [border_top]
    for line in content_lines:
        clean_line = ansi_escape.sub('', line)
        vis_len = len(clean_line)
        
        if vis_len < width - 4:
            padded_line = line + " " * (width - 4 - vis_len)
        else:
            padded_line = clean_line[:width - 4]
                
        panel_lines.append(f"│ {padded_line} │")
    panel_lines.append(border_bottom)
    return panel_lines

def is_pid_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def find_running_solver_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "fr_solver"]).decode().strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        pids = [p for p in pids if p != os.getpid()]
        if pids:
            return pids[0]
    except Exception:
        pass
    return None

def find_latest_checkpoint_from_pvd():
    pvd_path = "pv_outputs/solution.pvd"
    if not os.path.exists(pvd_path):
        return None, 0.0
    
    latest_file = None
    latest_time = 0.0
    
    try:
        dataset_pattern = re.compile(r'<DataSet\s+timestep="([0-9eE\.\+-]+)"[^>]*file="([^"]+)"')
        with open(pvd_path, "r") as f:
            for line in f:
                match = dataset_pattern.search(line)
                if match:
                    t_val = float(match.group(1))
                    f_val = match.group(2)
                    if t_val >= latest_time:
                        latest_time = t_val
                        latest_file = f_val
    except Exception:
        pass
        
    if latest_file:
        if not latest_file.startswith("pv_outputs/"):
            latest_file = "pv_outputs/" + latest_file
            
    return latest_file, latest_time

def update_restart_in_inputs_dat(restart_file, restart_time):
    if not os.path.exists("inputs.dat"):
        return False
    
    try:
        lines = []
        file_updated = False
        time_updated = False
        
        with open("inputs.dat", "r") as f:
            for line in f:
                # Keep comments and whitespace intact, check prefix
                stripped = line.strip()
                if stripped.startswith("RESTART_FILE"):
                    lines.append(f"RESTART_FILE = {restart_file}\n")
                    file_updated = True
                elif stripped.startswith("RESTART_TIME"):
                    lines.append(f"RESTART_TIME = {restart_time}\n")
                    time_updated = True
                else:
                    lines.append(line)
                    
        if not file_updated:
            lines.append(f"RESTART_FILE = {restart_file}\n")
        if not time_updated:
            lines.append(f"RESTART_TIME = {restart_time}\n")
            
        with open("inputs.dat", "w") as f:
            f.writelines(lines)
        return True
    except Exception:
        return False

def edit_inputs_dat(nbi):
    if not nbi.is_tty:
        return
    
    # 1. Restore normal cooked terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, nbi.old_settings)
    
    # 2. Clear terminal screen
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()
    
    # 3. Choose the text editor
    editor = os.environ.get("EDITOR", "vim")
    print(f"Opening inputs.dat in {editor}...")
    time.sleep(0.5)
    
    try:
        # 4. Spawn editor in-place
        subprocess.run([editor, "inputs.dat"])
    except Exception as e:
        print(f"Error launching editor '{editor}': {e}")
        print("Falling back to nano...")
        try:
            subprocess.run(["nano", "inputs.dat"])
        except Exception:
            pass
            
    # 5. Re-enable non-blocking raw/cbreak mode
    tty.setcbreak(sys.stdin.fileno())
    
    # 6. Clear screen again to cleanly redraw the TUI
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

class SolverProcess:
    def __init__(self, popen_obj=None, attached_pid=None):
        self.popen_obj = popen_obj
        self.attached_pid = attached_pid
        
    @property
    def pid(self):
        if self.popen_obj:
            return self.popen_obj.pid
        return self.attached_pid
        
    def poll(self):
        if self.popen_obj:
            return self.popen_obj.poll()
        if self.attached_pid:
            if is_pid_running(self.attached_pid):
                return None
            return 0
        return 0

def main():
    # Initial state
    state = {
        "step": 0,
        "t": 0.0,
        "progress": 0.0,
        "eta": "0.0",
        "l2_sum": "0.0",
        "lim_nodes": None,
        "avg_theta": None,
        "last_warning": None,
        "sbm_max_lebesgue": None,
        "sbm_limiter_count": None,
        "sbm_max_dl_ratio": None,
        "sbm_max_ddl_ratio": None,
        "recent_logs": [],
        "t_final": 1.0,
        "num_threads": 1
    }
    
    # Parse inputs.dat parameters
    inputs = parse_inputs_dat()
    state["t_final"] = float(inputs.get("T_FINAL", 1.0))
    state["num_threads"] = int(inputs.get("NUM_THREADS", 1))
    enable_sbm_diags = (inputs.get("ENABLE_SBM_DIAGNOSTICS", "false").lower() == "true") and \
                       (inputs.get("IB_METHOD", "").upper() == "SBM")

    # Process handles
    solver_proc = None
    log_file = None
    status = "STOPPED"
    log_position = 0

    def start_solver(clean=False):
        nonlocal solver_proc, log_file, status, log_position
        status = "STARTING..."
        
        # Reset state fields on start/restart
        state["step"] = 0
        state["t"] = 0.0
        state["progress"] = 0.0
        state["eta"] = "0.0"
        state["l2_sum"] = "0.0"
        state["lim_nodes"] = None
        state["avg_theta"] = None
        state["last_warning"] = None
        state["sbm_max_lebesgue"] = None
        state["sbm_limiter_count"] = None
        state["sbm_max_dl_ratio"] = None
        state["sbm_max_ddl_ratio"] = None
        
        # Clean any old STOP file
        if os.path.exists("STOP"):
            try: os.remove("STOP")
            except: pass
            
        # Re-parse inputs
        curr_inputs = parse_inputs_dat()
        state["t_final"] = float(curr_inputs.get("T_FINAL", 1.0))
        state["num_threads"] = int(curr_inputs.get("NUM_THREADS", 1))
        
        # Open out.log
        log_file = open("out.log", "w")
        
        # Set environment and start run.sh in a new process group
        env = os.environ.copy()
        env["TUI_ACTIVE"] = "1"
        
        cmd = ["./run.sh"]
        if clean:
            cmd.append("-clean")
            
        popen_obj = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env,
            preexec_fn=os.setsid
        )
        solver_proc = SolverProcess(popen_obj=popen_obj)
        status = "RUNNING"
        log_position = 0
        state["recent_logs"] = ["Process started (PID: {}){}".format(solver_proc.pid, " with -clean" if clean else "")]

    def stop_solver_clean():
        nonlocal status
        if solver_proc and solver_proc.poll() is None:
            status = "STOPPING..."
            # Write STOP file
            with open("STOP", "w") as f:
                f.write("STOP")
            
            # Wait for exit
            start_wait = time.time()
            while solver_proc.poll() is None:
                if time.time() - start_wait > 4.0:
                    # Timeout, kill process group
                    status = "FORCE KILLING..."
                    try:
                        if solver_proc.popen_obj:
                            os.killpg(os.getpgid(solver_proc.popen_obj.pid), signal.SIGKILL)
                        else:
                            os.kill(solver_proc.attached_pid, signal.SIGKILL)
                    except:
                        pass
                    break
                time.sleep(0.1)
        status = "STOPPED"

    def kill_solver_forced():
        nonlocal status
        if solver_proc and solver_proc.poll() is None:
            status = "KILLING..."
            try:
                if solver_proc.popen_obj:
                    os.killpg(os.getpgid(solver_proc.popen_obj.pid), signal.SIGKILL)
                else:
                    os.kill(solver_proc.attached_pid, signal.SIGKILL)
            except:
                pass
            status = "STOPPED"

    # Start solver or attach to existing at TUI startup
    attach_mode = False
    for arg in sys.argv[1:]:
        if arg in ("--attach", "-attach", "attach"):
            attach_mode = True

    if attach_mode:
        pid = find_running_solver_pid()
        if pid:
            solver_proc = SolverProcess(attached_pid=pid)
            status = "RUNNING"
            state["recent_logs"] = [f"Attached to running process (PID: {pid})"]
        else:
            solver_proc = None
            status = "STOPPED"
            state["recent_logs"] = ["No running solver found. Ready to start."]
        log_position = 0
    else:
        start_solver()

    # Clear screen once at startup
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

    last_render_time = 0.0
    last_status = status

    # Enter TUI loop
    with NonBlockingInput() as nbi:
        try:
            while True:
                # Check solver process status
                if solver_proc:
                    ret_code = solver_proc.poll()
                    if ret_code is not None:
                        if status == "RUNNING" or status == "STOPPING...":
                            status = "STOPPED"
                
                # Poll/read out.log
                if os.path.exists("out.log"):
                    file_size = os.path.getsize("out.log")
                    if file_size < log_position:
                        log_position = 0 # reset if truncated
                        
                    with open("out.log", "r", errors="ignore") as f:
                        f.seek(log_position)
                        new_lines = f.readlines()
                        log_position = f.tell()
                        
                    for line in new_lines:
                        raw_line = line.strip()
                        if not raw_line:
                            continue
                            
                        # Parse step iteration
                        step_match = step_pattern.search(raw_line)
                        if step_match:
                            state["step"] = int(step_match.group(1))
                            state["t"] = float(step_match.group(2))
                            state["progress"] = float(step_match.group(3))
                            state["eta"] = step_match.group(4)
                            state["l2_sum"] = step_match.group(5)
                            if step_match.group(6):
                                state["lim_nodes"] = int(step_match.group(6))
                                state["avg_theta"] = step_match.group(7)
                            else:
                                state["lim_nodes"] = None
                                state["avg_theta"] = None
                            continue
                            
                        # Parse Warning
                        if warning_pattern.search(raw_line):
                            state["last_warning"] = raw_line
                            
                        # Parse SBM diagnostics
                        sbm_match = sbm_diag_pattern.search(raw_line)
                        if sbm_match:
                            state["sbm_max_lebesgue"] = sbm_match.group(1)
                            state["sbm_limiter_count"] = sbm_match.group(2)
                            state["sbm_max_dl_ratio"] = sbm_match.group(3)
                            state["sbm_max_ddl_ratio"] = sbm_match.group(4)
                            
                        # Append to log feed (exclude progress lines to keep it clean)
                        if not step_match and not sbm_match:
                            state["recent_logs"].append(raw_line)
                            if len(state["recent_logs"]) > 50:
                                state["recent_logs"].pop(0)

                # Render UI (at 1Hz or immediately on state/status change)
                now = time.time()
                status_changed = (status != last_status)
                if now - last_render_time >= 1.0 or status_changed:
                    last_render_time = now
                    last_status = status
                    
                    # 1. Header
                    current_time = datetime.now().strftime("%H:%M:%S")
                    pid_str = str(solver_proc.pid) if solver_proc else "N/A"
                    
                    status_color = CLR_GREEN if status == "RUNNING" else (CLR_YELLOW if "..." in status else CLR_RED)
                    status_part = f"Status: {status_color}{status}{CLR_RESET}"
                    time_part = f"Time: {current_time}"
                    pid_part = f"PID: {pid_str}"
                    
                    ui_output = []
                    ui_output.append("\033[H") # Move cursor to top-left instead of clearing
                    ui_output.append("┌" + "─" * 76 + "┐")
                    ui_output.append(f"│ {CLR_BOLD}{BG_BLUE}{'  FR-IGR SOLVER RUNTIME MONITOR  ':^74}{CLR_RESET} │")
                    
                    line2 = pad_ansi(status_part, 30) + pad_ansi(time_part, 24) + pad_ansi(pid_part, 20)
                    ui_output.append(f"│ {line2} │")
                    
                    threads_part = f"Threads: {state['num_threads']}"
                    line3 = pad_ansi(threads_part, 74)
                    ui_output.append(f"│ {line3} │")
                    ui_output.append("└" + "─" * 76 + "┘")

                    # 2. Simulation Progress Panel
                    bar_width = 40
                    filled_width = int(state["progress"] * bar_width / 100.0)
                    filled_width = min(bar_width, max(0, filled_width))
                    bar = "█" * filled_width + "░" * (bar_width - filled_width)
                    
                    progress_lines = [
                        f"Step:      {state['step']:<10}      Time:   {state['t']:.4f} / {state['t_final']:.4f}",
                        f"Progress:  [{bar}] {state['progress']:.2f}%",
                        f"ETA:       {state['eta']}s          L2 Sum: {state['l2_sum']}"
                    ]
                    if state["lim_nodes"] is not None:
                        progress_lines.append(f"Limiter:   {state['lim_nodes']} nodes limited (Avg Theta: {state['avg_theta']})")
                    else:
                        progress_lines.append("Limiter:   No limiting active or disabled")
                        
                    ui_output.extend(make_panel("SIMULATION PROGRESS", progress_lines))

                    # 3. Warning Panel
                    warn_text = state["last_warning"] if state["last_warning"] else "No warnings detected."
                    warn_color = CLR_YELLOW if state["last_warning"] else CLR_GRAY
                    ui_output.extend(make_panel("LATEST WARNING", [f"{warn_color}{warn_text}{CLR_RESET}"]))

                    # 4. SBM Diagnostics Panel
                    if enable_sbm_diags:
                        if state["sbm_max_lebesgue"] is not None:
                            sbm_lines = [
                                f"Max Lebesgue: {CLR_BOLD}{state['sbm_max_lebesgue']}{CLR_RESET} | Limiter Triggers: {CLR_BOLD}{state['sbm_limiter_count']}{CLR_RESET}",
                                f"Max D/L:      {CLR_BOLD}{state['sbm_max_dl_ratio']}{CLR_RESET} | Max D/dL:         {CLR_BOLD}{state['sbm_max_ddl_ratio']}{CLR_RESET}"
                            ]
                        else:
                            sbm_lines = [f"{CLR_GRAY}SBM diagnostics active. Waiting for solver update...{CLR_RESET}"]
                    else:
                        sbm_lines = [f"{CLR_GRAY}SBM diagnostics not enabled in inputs.dat.{CLR_RESET}"]
                    ui_output.extend(make_panel("SBM DIAGNOSTICS", sbm_lines))

                    # 5. Recent Log Output Feed
                    log_lines = []
                    for l in state["recent_logs"][-6:]:
                        clean_l = l[:72]
                        log_lines.append(f"{CLR_GRAY}{clean_l}{CLR_RESET}")
                    while len(log_lines) < 6:
                        log_lines.append("")
                    ui_output.extend(make_panel("RECENT LOG FEED", log_lines))

                    # 6. Actions Panel
                    action_line = f"[{CLR_BOLD}{CLR_GREEN}S{CLR_RESET}] Stop  [{CLR_BOLD}{CLR_GREEN}R{CLR_RESET}] Restart  [{CLR_BOLD}{CLR_GREEN}C{CLR_RESET}] Clean Restart  [{CLR_BOLD}{CLR_GREEN}E{CLR_RESET}] Edit  [{CLR_BOLD}{CLR_GREEN}K{CLR_RESET}] Kill  [{CLR_BOLD}{CLR_GREEN}Q{CLR_RESET}] Quit"
                    ui_output.extend(make_panel("ACTIONS", [action_line]))

                    # Print screen at once
                    sys.stdout.write("\n".join(ui_output) + "\n")
                    sys.stdout.flush()

                # Handle user keyboard input
                char = nbi.get_char()
                if char:
                    c = char.upper()
                    if c == "S":
                        state["recent_logs"].append("Stop requested. Signaling solver...")
                        stop_solver_clean()
                        last_status = "" # Force immediate redraw
                    elif c == "R":
                        state["recent_logs"].append("Restart requested. Stopping solver...")
                        stop_solver_clean()
                        
                        # Find and update latest checkpoint
                        latest_file, latest_time = find_latest_checkpoint_from_pvd()
                        if latest_file:
                            state["recent_logs"].append(f"Auto-restart: latest checkpoint {latest_file} at t={latest_time}")
                            if update_restart_in_inputs_dat(latest_file, latest_time):
                                state["recent_logs"].append("Updated inputs.dat with restart configuration.")
                            else:
                                state["recent_logs"].append("Warning: Failed to update inputs.dat.")
                        else:
                            state["recent_logs"].append("No checkpoints found. Resetting restart configuration in inputs.dat.")
                            update_restart_in_inputs_dat("", 0.0)
                            
                        state["recent_logs"].append("Restarting solver process...")
                        curr_inputs = parse_inputs_dat()
                        enable_sbm_diags = (curr_inputs.get("ENABLE_SBM_DIAGNOSTICS", "false").lower() == "true") and \
                                           (curr_inputs.get("IB_METHOD", "").upper() == "SBM")
                        start_solver(clean=False)
                        last_status = "" # Force immediate redraw
                    elif c == "C":
                        state["recent_logs"].append("Clean Restart requested. Stopping solver...")
                        stop_solver_clean()
                        
                        # Reset restart configuration for clean restart
                        state["recent_logs"].append("Resetting restart configuration in inputs.dat for clean run.")
                        update_restart_in_inputs_dat("", 0.0)
                        
                        state["recent_logs"].append("Running clean compilation and starting fresh...")
                        curr_inputs = parse_inputs_dat()
                        enable_sbm_diags = (curr_inputs.get("ENABLE_SBM_DIAGNOSTICS", "false").lower() == "true") and \
                                           (curr_inputs.get("IB_METHOD", "").upper() == "SBM")
                        start_solver(clean=True)
                    elif c == "E":
                        state["recent_logs"].append("Opening inputs.dat for editing...")
                        edit_inputs_dat(nbi)
                        
                        # Re-parse inputs.dat configuration to update state
                        curr_inputs = parse_inputs_dat()
                        state["t_final"] = float(curr_inputs.get("T_FINAL", 1.0))
                        state["num_threads"] = int(curr_inputs.get("NUM_THREADS", 1))
                        enable_sbm_diags = (curr_inputs.get("ENABLE_SBM_DIAGNOSTICS", "false").lower() == "true") and \
                                           (curr_inputs.get("IB_METHOD", "").upper() == "SBM")
                        
                        state["recent_logs"].append("Finished editing inputs.dat. Configuration reloaded.")
                        last_status = "" # Force immediate redraw
                    elif c == "K":
                        state["recent_logs"].append("Forcing process SIGKILL...")
                        kill_solver_forced()
                        last_status = "" # Force immediate redraw
                    elif c == "Q":
                        if status == "RUNNING":
                            if nbi.is_tty:
                                sys.stdout.write("\033[H\033[J")
                                sys.stdout.flush()
                                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, nbi.old_settings)
                                ans = input("Solver is still running. Stop it before quitting? (y/n): ")
                                if ans.strip().lower() in ("y", "yes", ""):
                                    tty.setcbreak(sys.stdin.fileno())
                                    stop_solver_clean()
                                    break
                                else:
                                    break
                            else:
                                stop_solver_clean()
                                break
                        else:
                            break

                time.sleep(0.05) # fast input poll sleep
        finally:
            # Ensure log file is closed
            if log_file and not log_file.closed:
                log_file.close()

if __name__ == "__main__":
    main()

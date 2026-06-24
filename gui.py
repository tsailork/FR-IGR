#!/usr/bin/env python3
import os
import sys
import time
import traceback
import json
import re
import subprocess
import platform
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global variables
CASE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "cases/default_case")
PORT = 8080
SOLVER_PROC = None
RELOAD_VERSION = 0

# Server Error Logger
def write_error_log(path, method, error_msg, tb_str=None):
    try:
        log_path = os.path.join(CASE_DIR, "gui_error.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== ERROR {timestamp} ===\n")
            f.write(f"Request: {method} {path}\n")
            f.write(f"Error: {error_msg}\n")
            if tb_str:
                f.write(f"Traceback:\n{tb_str}\n")
            f.write("=========================\n\n")
    except Exception:
        pass

# Print target case details
print(f"FR-IGR Case Setup GUI starting...")
print(f"Target Case Directory: {CASE_DIR}")
if not os.path.exists(CASE_DIR):
    print(f"Warning: Case directory '{CASE_DIR}' does not exist. Creating it...")
    os.makedirs(CASE_DIR, exist_ok=True)

# Helper function to parse inputs.dat or domain.grid in case folder
def parse_config_file(filepath):
    config = {}
    if not os.path.exists(filepath):
        return config
    
    current_section = "Global"
    with open(filepath, "r") as f:
        for line in f:
            # Strip comments
            line = line.split("#")[0].split(";")[0].strip()
            if not line:
                continue
            
            # Check for section header
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                if current_section not in config:
                    config[current_section] = {}
            else:
                # Key-value
                if "=" in line:
                    parts = line.split("=", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if current_section not in config:
                        config[current_section] = {}
                    config[current_section][key] = val
    return config

# Helper function to write config back to file
def write_config_file(filepath, config_dict):
    with open(filepath, "w") as f:
        for section, kv in config_dict.items():
            f.write(f"[{section}]\n")
            for key, val in kv.items():
                # Skip empty values to let C++ solver use its default values,
                # except for RESTART_FILE which can be written empty.
                if str(val).strip() == "" and key != "RESTART_FILE":
                    continue
                f.write(f"{key} = {val}\n")
            f.write("\n")

# Process Checker
def check_solver_running():
    global SOLVER_PROC
    # 1. Check Python subprocess handle
    if SOLVER_PROC is not None:
        if SOLVER_PROC.poll() is None:
            return True
        else:
            SOLVER_PROC = None

    # 2. Check WSL/Linux process table using pgrep
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["wsl", "pgrep", "-f", "fr_solver"]).decode().strip()
        else:
            out = subprocess.check_output(["pgrep", "-f", "fr_solver"]).decode().strip()
        if out:
            return True
    except subprocess.CalledProcessError:
        pass
    return False

# VTK XML Parser for flow visualization
def parse_latest_vts(var_name):
    # Map lowercase frontend var names to actual VTK array names
    mapping = {
        "rho": "rho",
        "rho_u": "rho_u",
        "rho_v": "rho_v",
        "rho_E": "rho_E",
        "u": "u",
        "v": "v",
        "press": "Pressure",
        "pressure": "Pressure",
        "temp": "Temperature",
        "temperature": "Temperature",
        "mach": "Mach",
        "sigma": "Sigma",
        "phi": "phi"
    }
    var_name = mapping.get(var_name.lower(), var_name)

    # 1. Find latest VTM path from plot.pvd or solution.pvd
    pvd_path = os.path.join(CASE_DIR, "pv_outputs", "plot.pvd")
    if not os.path.exists(pvd_path):
        pvd_path = os.path.join(CASE_DIR, "pv_outputs", "solution.pvd")
    if not os.path.exists(pvd_path):
        return []

    try:
        tree = ET.parse(pvd_path)
        root = tree.getroot()
        datasets = root.findall(".//DataSet")
        if not datasets:
            return []
        
        # Get the latest written VTM file relative path
        latest_file = datasets[-1].attrib.get("file")
        if not latest_file:
            return []
        
        vtm_path = os.path.join(CASE_DIR, "pv_outputs", os.path.basename(latest_file))
        if not os.path.exists(vtm_path):
            # Try absolute or subfolder paths
            vtm_path = os.path.join(CASE_DIR, "pv_outputs", latest_file)
            if not os.path.exists(vtm_path):
                vtm_path = os.path.join(CASE_DIR, latest_file)
                if not os.path.exists(vtm_path):
                    return []

        # 2. Parse VTM file to list VTS/VTU block files
        vtm_dir = os.path.dirname(vtm_path)
        vtm_tree = ET.parse(vtm_path)
        vtm_root = vtm_tree.getroot()
        datasets_vtm = vtm_root.findall(".//DataSet")
        
        blocks_data = []
        for dataset in datasets_vtm:
            block_id = int(dataset.attrib.get("index", 0))
            block_file = dataset.attrib.get("file")
            vts_path = os.path.join(vtm_dir, block_file)
            if not os.path.exists(vts_path):
                continue
            
            # 3. Parse VTS/VTU file
            vts_tree = ET.parse(vts_path)
            vts_root = vts_tree.getroot()
            grid = vts_root.find(".//StructuredGrid")
            is_structured = True
            if grid is None:
                grid = vts_root.find(".//UnstructuredGrid")
                if grid is None:
                    continue
                is_structured = False
            
            if is_structured:
                extent = grid.attrib.get("WholeExtent", "0 0 0 0 0 0").split()
                nx = int(extent[1]) - int(extent[0]) + 1
                ny = int(extent[3]) - int(extent[2]) + 1
            else:
                piece = grid.find(".//Piece")
                if piece is None:
                    continue
                n_points = int(piece.attrib.get("NumberOfPoints", 0))
                nx = n_points
                ny = 1
            
            # Extract points
            points_node = vts_root.find(".//Points/DataArray")
            if points_node is None or not points_node.text:
                continue
            points_vals = [float(x) for x in points_node.text.split()]
            x_coords = points_vals[0::3]
            y_coords = points_vals[1::3]
            
            # Extract target scalar values
            scalar_node = vts_root.find(f".//PointData/DataArray[@Name='{var_name}']")
            if scalar_node is None:
                # Fallback to rho if specific var not found
                scalar_node = vts_root.find(".//PointData/DataArray[@Name='rho']")
            
            if scalar_node is not None and scalar_node.text:
                values = [float(x) for x in scalar_node.text.split()]
            else:
                values = [0.0] * (nx * ny)
                
            blocks_data.append({
                "block_id": block_id,
                "nx": nx,
                "ny": ny,
                "x": x_coords,
                "y": y_coords,
                "values": values,
                "is_structured": is_structured
            })
            
        return blocks_data
    except Exception as e:
        print(f"Error parsing VTK files: {e}", file=sys.stderr)
        return []

def fast_parse_vts(vts_path, var_name):
    # Map lowercase frontend var names to actual VTK array names
    mapping = {
        "rho": "rho",
        "rho_u": "rho_u",
        "rho_v": "rho_v",
        "rho_E": "rho_E",
        "u": "u",
        "v": "v",
        "press": "Pressure",
        "pressure": "Pressure",
        "temp": "Temperature",
        "temperature": "Temperature",
        "mach": "Mach",
        "sigma": "Sigma",
        "phi": "phi"
    }
    var_name = mapping.get(var_name.lower(), var_name)

    try:
        with open(vts_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        is_structured = True
        extent_match = re.search(r'WholeExtent="([^"]+)"', content)
        if extent_match:
            extent = [int(x) for x in extent_match.group(1).split()]
            nx = extent[1] - extent[0] + 1
            ny = extent[3] - extent[2] + 1
        else:
            # Try parsing as unstructured grid Piece NumberOfPoints
            points_match = re.search(r'NumberOfPoints="(\d+)"', content)
            if not points_match:
                points_match = re.search(r'<Piece[^>]+NumberOfPoints="(\d+)"', content)
            if not points_match:
                return None
            n_points = int(points_match.group(1))
            nx = n_points
            ny = 1
            is_structured = False
        
        points_block_match = re.search(r'<Points>\s*<DataArray[^>]*>(.*?)</DataArray>\s*</Points>', content, re.DOTALL)
        if not points_block_match:
            return None
        points_str = points_block_match.group(1).strip()
        points_vals = np.fromstring(points_str, dtype=float, sep=' ')
        x_coords = points_vals[0::3]
        y_coords = points_vals[1::3]
        
        scalar_pattern = rf'<DataArray[^>]+Name=["\']{var_name}["\'][^>]*>(.*?)</DataArray>'
        scalar_match = re.search(scalar_pattern, content, re.DOTALL)
        if not scalar_match:
            scalar_match = re.search(r'<DataArray[^>]+Name=["\']rho["\'][^>]*>(.*?)</DataArray>', content, re.DOTALL)
            if not scalar_match:
                return None
        
        scalar_str = scalar_match.group(1).strip()
        values = np.fromstring(scalar_str, dtype=float, sep=' ')
        
        return {
            "nx": nx,
            "ny": ny,
            "x": x_coords,
            "y": y_coords,
            "values": values,
            "is_structured": is_structured
        }
    except Exception as e:
        print(f"Error in fast_parse_vts: {e}", file=sys.stderr)
        return None

def fast_parse_vtm(vtm_path):
    try:
        tree = ET.parse(vtm_path)
        root = tree.getroot()
        datasets = root.findall(".//DataSet")
        res = []
        seen = set()
        for d in datasets:
            idx_str = d.attrib.get("index")
            file_val = d.attrib.get("file")
            if idx_str is not None and file_val is not None:
                try:
                    idx = int(idx_str)
                    if idx not in seen:
                        res.append((idx, file_val))
                        seen.add(idx)
                except ValueError:
                    pass
        return res
    except Exception as e:
        print(f"Error in fast_parse_vtm: {e}", file=sys.stderr)
        return []

def get_webcontour_settings(var_name):
    path = os.path.join(CASE_DIR, ".webcontour")
    defaults = {
        "cmap": "viridis",
        "levels": 50,
        "range_mode": "auto",
        "vmin": 0.0,
        "vmax": 1.0,
        "show_grid": False
    }
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data.get(var_name, defaults)
        except Exception:
            pass
    return defaults

CONTOUR_CACHE = {}

def generate_contour_plot(var_name, vtm_name=None):
    vtm_path = None
    if vtm_name:
        vtm_path = os.path.join(CASE_DIR, "pv_outputs", vtm_name)
        if not os.path.exists(vtm_path):
            vtm_path = os.path.join(CASE_DIR, vtm_name)
    else:
        pvd_path = os.path.join(CASE_DIR, "pv_outputs", "plot.pvd")
        if not os.path.exists(pvd_path):
            pvd_path = os.path.join(CASE_DIR, "pv_outputs", "solution.pvd")
        if os.path.exists(pvd_path):
            try:
                with open(pvd_path, "r") as f:
                    pvd_content = f.read()
                datasets = re.findall(r'<DataSet[^>]+file=["\']([^"\']+)["\']', pvd_content)
                if datasets:
                    vtm_file = os.path.basename(datasets[-1])
                    vtm_path = os.path.join(CASE_DIR, "pv_outputs", vtm_file)
            except Exception:
                pass
                
    if not vtm_path or not os.path.exists(vtm_path):
        return None

    # Check cache using mtime
    global RELOAD_VERSION
    try:
        mtime = os.path.getmtime(vtm_path)
    except Exception:
        mtime = 0.0
        
    webcontour_path = os.path.join(CASE_DIR, ".webcontour")
    webcontour_mtime = 0.0
    if os.path.exists(webcontour_path):
        try:
            webcontour_mtime = os.path.getmtime(webcontour_path)
        except Exception:
            pass
            
    mtime = max(mtime, webcontour_mtime, RELOAD_VERSION)
        
    cache_key = (var_name, os.path.basename(vtm_path), mtime)
    if cache_key in CONTOUR_CACHE:
        return CONTOUR_CACHE[cache_key]

    vts_entries = fast_parse_vtm(vtm_path)
    vtm_dir = os.path.dirname(vtm_path)
    
    blocks_data = []
    global_min = float('inf')
    global_max = float('-inf')
    
    for block_id, vts_file in vts_entries:
        vts_path = os.path.join(vtm_dir, vts_file)
        if not os.path.exists(vts_path):
            continue
        data = fast_parse_vts(vts_path, var_name)
        if data:
            data["block_id"] = block_id
            blocks_data.append(data)
            val_min = np.min(data["values"])
            val_max = np.max(data["values"])
            if val_min < global_min: global_min = val_min
            if val_max > global_max: global_max = val_max

    if not blocks_data:
        return None

    settings = get_webcontour_settings(var_name)
    cmap = settings.get("cmap", "viridis")
    levels_count = int(settings.get("levels", 50))
    range_mode = settings.get("range_mode", "auto")
    show_grid = settings.get("show_grid", False)
    
    if range_mode == "manual":
        vmin = float(settings.get("vmin", 0.0))
        vmax = float(settings.get("vmax", 1.0))
    else:
        vmin = global_min
        vmax = global_max
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1.0
            
    levels = np.linspace(vmin, vmax, levels_count + 1)
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    
    cf = None
    for b in blocks_data:
        is_structured = b.get("is_structured", True)
        if is_structured:
            nx, ny = b["nx"], b["ny"]
            X = b["x"].reshape((ny, nx))
            Y = b["y"].reshape((ny, nx))
            V = b["values"].reshape((ny, nx))
            
            cf = ax.contourf(X, Y, V, levels=levels, cmap=cmap, extend='both')
            
            if show_grid:
                ax.plot(X, Y, color='white', alpha=0.15, linewidth=0.5)
                ax.plot(X.T, Y.T, color='white', alpha=0.15, linewidth=0.5)
        else:
            X = b["x"]
            Y = b["y"]
            V = b["values"]
            cf = ax.tricontourf(X, Y, V, levels=levels, cmap=cmap, extend='both')
            
            if show_grid:
                ax.scatter(X, Y, color='white', alpha=0.15, s=0.5)

    if cf:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        
    ax.set_title(f"Flow Contour: {var_name}", fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_aspect('equal', 'box')
    
    import io
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    
    # Store in cache
    CONTOUR_CACHE[cache_key] = png_bytes
    return png_bytes

def find_latest_checkpoint_from_pvd():
    pvd_path = os.path.join(CASE_DIR, "pv_outputs", "solution.pvd")
    if not os.path.exists(pvd_path):
        return None, 0.0
    
    latest_file = None
    latest_time = 0.0
    
    try:
        dataset_pattern = re.compile(r'<DataSet\s+timestep="([0-9eE\.\+-]+)"[^>]*file="([^"]+)"')
        dataset_pattern_alt = re.compile(r'<DataSet\s+file="([^"]+)"[^>]*timestep="([0-9eE\.\+-]+)"')
        with open(pvd_path, "r") as f:
            for line in f:
                match = dataset_pattern.search(line)
                if match:
                    t_val = float(match.group(1))
                    f_val = match.group(2)
                    if t_val >= latest_time:
                        latest_time = t_val
                        latest_file = f_val
                else:
                    match_alt = dataset_pattern_alt.search(line)
                    if match_alt:
                        t_val = float(match_alt.group(2))
                        f_val = match_alt.group(1)
                        if t_val >= latest_time:
                            latest_time = t_val
                            latest_file = f_val
    except Exception:
        pass
        
    if latest_file:
        latest_file = os.path.basename(latest_file)
        latest_file = "pv_outputs/" + latest_file
            
    return latest_file, latest_time

def update_restart_in_inputs_dat(restart_file, restart_time):
    inputs_path = os.path.join(CASE_DIR, "inputs.dat")
    if not os.path.exists(inputs_path):
        return False
    
    try:
        lines = []
        file_updated = False
        time_updated = False
        
        with open(inputs_path, "r") as f:
            for line in f:
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
            
        with open(inputs_path, "w") as f:
            f.writelines(lines)
        return True
    except Exception:
        return False

def validate_config(inputs_dict, domain_dict):
    blocks = []
    if domain_dict:
        for section, kv in domain_dict.items():
            if section.startswith("Block"):
                try:
                    bid = int(section.replace("Block", ""))
                except ValueError:
                    return False, f"Invalid block section name: '{section}'"
                    
                try:
                    nx = int(kv.get("N_ELEM_X", 0))
                    ny = int(kv.get("N_ELEM_Y", 0))
                except ValueError:
                    return False, f"Block {bid} has non-integer element counts."
                    
                try:
                    xmin = float(kv.get("X_MIN", 0.0))
                    xmax = float(kv.get("X_MAX", 0.0))
                    ymin = float(kv.get("Y_MIN", 0.0))
                    ymax = float(kv.get("Y_MAX", 0.0))
                except ValueError:
                    return False, f"Block {bid} has invalid floating-point coordinates."
                    
                if nx <= 0 or ny <= 0:
                    return False, f"Block {bid} has invalid element counts ({nx}, {ny}). Must be > 0."
                if xmax <= xmin or ymax <= ymin:
                    return False, f"Block {bid} has invalid dimensions (X: {xmin} to {xmax}, Y: {ymin} to {ymax})."
                    
                blocks.append({
                    "id": bid,
                    "nx": nx,
                    "ny": ny,
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "bc_l": kv.get("BC_L", "TRANSMISSIVE"),
                    "bc_r": kv.get("BC_R", "TRANSMISSIVE"),
                    "bc_b": kv.get("BC_B", "TRANSMISSIVE"),
                    "bc_t": kv.get("BC_T", "TRANSMISSIVE")
                })
                
        blocks.sort(key=lambda b: b["id"])
        
        # Check interfaces
        for b in blocks:
            faces = [("L", b["bc_l"]), ("R", b["bc_r"]), ("B", b["bc_b"]), ("T", b["bc_t"])]
            for face_name, bc in faces:
                bc = bc.strip()
                parts = bc.split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1] in ["L", "R", "B", "T"]:
                    nid = int(parts[0])
                    nface = parts[1]
                    
                    neighbor = next((x for x in blocks if x["id"] == nid), None)
                    if not neighbor:
                        return False, f"Block {b['id']} face {face_name} points to missing Block {nid}."
                        
                    expected_neighbor_bc = f"{b['id']}:{face_name}"
                    actual_neighbor_bc = ""
                    if nface == "L": actual_neighbor_bc = neighbor["bc_l"]
                    elif nface == "R": actual_neighbor_bc = neighbor["bc_r"]
                    elif nface == "B": actual_neighbor_bc = neighbor["bc_b"]
                    elif nface == "T": actual_neighbor_bc = neighbor["bc_t"]
                    
                    if actual_neighbor_bc.strip() != expected_neighbor_bc:
                        return False, (f"Asymmetric boundary condition. Block {b['id']} face {face_name} "
                                       f"points to Block {nid} face {nface}, but that face has BC "
                                       f"'{actual_neighbor_bc}' (expected '{expected_neighbor_bc}').")
                                       
                    my_elems = b["ny"] if face_name in ["L", "R"] else b["nx"]
                    n_elems = neighbor["ny"] if nface in ["L", "R"] else neighbor["nx"]
                    if my_elems != n_elems:
                        return False, (f"Element count mismatch between Block {b['id']} ({my_elems} elems) "
                                       f"and Block {nid} ({n_elems} elems) on shared interface.")
                                       
                    my_len = (b["ymax"] - b["ymin"]) if face_name in ["L", "R"] else (b["xmax"] - b["xmin"])
                    n_len = (neighbor["ymax"] - neighbor["ymin"]) if nface in ["L", "R"] else (neighbor["xmax"] - neighbor["xmin"])
                    if abs(my_len - n_len) > 1e-8:
                        return False, (f"Physical length mismatch between Block {b['id']} ({my_len}) "
                                       f"and Block {nid} ({n_len}) on shared interface.")
                                       
    if inputs_dict:
        float_keys = [
            "GAMMA", "RHO_INF", "U_INF", "V_INF", "P_INF",
            "CFL", "T_FINAL",
            "RE", "PR", "MACH_REF", "NS_BR2_ETA",
            "ALPHA_SCALE", "IGR_TAU_R", "IGR_BR2_ETA",
            "POS_LIMITER_EPS",
            "IB_DL_SCALE", "IB_L_SCALE", "IB_AOA", "IB_CENTER_X", "IB_CENTER_Y", "IB_RADIUS", "IB_SMOOTH_WIDTH",
            "IB_PENALIZATION_ETA", "IB_VELOCITY_X", "IB_VELOCITY_Y", "IB_TEMPERATURE",
            "OUTPUT_INTERVAL", "RESTART_INTERVAL", "RESIDUAL_INTERVAL", "PROBE_INTERVAL", "PRINT_INTERVAL", "RESTART_TIME"
        ]
        int_keys = [
            "P_DEG", "NUM_THREADS", "IGR_SUB_ITERS", "IB_NUM_QUADS", "IB_NUM_POLYS"
        ]
        
        for section, kv in inputs_dict.items():
            for key, val in kv.items():
                val_str = str(val).strip()
                if not val_str:
                    continue  # We will omit empty values in write_config_file
                
                if key in float_keys:
                    try:
                        float(val_str)
                    except ValueError:
                        return False, f"Parameter '{key}' in section [{section}] must be a valid floating-point number (got '{val_str}')."
                        
                elif key in int_keys:
                    try:
                        int(val_str)
                    except ValueError:
                        return False, f"Parameter '{key}' in section [{section}] must be a valid integer (got '{val_str}')."
                            
    return True, "Success"

# HTTP Request Handler
class GUIRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress console spam
        pass

    def send_error(self, code, message=None, explain=None):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = None
        if exc_type is not None:
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        write_error_log(self.path, self.command, f"HTTP {code}: {message or ''}", tb_str)
        super().send_error(code, message, explain)

    def do_GET(self):
        try:
            self._do_GET()
        except Exception as e:
            self.send_error(500, f"Internal Server Error in GET {self.path}: {e}")

    def _do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path
        
        # 1. Static file routing
        if path in ["", "/", "/index.html"]:
            self.serve_static_file("gui/index.html", "text/html")
        elif path == "/style.css":
            self.serve_static_file("gui/style.css", "text/css")
        elif path == "/app.js":
            self.serve_static_file("gui/app.js", "application/javascript")
        elif path == "/js/chart.js":
            self.serve_static_file("gui/js/chart.js", "application/javascript")
            
        # 2. Config reading API
        elif path == "/api/config":
            inputs = parse_config_file(os.path.join(CASE_DIR, "inputs.dat"))
            domain = parse_config_file(os.path.join(CASE_DIR, "domain.grid"))
            payload = {"inputs": inputs, "domain": domain}
            self.send_json_response(payload)
            
        # 3. Status and logs API
        elif path == "/api/status":
            running = check_solver_running()
            
            # Read last 150 lines of logs
            logs = []
            log_path = os.path.join(CASE_DIR, "out.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        logs = [l.rstrip() for l in lines[-150:]]
                except Exception as e:
                    logs = [f"Error reading log file: {e}"]
                    
            # Parse residuals
            residuals = []
            res_path = os.path.join(CASE_DIR, "csv_outputs", "residuals.csv")
            if os.path.exists(res_path):
                try:
                    with open(res_path, "r") as f:
                        # Skip comment lines, parse columns
                        for line in f:
                            if line.startswith("#") or line.strip() == "":
                                continue
                            parts = line.strip().split(",")
                            if len(parts) >= 5:
                                try:
                                    residuals.append({
                                        "time": float(parts[0]),
                                        "rho": float(parts[1]),
                                        "rhou": float(parts[2]),
                                        "rhov": float(parts[3]),
                                        "E": float(parts[4])
                                    })
                                except ValueError:
                                    pass
                except Exception:
                    pass
            # Limit returned data points for performance
            residuals = residuals[-300:]

            # Parse probe data
            probes = []
            probe_path = os.path.join(CASE_DIR, "csv_outputs", "probe.csv")
            if os.path.exists(probe_path):
                try:
                    with open(probe_path, "r") as f:
                        header = None
                        for line in f:
                            parts = line.strip().split(",")
                            if not header:
                                header = parts
                                continue
                            if len(parts) == len(header):
                                try:
                                    row = {"Time": float(parts[0])}
                                    for i in range(1, len(parts)):
                                        row[header[i].strip()] = float(parts[i])
                                    probes.append(row)
                                except ValueError:
                                    pass
                except Exception:
                    pass
            probes = probes[-300:]
            
            self.send_json_response({
                "running": running,
                "logs": logs,
                "residuals": residuals,
                "probes": probes
            })
            
        # 4. Flow visualizer VTS data API
        elif path == "/api/vts_data":
            query_params = urllib.parse.parse_qs(url_parsed.query)
            var_name = query_params.get("var", ["rho"])[0]
            data = parse_latest_vts(var_name)
            self.send_json_response(data)
            
        # 5. Playback history API
        elif path == "/api/history":
            pvd_path = os.path.join(CASE_DIR, "pv_outputs", "plot.pvd")
            if not os.path.exists(pvd_path):
                pvd_path = os.path.join(CASE_DIR, "pv_outputs", "solution.pvd")
            if not os.path.exists(pvd_path):
                self.send_json_response({"timesteps": []})
                return
            
            try:
                with open(pvd_path, "r") as f:
                    content = f.read()
                
                # Regex match for DataSet tags
                matches = re.findall(r'<DataSet\s+timestep=["\']([\d\.eE\-+]+)["\'][^>]+file=["\']([^"\']+)["\']', content)
                matches += re.findall(r'<DataSet\s+file=["\']([^"\']+)["\']\s+timestep=["\']([\d\.eE\-+]+)["\']', content)
                
                timesteps = []
                seen_times = set()
                for m in matches:
                    try:
                        # Try parsing first index as float, second as file
                        t_val = float(m[0])
                        vtm_file = os.path.basename(m[1])
                    except ValueError:
                        try:
                            # Try parsing second index as float, first as file
                            t_val = float(m[1])
                            vtm_file = os.path.basename(m[0])
                        except ValueError:
                            continue
                    
                    if t_val not in seen_times:
                        vtm_path = os.path.join(CASE_DIR, "pv_outputs", vtm_file)
                        global RELOAD_VERSION
                        try:
                            mtime = os.path.getmtime(vtm_path)
                        except Exception:
                            mtime = 0.0
                            
                        webcontour_path = os.path.join(CASE_DIR, ".webcontour")
                        webcontour_mtime = 0.0
                        if os.path.exists(webcontour_path):
                            try:
                                webcontour_mtime = os.path.getmtime(webcontour_path)
                            except Exception:
                                pass
                        mtime = max(mtime, webcontour_mtime, RELOAD_VERSION)
                            
                        timesteps.append({
                            "time": t_val,
                            "vtm": vtm_file,
                            "mtime": mtime
                        })
                        seen_times.add(t_val)
                        
                timesteps.sort(key=lambda x: x["time"])
                self.send_json_response({"timesteps": timesteps})
            except Exception as e:
                self.send_json_response({"error": str(e)}, status_code=500)
                
        # 6. Contour image generator
        elif path == "/api/contour_image":
            query_params = urllib.parse.parse_qs(url_parsed.query)
            var_name = query_params.get("var", ["rho"])[0]
            vtm_file = query_params.get("vtm", [None])[0]
            mtime = query_params.get("mtime", [None])[0]
            
            png_data = generate_contour_plot(var_name, vtm_file)
            if not png_data:
                self.send_error(404, "Contour data not found or plot failed")
                return
                
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_data)))
            if mtime:
                self.send_header("Cache-Control", "public, max-age=31536000")
            else:
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(png_data)
            
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        try:
            self._do_POST()
        except Exception as e:
            self.send_error(500, f"Internal Server Error in POST {self.path}: {e}")

    def _do_POST(self):
        global SOLVER_PROC
        path = self.path
        
        # 1. Update config API
        if path == "/api/config":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            
            inputs = payload.get("inputs")
            domain = payload.get("domain")
            
            # Validation engine check
            success, err_msg = validate_config(inputs, domain)
            if not success:
                self.send_json_response({"status": "error", "message": err_msg}, status_code=400)
                return
            
            if inputs:
                write_config_file(os.path.join(CASE_DIR, "inputs.dat"), inputs)
            if domain:
                write_config_file(os.path.join(CASE_DIR, "domain.grid"), domain)
                
            self.send_json_response({"status": "success", "message": "Config files saved successfully"})
            
        # 1.5. Webcontour settings API
        elif path == "/api/webcontour":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            
            var_name = payload.get("var")
            settings = payload.get("settings")
            
            if not var_name or not settings:
                self.send_json_response({"status": "error", "message": "Missing var or settings"}, status_code=400)
                return
            
            dotfile_path = os.path.join(CASE_DIR, ".webcontour")
            data = {}
            if os.path.exists(dotfile_path):
                try:
                    with open(dotfile_path, "r") as f:
                        data = json.load(f)
                except Exception:
                    pass
            
            data[var_name] = settings
            
            try:
                with open(dotfile_path, "w") as f:
                    json.dump(data, f, indent=4)
                self.send_json_response({"status": "success", "message": "Webcontour settings saved"})
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to save .webcontour: {e}"}, status_code=500)
            
        # 2. Run simulation API
        elif path == "/api/run":
            if check_solver_running():
                self.send_json_response({"status": "error", "message": "Solver is already running"}, status_code=400)
                return

            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            clean = payload.get("clean", False)
            
            # Make sure out.log is cleared or prepared
            log_file = open(os.path.join(CASE_DIR, "out.log"), "w" if clean else "a")
            
            # Setup process invocation
            if platform.system() == "Windows":
                cmd = ["wsl", "./run_case.sh", "-headless"]
            else:
                cmd = ["./run_case.sh", "-headless"]
                
            if clean:
                cmd.append("-clean")
                
            try:
                SOLVER_PROC = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=CASE_DIR,
                    shell=(platform.system() == "Windows")
                )
                self.send_json_response({"status": "success", "pid": SOLVER_PROC.pid})
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to launch solver: {e}"}, status_code=500)

        # 3. Stop simulation API
        elif path == "/api/stop":
            stop_file = os.path.join(CASE_DIR, "STOP")
            try:
                with open(stop_file, "w") as f:
                    f.write("STOP")
                self.send_json_response({"status": "success", "message": "Stop trigger written successfully"})
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to stop solver: {e}"}, status_code=500)

        # 4. Clean simulation data API
        elif path == "/api/clean":
            if check_solver_running():
                self.send_json_response({"status": "error", "message": "Cannot clean files while solver is running"}, status_code=400)
                return
            
            try:
                # Clean local folder contents
                pv_dir = os.path.join(CASE_DIR, "pv_outputs")
                csv_dir = os.path.join(CASE_DIR, "csv_outputs")
                
                if os.path.exists(pv_dir):
                    for f in os.listdir(pv_dir):
                        os.remove(os.path.join(pv_dir, f))
                if os.path.exists(csv_dir):
                    for f in os.listdir(csv_dir):
                        os.remove(os.path.join(csv_dir, f))
                        
                log_path = os.path.join(CASE_DIR, "out.log")
                if os.path.exists(log_path):
                    os.remove(log_path)
                    
                stop_path = os.path.join(CASE_DIR, "STOP")
                if os.path.exists(stop_path):
                    os.remove(stop_path)
                    
                residuals_path = os.path.join(CASE_DIR, "residuals.dat")
                if os.path.exists(residuals_path):
                    os.remove(residuals_path)
                    
                self.send_json_response({"status": "success", "message": "Outputs cleaned successfully"})
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to clean outputs: {e}"}, status_code=500)
                
        # 5. Restart simulation API
        elif path == "/api/restart":
            # 1. Stop current active solver run if running
            if check_solver_running():
                # Write STOP file
                stop_file = os.path.join(CASE_DIR, "STOP")
                try:
                    with open(stop_file, "w") as f:
                        f.write("STOP")
                except Exception as e:
                    self.send_json_response({"status": "error", "message": f"Failed to send stop signal: {e}"}, status_code=500)
                    return
                
                # Wait for it to exit (up to 10 seconds)
                stopped = False
                for _ in range(100):
                    if not check_solver_running():
                        stopped = True
                        break
                    time.sleep(0.1)
                
                if not stopped:
                    # Force terminate if still running
                    try:
                        if SOLVER_PROC:
                            SOLVER_PROC.terminate()
                            SOLVER_PROC.wait(timeout=2.0)
                    except Exception:
                        pass
            
            # 2. Modify input file to point to last solution/plot file and associated time
            restart_file, restart_time = find_latest_checkpoint_from_pvd()
            if not restart_file:
                self.send_json_response({"status": "error", "message": "No checkpoint files found in case directory to restart from"}, status_code=400)
                return
            
            success = update_restart_in_inputs_dat(restart_file, restart_time)
            if not success:
                self.send_json_response({"status": "error", "message": "Failed to update inputs.dat with restart configuration"}, status_code=500)
                return
            
            # 3. Restart the run
            log_file = open(os.path.join(CASE_DIR, "out.log"), "a")
            
            if platform.system() == "Windows":
                cmd = ["wsl", "./run_case.sh", "-headless"]
            else:
                cmd = ["./run_case.sh", "-headless"]
                
            try:
                SOLVER_PROC = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=CASE_DIR,
                    shell=(platform.system() == "Windows")
                )
                self.send_json_response({
                    "status": "success", 
                    "message": f"Restarted solver successfully from t={restart_time:.3f}", 
                    "pid": SOLVER_PROC.pid,
                    "restart_file": restart_file,
                    "restart_time": restart_time
                })
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to launch solver on restart: {e}"}, status_code=500)

        # 6. Clear visualizer cache API
        elif path == "/api/clear_cache":
            global CONTOUR_CACHE, RELOAD_VERSION
            CONTOUR_CACHE.clear()
            RELOAD_VERSION = int(time.time())
            self.send_json_response({"status": "success", "message": "Contour caches cleared successfully"})
            
        else:
            self.send_error(404, "Endpoint Not Found")

    # Static File Server Helper
    def serve_static_file(self, rel_path, content_type):
        proj_root = os.path.abspath(os.path.dirname(__file__))
        full_path = os.path.join(proj_root, rel_path)
        if not os.path.exists(full_path):
            self.send_error(404, "File Not Found")
            return
            
        try:
            with open(full_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")

    def send_json_response(self, data, status_code=200):
        try:
            if status_code >= 400:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = None
                if exc_type is not None:
                    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                error_msg = data.get("message") or data.get("error") or str(data)
                write_error_log(self.path, self.command, error_msg, tb_str)

            content = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            # Add CORS headers
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.log_message(f"Error encoding JSON response: {e}")

# Run Server
def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, GUIRequestHandler)
    print(f"Server running at http://localhost:{PORT}")
    
    # Auto-open browser tab
    webbrowser.open(f"http://localhost:{PORT}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()

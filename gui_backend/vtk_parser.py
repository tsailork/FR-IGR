import os
import sys
import re
import xml.etree.ElementTree as ET
import numpy as np
from gui_backend.state import state

# Pre-compiled regular expressions for performance optimization
EXTENT_RE = re.compile(r'WholeExtent="([^"]+)"')
PIECE_POINTS_RE = re.compile(r'NumberOfPoints="(\d+)"')
PIECE_POINTS_ALT_RE = re.compile(r'<Piece[^>]+NumberOfPoints="(\d+)"')
POINTS_BLOCK_RE = re.compile(r'<Points>\s*<DataArray[^>]*>(.*?)</DataArray>\s*</Points>', re.DOTALL)

# Dict mapping lowercase frontend names to actual VTK scalar array names
VTK_VAR_MAPPING = {
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

def get_vtk_var_name(var_name):
    """Maps lowercase frontend variable name to target array name written in VTK files."""
    return VTK_VAR_MAPPING.get(var_name.lower(), var_name)

def parse_latest_vts(var_name):
    """Parses all blocks in the latest written VTM dataset file using ET."""
    var_name = get_vtk_var_name(var_name)

    # Find latest VTM path from plot.pvd or solution.pvd
    pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "plot.pvd")
    if not os.path.exists(pvd_path) or os.path.getsize(pvd_path) == 0:
        pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "solution.pvd")
    if not os.path.exists(pvd_path) or os.path.getsize(pvd_path) == 0:
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
        
        vtm_path = os.path.join(state.CASE_DIR, "pv_outputs", os.path.basename(latest_file))
        if not os.path.exists(vtm_path) or os.path.getsize(vtm_path) == 0:
            vtm_path = os.path.join(state.CASE_DIR, "pv_outputs", latest_file)
            if not os.path.exists(vtm_path) or os.path.getsize(vtm_path) == 0:
                vtm_path = os.path.join(state.CASE_DIR, latest_file)
                if not os.path.exists(vtm_path) or os.path.getsize(vtm_path) == 0:
                    return []

        # Parse VTM file to list VTS/VTU block files
        vtm_dir = os.path.dirname(vtm_path)
        vtm_tree = ET.parse(vtm_path)
        vtm_root = vtm_tree.getroot()
        datasets_vtm = vtm_root.findall(".//DataSet")
        
        blocks_data = []
        for dataset in datasets_vtm:
            block_id = int(dataset.attrib.get("index", 0))
            block_file = dataset.attrib.get("file")
            vts_path = os.path.join(vtm_dir, block_file)
            if not os.path.exists(vts_path) or os.path.getsize(vts_path) == 0:
                continue
            
            # Parse VTS/VTU file
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
    """Optimized parser utilizing regex scans to extract coordinate & value blocks without ET DOM build overhead."""
    if not os.path.exists(vts_path) or os.path.getsize(vts_path) == 0:
        return None

    var_name = get_vtk_var_name(var_name)

    try:
        with open(vts_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        is_structured = True
        extent_match = EXTENT_RE.search(content)
        if extent_match:
            extent = [int(x) for x in extent_match.group(1).split()]
            nx = extent[1] - extent[0] + 1
            ny = extent[3] - extent[2] + 1
        else:
            points_match = PIECE_POINTS_RE.search(content)
            if not points_match:
                points_match = PIECE_POINTS_ALT_RE.search(content)
            if not points_match:
                return None
            n_points = int(points_match.group(1))
            nx = n_points
            ny = 1
            is_structured = False
        
        points_block_match = POINTS_BLOCK_RE.search(content)
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
    """Fast XML parser to extract the constituent block datasets inside a VTM file."""
    if not os.path.exists(vtm_path) or os.path.getsize(vtm_path) == 0:
        return []
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

def find_latest_checkpoint_from_pvd():
    """Scans solution.pvd file to find the path and simulation time of the latest checkpoint."""
    pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "solution.pvd")
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
    """In-place updates RESTART_FILE and RESTART_TIME parameters inside inputs.dat under [IO] section."""
    inputs_path = os.path.join(state.CASE_DIR, "inputs.dat")
    if not os.path.exists(inputs_path):
        return False
    
    try:
        with open(inputs_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        current_section = None
        file_updated = False
        time_updated = False
        
        # Scan and update or filter existing keys
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1].strip()
                
            if current_section == "IO":
                if "=" in stripped:
                    key = stripped.split("=")[0].strip()
                    if key == "RESTART_FILE":
                        new_lines.append(f"RESTART_FILE = {restart_file}\n")
                        file_updated = True
                        continue
                    elif key == "RESTART_TIME":
                        new_lines.append(f"RESTART_TIME = {restart_time}\n")
                        time_updated = True
                        continue
            
            # Remove any stray restart variables outside of [IO] to prevent parsing bugs
            if "=" in stripped:
                key = stripped.split("=")[0].strip()
                if key in ["RESTART_FILE", "RESTART_TIME"] and current_section != "IO":
                    continue
                    
            new_lines.append(line)
            
        # If any variables were missing from [IO], inject them into it
        if not file_updated or not time_updated:
            final_lines = []
            current_section = None
            inserted = False
            for line in new_lines:
                final_lines.append(line)
                stripped = line.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    current_section = stripped[1:-1].strip()
                
                if current_section == "IO" and not inserted:
                    if not file_updated:
                        final_lines.append(f"RESTART_FILE = {restart_file}\n")
                    if not time_updated:
                        final_lines.append(f"RESTART_TIME = {restart_time}\n")
                    inserted = True
            new_lines = final_lines
            
        with open(inputs_path, "w") as f:
            f.writelines(new_lines)
        return True
    except Exception:
        return False

import os
import re
import json
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gui_backend.state import state
from gui_backend.vtk_parser import fast_parse_vtm, fast_parse_vts

def get_webcontour_settings(var_name):
    """Retrieves plot limits, colormaps, and mesh grid settings from case's .webcontour file."""
    path = os.path.join(state.CASE_DIR, ".webcontour")
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

def generate_contour_plot(var_name, vtm_name=None):
    """Parses VTK blocks and generates a high-quality 2D contour plot (structured/unstructured grids)."""
    vtm_path = None
    if vtm_name:
        vtm_path = os.path.join(state.CASE_DIR, "pv_outputs", vtm_name)
        if not os.path.exists(vtm_path):
            vtm_path = os.path.join(state.CASE_DIR, vtm_name)
    else:
        pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "plot.pvd")
        if not os.path.exists(pvd_path):
            pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "solution.pvd")
        if os.path.exists(pvd_path):
            try:
                with open(pvd_path, "r") as f:
                    pvd_content = f.read()
                datasets = re.findall(r'<DataSet[^>]+file=["\']([^"\']+)["\']', pvd_content)
                if datasets:
                    vtm_file = os.path.basename(datasets[-1])
                    vtm_path = os.path.join(state.CASE_DIR, "pv_outputs", vtm_file)
            except Exception:
                pass
                
    if not vtm_path or not os.path.exists(vtm_path):
        return None

    # Check cache using mtime
    try:
        mtime = os.path.getmtime(vtm_path)
    except Exception:
        mtime = 0.0
        
    webcontour_path = os.path.join(state.CASE_DIR, ".webcontour")
    webcontour_mtime = 0.0
    if os.path.exists(webcontour_path):
        try:
            webcontour_mtime = os.path.getmtime(webcontour_path)
        except Exception:
            pass
            
    mtime = max(mtime, webcontour_mtime, state.RELOAD_VERSION)
        
    cache_key = (var_name, os.path.basename(vtm_path), mtime)
    if cache_key in state.CONTOUR_CACHE:
        return state.CONTOUR_CACHE[cache_key]

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
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    
    # Store in cache
    state.CONTOUR_CACHE[cache_key] = png_bytes
    return png_bytes

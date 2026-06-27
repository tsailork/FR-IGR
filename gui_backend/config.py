import os

def parse_config_file(filepath):
    """Parses C++ style ini config files (.dat, .grid) into structured dictionaries."""
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

def write_config_file(filepath, config_dict):
    """Writes a dictionary back to C++ style ini config files."""
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

def validate_config(inputs_dict, domain_dict):
    """Validates parameters, element resolutions, and boundary condition interface symmetry."""
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

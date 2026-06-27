import os
import time
from gui_backend.state import state

def write_error_log(path, method, error_msg, tb_str=None):
    """Safely logs traceback and HTTP handler errors to a file inside the case directory."""
    try:
        log_path = os.path.join(state.CASE_DIR, "gui_error.log")
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

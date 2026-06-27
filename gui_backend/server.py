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

from gui_backend.state import state
from gui_backend.logging import write_error_log
from gui_backend.config import parse_config_file, write_config_file, validate_config
from gui_backend.vtk_parser import (
    parse_latest_vts,
    find_latest_checkpoint_from_pvd,
    update_restart_in_inputs_dat
)
from gui_backend.contour import generate_contour_plot

def check_solver_running():
    """Checks if the CFD C++ solver process is active locally or under WSL."""
    # 1. Check Python subprocess handle
    if state.SOLVER_PROC is not None:
        if state.SOLVER_PROC.poll() is None:
            return True
        else:
            state.SOLVER_PROC = None

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


class GUIRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress default HTTP access logging console spam
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
        
        # Clean path and prevent directory traversal
        parts = path.lstrip('/').split('/')
        if any(p == '..' for p in parts):
            self.send_error(403, "Access Denied")
            return
            
        # 1. Static file routing
        if path in ["", "/", "/index.html"]:
            self.serve_static_file("gui/index.html", "text/html")
            return
            
        # Check if the requested file exists in the 'gui' subdirectory
        rel_path = os.path.join("gui", *parts)
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        full_path = os.path.normpath(os.path.join(proj_root, rel_path))
        
        # Verify it remains inside the gui directory
        gui_dir = os.path.normpath(os.path.join(proj_root, "gui"))
        if os.path.exists(full_path) and os.path.commonpath([gui_dir, full_path]) == gui_dir and os.path.isfile(full_path):
            # Map extensions to standard MIME types
            _, ext = os.path.splitext(full_path)
            mime_types = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".json": "application/json",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".ico": "image/x-icon",
                ".svg": "image/svg+xml"
            }
            content_type = mime_types.get(ext.lower(), "application/octet-stream")
            self.serve_static_file(rel_path, content_type)
            return

        # 2. Config reading API
        if path == "/api/config":
            inputs = parse_config_file(os.path.join(state.CASE_DIR, "inputs.dat"))
            domain = parse_config_file(os.path.join(state.CASE_DIR, "domain.grid"))
            payload = {"inputs": inputs, "domain": domain}
            self.send_json_response(payload)
            
        # 3. Status and logs API
        elif path == "/api/status":
            running = check_solver_running()
            
            # Read last 150 lines of logs
            logs = []
            log_path = os.path.join(state.CASE_DIR, "out.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        logs = [l.rstrip() for l in lines[-150:]]
                except Exception as e:
                    logs = [f"Error reading log file: {e}"]
                    
            # Parse residuals
            residuals = []
            res_path = os.path.join(state.CASE_DIR, "csv_outputs", "residuals.csv")
            if os.path.exists(res_path):
                try:
                    with open(res_path, "r") as f:
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
            residuals = residuals[-300:]

            # Parse probe data
            probes = []
            probe_path = os.path.join(state.CASE_DIR, "csv_outputs", "probe.csv")
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
            pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "plot.pvd")
            if not os.path.exists(pvd_path):
                pvd_path = os.path.join(state.CASE_DIR, "pv_outputs", "solution.pvd")
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
                        t_val = float(m[0])
                        vtm_file = os.path.basename(m[1])
                    except ValueError:
                        try:
                            t_val = float(m[1])
                            vtm_file = os.path.basename(m[0])
                        except ValueError:
                            continue
                    
                    if t_val not in seen_times:
                        vtm_path = os.path.join(state.CASE_DIR, "pv_outputs", vtm_file)
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
                write_config_file(os.path.join(state.CASE_DIR, "inputs.dat"), inputs)
            if domain:
                write_config_file(os.path.join(state.CASE_DIR, "domain.grid"), domain)
                
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
            
            dotfile_path = os.path.join(state.CASE_DIR, ".webcontour")
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
            
            log_file = open(os.path.join(state.CASE_DIR, "out.log"), "w" if clean else "a")
            
            # Setup process invocation (supports Windows/WSL and native Linux)
            if platform.system() == "Windows":
                cmd = ["wsl", "./run_case.sh", "-headless"]
            else:
                cmd = ["./run_case.sh", "-headless"]
                
            if clean:
                update_restart_in_inputs_dat("", 0.0)
                cmd.append("-clean")
                
            try:
                state.SOLVER_PROC = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=state.CASE_DIR,
                    shell=(platform.system() == "Windows")
                )
                self.send_json_response({"status": "success", "pid": state.SOLVER_PROC.pid})
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to launch solver: {e}"}, status_code=500)

        # 3. Stop simulation API
        elif path == "/api/stop":
            stop_file = os.path.join(state.CASE_DIR, "STOP")
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
                pv_dir = os.path.join(state.CASE_DIR, "pv_outputs")
                csv_dir = os.path.join(state.CASE_DIR, "csv_outputs")
                
                if os.path.exists(pv_dir):
                    for f in os.listdir(pv_dir):
                        os.remove(os.path.join(pv_dir, f))
                if os.path.exists(csv_dir):
                    for f in os.listdir(csv_dir):
                        os.remove(os.path.join(csv_dir, f))
                        
                log_path = os.path.join(state.CASE_DIR, "out.log")
                if os.path.exists(log_path):
                    os.remove(log_path)
                    
                stop_path = os.path.join(state.CASE_DIR, "STOP")
                if os.path.exists(stop_path):
                    os.remove(stop_path)
                    
                residuals_path = os.path.join(state.CASE_DIR, "residuals.dat")
                if os.path.exists(residuals_path):
                    os.remove(residuals_path)
                    
                update_restart_in_inputs_dat("", 0.0)
                self.send_json_response({"status": "success", "message": "Outputs cleaned successfully"})
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to clean outputs: {e}"}, status_code=500)
                
        # 5. Restart simulation API
        elif path == "/api/restart":
            if check_solver_running():
                # Write STOP file
                stop_file = os.path.join(state.CASE_DIR, "STOP")
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
                    try:
                        if state.SOLVER_PROC:
                            state.SOLVER_PROC.terminate()
                            state.SOLVER_PROC.wait(timeout=2.0)
                    except Exception:
                        pass
            
            # Modify input file to point to last solution/plot file and associated time
            restart_file, restart_time = find_latest_checkpoint_from_pvd()
            if not restart_file:
                self.send_json_response({"status": "error", "message": "No checkpoint files found in case directory to restart from"}, status_code=400)
                return
            
            success = update_restart_in_inputs_dat(restart_file, restart_time)
            if not success:
                self.send_json_response({"status": "error", "message": "Failed to update inputs.dat with restart configuration"}, status_code=500)
                return
            
            # Restart the run
            log_file = open(os.path.join(state.CASE_DIR, "out.log"), "a")
            
            if platform.system() == "Windows":
                cmd = ["wsl", "./run_case.sh", "-headless"]
            else:
                cmd = ["./run_case.sh", "-headless"]
                
            try:
                state.SOLVER_PROC = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=state.CASE_DIR,
                    shell=(platform.system() == "Windows")
                )
                self.send_json_response({
                    "status": "success", 
                    "message": f"Restarted solver successfully from t={restart_time:.3f}", 
                    "pid": state.SOLVER_PROC.pid,
                    "restart_file": restart_file,
                    "restart_time": restart_time
                })
            except Exception as e:
                self.send_json_response({"status": "error", "message": f"Failed to launch solver on restart: {e}"}, status_code=500)

        # 6. Clear visualizer cache API
        elif path == "/api/clear_cache":
            state.CONTOUR_CACHE.clear()
            state.RELOAD_VERSION = int(time.time())
            self.send_json_response({"status": "success", "message": "Contour caches cleared successfully"})
            
        else:
            self.send_error(404, "Endpoint Not Found")

    def serve_static_file(self, rel_path, content_type):
        """Locates and transmits a static file from the gui folder."""
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        """Sends data as a JSON HTTP response with CORS headers."""
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
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.log_message(f"Error encoding JSON response: {e}")


def run_server():
    """Launches the backend HTTP server and opens a local browser tab."""
    server_address = ("", state.PORT)
    httpd = HTTPServer(server_address, GUIRequestHandler)
    print(f"FR-IGR Case Setup GUI starting...")
    print(f"Target Case Directory: {state.CASE_DIR}")
    print(f"Server running at http://localhost:{state.PORT}")
    
    # Auto-open browser tab
    webbrowser.open(f"http://localhost:{state.PORT}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

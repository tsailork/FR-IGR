import os
import sys

class AppState:
    """Class to hold shared backend state variables."""
    def __init__(self):
        self.CASE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "cases/default_case")
        self.PORT = 8080
        self.SOLVER_PROC = None
        self.RELOAD_VERSION = 0
        self.CONTOUR_CACHE = {}

state = AppState()

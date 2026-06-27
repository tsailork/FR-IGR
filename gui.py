#!/usr/bin/env python3
"""FR-IGR Solver Case Setup GUI Entry Point.

This script imports and launches the modularized server package from `gui_backend`.
"""
import sys
from gui_backend.server import run_server

if __name__ == "__main__":
    run_server()

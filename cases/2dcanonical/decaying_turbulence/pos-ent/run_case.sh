#!/bin/bash
set -e

PROJECT_DIR="/home/tsk/Documents/GitHub/FR-IGR/"

# Determine active running directory (the case directory)
#CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(pwd)"
cd "$CASE_DIR"

# Parse arguments
CLEAN=false
HEADLESS=false
ATTACH=false

for arg in "$@"; do
    if [ "$arg" = "-clean" ]; then
        CLEAN=true
    fi
    if [ "$arg" = "-headless" ]; then
        HEADLESS=true
    fi
    if [ "$arg" = "-attach" ] || [ "$arg" = "--attach" ]; then
        ATTACH=true
    fi
done

# Clean local case outputs if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning local case outputs..."
    rm -rf pv_outputs/*
    rm -rf csv_outputs/*
    rm -f out.log STOP residuals.dat
fi

# Extract NUM_THREADS from local inputs.dat if it exists
if [ -f "inputs.dat" ]; then
    if grep -q "NUM_THREADS" inputs.dat; then
        THREADS=$(grep "NUM_THREADS" inputs.dat | tr -d ' ' | cut -d'=' -f2)
    else
        THREADS=1
    fi
else
    echo "Error: inputs.dat not found in case directory $CASE_DIR"
    exit 1
fi

export OMP_NUM_THREADS=$THREADS
echo "Setting OMP_NUM_THREADS=$THREADS"

# Build solver in project directory
echo "Ensuring FR-IGR Solver is built..."
make -C "$PROJECT_DIR" -j12 all

# Run based on interface mode
if [ "$TUI_ACTIVE" = "1" ] || [ "$HEADLESS" = true ]; then
    if [ "$HEADLESS" = true ]; then
        echo "Running Simulation (Headless)..."
        echo "STOP FILE PROTOCOL: To stop this headless run gracefully, create a file named 'STOP' in: $CASE_DIR"
    else
        echo "Running Simulation..."
    fi
    "$PROJECT_DIR/bin/fr_solver"
elif [ "$ATTACH" = true ]; then
    echo "Attaching TUI Monitor to running simulation..."
    python3 "$PROJECT_DIR/tui.py" --attach
else
    echo "Launching Runtime Monitor TUI..."
    python3 "$PROJECT_DIR/tui.py"
fi

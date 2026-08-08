#!/bin/bash
set -e

PROJECT_DIR="/home/tsk/Documents/GitHub/FR-IGR/"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CASE_DIR"

CLEAN=false
HEADLESS=false
ATTACH=false

for arg in "$@"; do
    if [ "$arg" = "-clean" ]; then CLEAN=true; fi
    if [ "$arg" = "-headless" ]; then HEADLESS=true; fi
    if [ "$arg" = "-attach" ] || [ "$arg" = "--attach" ]; then ATTACH=true; fi
done

if [ "$CLEAN" = true ]; then
    rm -rf pv_outputs/* csv_outputs/* out.log STOP residuals.dat
fi

THREADS=$(grep "NUM_THREADS" inputs.dat | tr -d ' ' | cut -d'=' -f2 || echo 4)
export OMP_NUM_THREADS=$THREADS

make -C "$PROJECT_DIR" -j12 all

if [ "$TUI_ACTIVE" = "1" ] || [ "$HEADLESS" = true ]; then
    "$PROJECT_DIR/bin/fr_solver"
elif [ "$ATTACH" = true ]; then
    python3 "$PROJECT_DIR/tui.py" --attach
else
    python3 "$PROJECT_DIR/tui.py"
fi

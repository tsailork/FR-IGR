#!/bin/bash
set -e

PROJECT_DIR="/home/tsk/Documents/GitHub/FR-IGR/"
CASE_DIR="$(pwd)"
cd "$CASE_DIR"

CLEAN=false

for arg in "$@"; do
    if [ "$arg" = "-clean" ]; then CLEAN=true; fi
done

if [ "$CLEAN" = true ]; then
    echo "Cleaning local case outputs..."
    rm -rf pv_outputs/* csv_outputs/* out.log STOP
fi

if [ -f "inputs.dat" ]; then
    THREADS=$(grep "NUM_THREADS" inputs.dat | tr -d ' ' | cut -d'=' -f2)
else
    THREADS=4
fi

export OMP_NUM_THREADS=$THREADS
echo "Setting OMP_NUM_THREADS=$THREADS"

echo "Ensuring FR-IGR Solver is built..."
make -C "$PROJECT_DIR" -j8 all

echo "Executing 3D ABC Flow Case..."
"$PROJECT_DIR/bin/fr_solver"

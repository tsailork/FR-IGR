#!/bin/bash
set -e

# Extract NUM_THREADS from inputs.dat if it exists, otherwise default to 1
if grep -q "NUM_THREADS" inputs.dat; then
    THREADS=$(grep "NUM_THREADS" inputs.dat | tr -d ' ' | cut -d'=' -f2)
else
    THREADS=1
fi

echo "Setting OMP_NUM_THREADS=$THREADS"
export OMP_NUM_THREADS=$THREADS

echo "Building FR-IGR Solver..."
make clean
make -j12 all

echo "Running Simulation..."
./fr_solver

# Check if plot script exists and run it
if [ -f "plot2d_pv.py" ]; then
    echo "Running visualization script..."
    python3 plot2d.py
else
    echo "To visualize, open pv_outputs/solution.pvd in ParaView."
fi

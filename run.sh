#!/bin/bash
set -e

# Extract NUM_THREADS from inputs.dat if it exists, otherwise default to 1
if grep -q "NUM_THREADS" inputs.dat; then
    THREADS=$(grep "NUM_THREADS" inputs.dat | tr -d ' ' | cut -d'=' -f2)
else
    THREADS=1
fi

# Check for -clean argument
CLEAN=false
for arg in "$@"; do
    if [ "$arg" = "-clean" ]; then
        CLEAN=true
    fi
done

if [ "$TUI_ACTIVE" = "1" ]; then
    echo "Setting OMP_NUM_THREADS=$THREADS"
    export OMP_NUM_THREADS=$THREADS

    echo "Building FR-IGR Solver..."
    if [ "$CLEAN" = true ]; then
        echo "Running make cleanall..."
        make cleanall
    fi
    make -j12 all

    echo "Running Simulation..."
    ./bin/fr_solver
else
    echo "Setting OMP_NUM_THREADS=$THREADS"
    export OMP_NUM_THREADS=$THREADS

    echo "Building FR-IGR Solver..."
    if [ "$CLEAN" = true ]; then
        echo "Running make cleanall..."
        make cleanall
    fi
    make -j12 all

    HEADLESS=false
    ATTACH=false
    for arg in "$@"; do
        if [ "$arg" = "-headless" ]; then
            HEADLESS=true
        fi
        if [ "$arg" = "-attach" ] || [ "$arg" = "--attach" ]; then
            ATTACH=true
        fi
    done

    if [ "$HEADLESS" = true ]; then
        echo "Running Simulation (Headless)..."
        ./bin/fr_solver
        
        # Check if plot script exists and run it
        if [ -f "plot2d_pv.py" ]; then
            echo "Running visualization script..."
            python3 plot2d.py
        else
            echo "To visualize, open pv_outputs/solution.pvd in ParaView."
        fi
    elif [ "$ATTACH" = true ]; then
        echo "Attaching TUI Monitor to running simulation..."
        python3 tui.py --attach
    else
        echo "Launching Runtime Monitor TUI..."
        python3 tui.py
    fi
fi

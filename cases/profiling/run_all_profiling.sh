#!/bin/bash
set -e

PROJECT_DIR="/home/tsk/Documents/GitHub/FR-IGR"
SOLVER_BIN="$PROJECT_DIR/bin/fr_solver"

echo "=========================================================================="
echo "          FR-IGR MASTER 2D PROFILING SUITE EXECUTION"
echo "=========================================================================="

cd "$PROJECT_DIR"
make -j18 all

PROFILING_DIR="$PROJECT_DIR/cases/profiling"
CASES=(
    "01_euler_vortex"
    "02_viscous_couette"
    "03_shock_vortex_pcg"
    "04_amr_cylinder"
    "05_blast"
)

printf "\n%-30s | %-12s | %-10s\n" "Profiling Case" "Status" "Runtime (s)"
echo "--------------------------------------------------------------------------"

TOTAL_START=$(date +%s.%N)

for c in "${CASES[@]}"; do
    CASE_PATH="$PROFILING_DIR/$c"
    if [ ! -d "$CASE_PATH" ]; then
        echo "Warning: Case directory $c not found, skipping..."
        continue
    fi

    cd "$CASE_PATH"
    rm -rf pv_outputs csv_outputs out.log STOP residuals.dat
    mkdir -p pv_outputs csv_outputs

    START_TIME=$(date +%s.%N)
    "$SOLVER_BIN" > out.log 2>&1 || { echo "FAILED"; exit 1; }
    END_TIME=$(date +%s.%N)

    ELAPSED=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")
    printf "%-30s | %-12s | %-10s\n" "$c" "PASSED ✅" "${ELAPSED}s"
done

TOTAL_END=$(date +%s.%N)
TOTAL_ELAPSED=$(awk "BEGIN {printf \"%.2f\", $TOTAL_END - $TOTAL_START}")

echo "--------------------------------------------------------------------------"
echo " Total Profiling Suite Runtime: ${TOTAL_ELAPSED}s"
echo "=========================================================================="

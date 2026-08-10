#!/bin/bash
# run_tests.sh

MODE=${1:-all}

run_unit() {
    echo "Building and running unit tests..."
    make test-unit
    if [ $? -ne 0 ]; then
        echo "Unit tests failed!"
        exit 1
    fi
}

run_regression() {
    echo "Building and running regression tests..."
    make test-regression
    if [ $? -ne 0 ]; then
        echo "Regression tests failed!"
        exit 1
    fi
}

case $MODE in
    "unit")
        run_unit
        ;;
    "regression")
        run_regression
        ;;
    "all")
        run_unit
        run_regression
        ;;
    *)
        echo "Unknown mode: $MODE. Use unit, regression, or all."
        exit 1
        ;;
esac

echo "All tests passed successfully!"
exit 0

#!/bin/bash
# Usage: ./benchmark.sh <logfile>
# Runs serial, MPI-only, CUDA-only, and hybrid MPI+CUDA, and records
# their wall-clock times into results.csv for speedup/efficiency analysis.

set -e
LOGFILE=${1:-sample.log}
OUT=results.csv
NP_LIST=(1 2 4 8)

echo "variant,nprocs,time_sec" > "$OUT"

echo ">>> Serial (CPU baseline)"
./log_analyzer_serial "$LOGFILE" | grep RESULT | sed 's/RESULT,//' >> "$OUT"

echo ">>> CUDA-only (single GPU)"
./log_analyzer_cuda "$LOGFILE" | grep RESULT | sed 's/RESULT,//' >> "$OUT"

echo ">>> MPI-only (CPU, varying process count)"
for np in "${NP_LIST[@]}"; do
    mpirun -np "$np" ./log_analyzer_mpi "$LOGFILE" | grep RESULT | sed 's/RESULT,//' >> "$OUT"
done

echo ">>> Hybrid MPI + CUDA (varying process count)"
for np in "${NP_LIST[@]}"; do
    mpirun -np "$np" ./log_analyzer_hybrid "$LOGFILE" | grep RESULT | sed 's/RESULT,//' >> "$OUT"
done

echo ""
echo "Done. Raw timings written to $OUT"
echo "Run: python3 analyze_results.py"

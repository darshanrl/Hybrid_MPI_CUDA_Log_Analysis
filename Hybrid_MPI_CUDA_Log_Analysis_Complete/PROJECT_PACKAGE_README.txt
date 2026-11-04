HYBRID MPI-CUDA LOG ANALYSIS - COMPLETE PROJECT PACKAGE

This package contains:
- All source code for Serial, MPI-only, CUDA-only and Hybrid MPI-CUDA implementations
- Headers and Makefile
- Synthetic log generator
- Benchmarking script
- Results analysis script
- Combined benchmark results CSV
- Speedup and execution-time graphs
- Project documentation and full report
- Demonstration/viva handbook
- Final PowerPoint presentation with graphs

Main source files:
  src/serial_main.c       Serial CPU implementation
  src/mpi_only_main.c     MPI implementation
  src/cuda_only_main.c    CUDA-only implementation
  src/main.c              Hybrid MPI-CUDA implementation
  src/cpu_analyze.c       CPU log analysis logic
  src/kernel.cu           CUDA kernel and GPU analysis

Build:
  make clean
  make

Example runs:
  ./log_analyzer_serial logs_2GB.log
  ./log_analyzer_cuda logs_2GB.log
  mpirun --oversubscribe -np 4 ./log_analyzer_mpi logs_2GB.log
  mpirun --oversubscribe -np 4 ./log_analyzer_hybrid logs_2GB.log

Generate graphs from the combined results:
  python3 analyze_results.py

Note: large raw log datasets are not bundled because they are multi-hundred-MB/GB files. They can be regenerated with generate_logs.py using the commands in logs/README.txt.

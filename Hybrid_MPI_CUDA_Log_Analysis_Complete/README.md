# Hybrid MPI-CUDA Framework for Large-Scale Log Analysis

A mini-project demonstrating **two-level parallelism**:
- **MPI** splits a huge log file into byte-range chunks across processes (can span multiple nodes).
- **CUDA** parallelizes the actual text scanning of each chunk across thousands of GPU threads.

Each MPI rank reads its own slice of the log file, hands it to the GPU, gets back
local counts (total lines + occurrences of `ERROR`/`WARN`/`INFO`/`DEBUG`), and rank 0
combines everything with `MPI_Reduce` into one final report.

## 1. Project layout
```
hybrid_mpi_cuda_log_analysis/
├── Makefile
├── generate_logs.py
├── include/kernel.h
├── src/kernel.cu     <- CUDA kernel + host wrapper (GPU side)
└── src/main.c        <- MPI driver (CPU side)
```

## 2. Prerequisites
You need a machine (or cluster) with:
- An NVIDIA GPU + CUDA toolkit (`nvcc --version` to check)
- An MPI implementation, e.g. OpenMPI (`mpicc --version` to check)
- Python 3 (only for generating test data)

Install on Ubuntu, if missing:
```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev
# CUDA toolkit: follow NVIDIA's instructions for your GPU/driver version
# https://developer.nvidia.com/cuda-downloads
```

## 3. Generate test data
```bash
python3 generate_logs.py 5000000 sample.log   # 5 million lines, ~350 MB
```

## 4. Build
```bash
# Edit the ARCH variable in the Makefile to match your GPU's compute capability
# (check with: nvidia-smi --query-gpu=compute_cap --format=csv)
make
```
This compiles `src/kernel.cu` with `nvcc` and `src/main.c` with `mpicc`, then
links both object files together into `log_analyzer`.

## 5. Run
Single node, 4 MPI processes:
```bash
mpirun -np 4 ./log_analyzer sample.log
```

Across multiple nodes (each with its own GPU), using a hostfile:
```bash
mpirun --hostfile hosts.txt -np 8 ./log_analyzer sample.log
```
`hosts.txt` example:
```
node1 slots=4
node2 slots=4
```

## 6. Sample output
```
[Rank 0] byte range [0, 87654321)  lines=1250012
[Rank 1] byte range [87654321, 175308642)  lines=1249998
...

===== Global Log Analysis Report =====
File                 : sample.log
File size            : 350000000 bytes
MPI processes        : 4
Total lines processed: 5000000
  ERROR : 249873
  WARN  : 749651
  INFO  : 2999872
  DEBUG : 1000604
Total wall time      : 0.842 sec
```

## 7. How it works (for your report)

**Data partitioning (MPI).** The file size is divided evenly by rank count.
Each rank's start/end offsets are snapped to line boundaries (scanning forward
to the next `\n`) so no line is split between two ranks and no line is double
counted.

**GPU scanning (CUDA).** One CUDA thread is launched per *byte* of a rank's
chunk. A thread only does real work if its byte is the first character of a
line; it then checks whether that line starts with any of the tracked
keywords, using `atomicAdd` to accumulate counts safely across all threads.
This avoids a serial pass over the buffer entirely — the GPU sees the whole
chunk at once.

**Reduction (MPI).** Each rank's local totals are combined with
`MPI_Reduce(..., MPI_SUM, root=0, ...)`, so only rank 0 needs to print the
final combined report.

## 8. Ideas to extend it (good for bonus marks)
- Track top-N most frequent IP addresses or status codes instead of just level keywords.
- Bucket counts by hour/minute for a time-series view (extract timestamp on GPU).
- Use `cudaStream`s + pinned host memory to overlap file I/O with GPU compute.
- Benchmark: plot speedup vs. number of MPI ranks, and vs. CPU-only baseline (a version of `analyzeChunkGPU` implemented as a plain C loop) to show the GPU's contribution.
- Add MPI-IO (`MPI_File_read_at`) instead of each rank doing its own `fopen`/`fseek`, which scales better on real parallel filesystems.

## 9. Report skeleton
1. **Abstract** — problem, approach, key result (e.g. "Xx speedup over serial baseline on N GB of logs")
2. **Introduction** — why log analysis at scale is hard (I/O + CPU bound)
3. **System architecture** — diagram: log file → MPI rank partitioning → per-rank GPU kernel → MPI_Reduce → report
4. **Implementation** — summarize what's in section 7 above
5. **Experimental setup** — hardware, dataset sizes, keyword set
6. **Results** — table/graph of runtime vs. (ranks × GPUs), vs. file size
7. **Conclusion & future work** — pull from section 8

MPICC      = mpicc
CC         = gcc
NVCC       = nvcc
CUDA_PATH  ?= /usr/local/cuda

# Change -arch to match your GPU's compute capability
# (sm_60 Pascal, sm_70 Volta, sm_75 Turing, sm_80/86 Ampere, sm_89 Ada, sm_90 Hopper)
ARCH       = sm_70

CFLAGS     = -O2 -Iinclude -I$(CUDA_PATH)/include
NVCCFLAGS  = -O2 -Iinclude -arch=$(ARCH)
LDFLAGS    = -L$(CUDA_PATH)/lib64 -lcudart

all: log_analyzer_serial log_analyzer_mpi log_analyzer_cuda log_analyzer_hybrid

# ---- 1. Serial baseline (no MPI, no CUDA) ----
log_analyzer_serial: src/serial_main.c src/cpu_analyze.c
	$(CC) $(CFLAGS) src/serial_main.c src/cpu_analyze.c -o log_analyzer_serial

# ---- 2. MPI-only (CPU scan, MPI-distributed) ----
log_analyzer_mpi: src/mpi_only_main.c src/cpu_analyze.c
	$(MPICC) $(CFLAGS) src/mpi_only_main.c src/cpu_analyze.c -o log_analyzer_mpi

# ---- 3. CUDA-only (single GPU, no MPI) ----
kernel.o: src/kernel.cu include/kernel.h
	$(NVCC) $(NVCCFLAGS) -c src/kernel.cu -o kernel.o

log_analyzer_cuda: src/cuda_only_main.c kernel.o
	$(CC) $(CFLAGS) src/cuda_only_main.c kernel.o -o log_analyzer_cuda $(LDFLAGS)

# ---- 4. Hybrid MPI + CUDA ----
main_hybrid.o: src/main.c include/kernel.h
	$(MPICC) $(CFLAGS) -c src/main.c -o main_hybrid.o

log_analyzer_hybrid: main_hybrid.o kernel.o
	$(MPICC) main_hybrid.o kernel.o -o log_analyzer_hybrid $(LDFLAGS)

clean:
	rm -f *.o log_analyzer_serial log_analyzer_mpi log_analyzer_cuda log_analyzer_hybrid results.csv speedup_report.png

.PHONY: all clean

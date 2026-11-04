#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include "kernel.h"

#define MAX_KW_LEN 8
#define THREADS_PER_BLOCK 256

// Keyword table lives in constant memory for fast, cached access by all threads.
__constant__ char d_keywords[NUM_KEYWORDS][MAX_KW_LEN];
__constant__ int  d_kwLen[NUM_KEYWORDS];

// One thread is launched per byte of the chunk. A thread only does work if
// its byte is the FIRST character of a line (idx == 0 or previous char == '\n').
// This keeps the kernel embarrassingly parallel with almost no branching cost
// for the vast majority of threads.
__global__ void analyzeKernel(const char* data, size_t size,
                               unsigned long long* lineCount,
                               unsigned long long* keywordCounts)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    bool isLineStart = (idx == 0) || (data[idx - 1] == '\n');
    if (!isLineStart) return;

    atomicAdd(lineCount, 1ULL);

    #pragma unroll
    for (int k = 0; k < NUM_KEYWORDS; k++) {
        int len = d_kwLen[k];
        bool match = true;
        for (int j = 0; j < len; j++) {
            if (idx + j >= size || data[idx + j] != d_keywords[k][j]) {
                match = false;
                break;
            }
        }
        if (match) {
            atomicAdd(&keywordCounts[k], 1ULL);
        }
    }
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

extern "C" void analyzeChunkGPU(const char* h_data, size_t size,
                                 unsigned long long* h_lineCount,
                                 unsigned long long* h_keywordCounts)
{
    static const char* keywordStrs[NUM_KEYWORDS] = {"ERROR", "WARN", "INFO", "DEBUG"};

    char h_kwArray[NUM_KEYWORDS][MAX_KW_LEN];
    int  h_kwLen[NUM_KEYWORDS];
    memset(h_kwArray, 0, sizeof(h_kwArray));
    for (int i = 0; i < NUM_KEYWORDS; i++) {
        h_kwLen[i] = (int)strlen(keywordStrs[i]);
        strncpy(h_kwArray[i], keywordStrs[i], MAX_KW_LEN - 1);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_keywords, h_kwArray, sizeof(h_kwArray)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kwLen, h_kwLen, sizeof(h_kwLen)));

    if (size == 0) {
        *h_lineCount = 0;
        memset(h_keywordCounts, 0, NUM_KEYWORDS * sizeof(unsigned long long));
        return;
    }

    char* d_data = nullptr;
    unsigned long long* d_lineCount = nullptr;
    unsigned long long* d_keywordCounts = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_data, size));
    CUDA_CHECK(cudaMalloc((void**)&d_lineCount, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_keywordCounts, NUM_KEYWORDS * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_lineCount, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_keywordCounts, 0, NUM_KEYWORDS * sizeof(unsigned long long)));

    int blocks = (int)((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    analyzeKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, size, d_lineCount, d_keywordCounts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_lineCount, d_lineCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_keywordCounts, d_keywordCounts, NUM_KEYWORDS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_lineCount);
    cudaFree(d_keywordCounts);
}

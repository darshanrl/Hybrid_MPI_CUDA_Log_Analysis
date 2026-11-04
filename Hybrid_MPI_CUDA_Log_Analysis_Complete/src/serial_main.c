#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_analyze.h"

static const char* KEYWORD_NAMES[NUM_KEYWORDS] = {"ERROR", "WARN", "INFO", "DEBUG"};

// Pure serial baseline: one process, one thread, reads the WHOLE file and
// scans it in a single loop. This is the reference point every speedup
// number (MPI-only, CUDA-only, hybrid) is measured against.
int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <logfile>\n", argv[0]);
        return 1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open file %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = (char*)malloc((size_t)fileSize);
    size_t readBytes = fread(buffer, 1, (size_t)fileSize, f);
    fclose(f);
    (void)readBytes;

    unsigned long long lineCount = 0;
    unsigned long long keywordCounts[NUM_KEYWORDS] = {0};
    analyzeChunkCPU(buffer, (size_t)fileSize, &lineCount, keywordCounts);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("===== Serial (CPU-only) Report =====\n");
    printf("File size            : %ld bytes\n", fileSize);
    printf("Total lines processed: %llu\n", lineCount);
    for (int i = 0; i < NUM_KEYWORDS; i++)
        printf("  %-6s: %llu\n", KEYWORD_NAMES[i], keywordCounts[i]);
    printf("Total wall time      : %.4f sec\n", elapsed);

    // Machine-readable line for the benchmark script
    printf("RESULT,serial,1,%.4f\n", elapsed);

    free(buffer);
    return 0;
}

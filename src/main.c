#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"

static const char* KEYWORD_NAMES[NUM_KEYWORDS] = {"ERROR", "WARN", "INFO", "DEBUG"};

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <logfile>\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double t_start = MPI_Wtime();
    const char* filename = argv[1];

    FILE* f = fopen(filename, "rb");
    if (!f) {
        if (rank == 0) fprintf(stderr, "Cannot open file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);

    // Divide the file into nprocs roughly-equal byte ranges.
    long chunkSize = fileSize / nprocs;
    long start = (long)rank * chunkSize;
    long end = (rank == nprocs - 1) ? fileSize : start + chunkSize;

    // Shift start forward to the next full line so we never read a partial
    // line that another rank already owns the front half of.
    if (rank != 0) {
        fseek(f, start, SEEK_SET);
        int c;
        while ((c = fgetc(f)) != EOF && c != '\n') start++;
        start++;
    }

    // Extend end to the end of whatever line it lands inside, so this rank
    // reads that line completely instead of truncating it.
    fseek(f, end, SEEK_SET);
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') end++;
    if (c != EOF) end++;
    if (end > fileSize) end = fileSize;

    long localSize = end - start;
    if (localSize < 0) localSize = 0;

    char* buffer = (char*)malloc((size_t)localSize);
    fseek(f, start, SEEK_SET);
    size_t readBytes = fread(buffer, 1, (size_t)localSize, f);
    fclose(f);
    (void)readBytes;

    unsigned long long localLineCount = 0;
    unsigned long long localKeywordCounts[NUM_KEYWORDS] = {0};

    // ---- GPU work happens here ----
    analyzeChunkGPU(buffer, (size_t)localSize, &localLineCount, localKeywordCounts);

    printf("[Rank %d] byte range [%ld, %ld)  lines=%llu\n",
           rank, start, end, localLineCount);
    fflush(stdout);

    unsigned long long globalLineCount = 0;
    unsigned long long globalKeywordCounts[NUM_KEYWORDS] = {0};

    MPI_Reduce(&localLineCount, &globalLineCount, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localKeywordCounts, globalKeywordCounts, NUM_KEYWORDS,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("\n===== Global Log Analysis Report =====\n");
        printf("File                 : %s\n", filename);
        printf("File size            : %ld bytes\n", fileSize);
        printf("MPI processes        : %d\n", nprocs);
        printf("Total lines processed: %llu\n", globalLineCount);
        for (int i = 0; i < NUM_KEYWORDS; i++) {
            printf("  %-6s: %llu\n", KEYWORD_NAMES[i], globalKeywordCounts[i]);
        }
        printf("Total wall time      : %.4f sec\n", t_end - t_start);
        printf("RESULT,hybrid,%d,%.4f\n", nprocs, t_end - t_start);
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}

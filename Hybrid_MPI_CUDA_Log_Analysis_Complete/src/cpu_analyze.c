#include "cpu_analyze.h"
#include <string.h>

static const char* KEYWORDS[NUM_KEYWORDS] = {"ERROR", "WARN", "INFO", "DEBUG"};

// Identical logic to the GPU kernel (analyzeKernel in kernel.cu), but run
// serially in a single loop on the CPU. Used as the baseline for speedup
// comparisons and inside the MPI-only variant (MPI parallelism, no GPU).
void analyzeChunkCPU(const char* data, size_t size,
                      unsigned long long* lineCount,
                      unsigned long long* keywordCounts)
{
    *lineCount = 0;
    for (int k = 0; k < NUM_KEYWORDS; k++) keywordCounts[k] = 0;

    for (size_t idx = 0; idx < size; idx++) {
        int isLineStart = (idx == 0) || (data[idx - 1] == '\n');
        if (!isLineStart) continue;

        (*lineCount)++;

        for (int k = 0; k < NUM_KEYWORDS; k++) {
            size_t len = strlen(KEYWORDS[k]);
            int match = 1;
            for (size_t j = 0; j < len; j++) {
                if (idx + j >= size || data[idx + j] != KEYWORDS[k][j]) {
                    match = 0;
                    break;
                }
            }
            if (match) keywordCounts[k]++;
        }
    }
}

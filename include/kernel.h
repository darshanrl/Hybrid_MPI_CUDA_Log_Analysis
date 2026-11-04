#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>

#define NUM_KEYWORDS 4

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Runs on the GPU: scans h_data (a raw chunk of log text, size bytes)
 * and counts total lines + occurrences of each keyword at line starts.
 * h_keywordCounts must point to an array of NUM_KEYWORDS unsigned long long.
 */
void analyzeChunkGPU(const char* h_data, size_t size,
                      unsigned long long* h_lineCount,
                      unsigned long long* h_keywordCounts);

#ifdef __cplusplus
}
#endif

#endif /* KERNEL_H */

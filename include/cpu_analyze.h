#ifndef CPU_ANALYZE_H
#define CPU_ANALYZE_H

#include <stddef.h>

#define NUM_KEYWORDS 4

void analyzeChunkCPU(const char* data, size_t size,
                      unsigned long long* lineCount,
                      unsigned long long* keywordCounts);

#endif

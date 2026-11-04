#!/usr/bin/env python3
"""Generate a synthetic log file for testing the hybrid MPI-CUDA analyzer.

Usage:
    python3 generate_logs.py [num_lines] [output_file]
"""
import random
import sys
import datetime

LEVELS = ["ERROR", "WARN", "INFO", "DEBUG"]
WEIGHTS = [0.05, 0.15, 0.60, 0.20]
MESSAGES = [
    "User login successful",
    "Connection timeout after {}ms",
    "Null pointer exception in module X",
    "Disk usage at {}%",
    "Cache miss for key {}",
    "Request processed in {}ms",
    "Failed to connect to database",
    "Retrying operation attempt {}",
]


def generate(n, filename):
    start = datetime.datetime.now()
    with open(filename, "w") as f:
        for i in range(n):
            ts = start + datetime.timedelta(milliseconds=i * 7)
            level = random.choices(LEVELS, WEIGHTS)[0]
            msg = random.choice(MESSAGES).format(random.randint(1, 999))
            f.write(f"{ts.isoformat()} [{level}] {msg}\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    out = sys.argv[2] if len(sys.argv) > 2 else "sample.log"
    generate(n, out)
    print(f"Generated {n} log lines -> {out}")

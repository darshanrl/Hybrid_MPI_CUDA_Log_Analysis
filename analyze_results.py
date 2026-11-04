#!/usr/bin/env python3
"""
Reads results.csv (produced by benchmark.sh) and computes, for every run:
  speedup    = serial_time / run_time
  efficiency = speedup / nprocs   (only meaningful for multi-process variants)

Prints a formatted table and saves a speedup-vs-processes chart to
speedup_report.png.
"""
import csv
from collections import defaultdict

rows = []
with open("results.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["nprocs"] = int(r["nprocs"])
        r["time_sec"] = float(r["time_sec"])
        rows.append(r)

serial_time = next(r["time_sec"] for r in rows if r["variant"] == "serial")

print(f"{'Variant':<12} {'Procs':>6} {'Time(s)':>10} {'Speedup':>9} {'Efficiency':>11}")
print("-" * 52)

by_variant = defaultdict(list)
for r in rows:
    speedup = serial_time / r["time_sec"]
    efficiency = speedup / r["nprocs"]
    r["speedup"] = speedup
    r["efficiency"] = efficiency
    by_variant[r["variant"]].append(r)
    print(f"{r['variant']:<12} {r['nprocs']:>6} {r['time_sec']:>10.4f} "
          f"{speedup:>9.2f} {efficiency:>10.1%}")

# ---- optional chart (requires matplotlib) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for variant in ["mpi_only", "hybrid"]:
        pts = sorted(by_variant.get(variant, []), key=lambda r: r["nprocs"])
        if pts:
            plt.plot([p["nprocs"] for p in pts],
                     [p["speedup"] for p in pts],
                     marker="o", label=variant)

    if by_variant.get("cuda_only"):
        plt.scatter([1], [by_variant["cuda_only"][0]["speedup"]],
                    color="red", marker="s", s=80, label="cuda_only (1 GPU)")

    plt.axhline(1, color="gray", linestyle="--", linewidth=1, label="serial baseline")
    plt.xlabel("Number of MPI processes")
    plt.ylabel("Speedup vs. serial baseline")
    plt.title("Speedup: MPI-only vs Hybrid MPI+CUDA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_report.png", dpi=150)
    print("\nChart saved to speedup_report.png")
except ImportError:
    print("\n(matplotlib not installed — skipping chart; table above is still valid)")

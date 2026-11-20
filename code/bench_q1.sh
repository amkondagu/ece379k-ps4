#!/usr/bin/env bash
# Run on a Vista compute node, from code/:
#   bash bench_q1.sh 2>&1 | tee -a results/q1.txt
set -euo pipefail
cd "$(dirname "$0")"

date -Is
hostname
git rev-parse HEAD
module list
g++ --version
nvcc --version
nvidia-smi
make -B q1 nbody

# Check the original port again and exercise partial shared-memory tiles.
for n in 257 16384; do
    ./nbody_serial "$n" 10 ref.bin
    ./nbody "$n" 10 gpu.bin direct
    ./compare ref.bin gpu.bin
    for mode in tiled optimized; do
        for threads in 128 256 512; do
            ./nbody "$n" 10 gpu.bin "$mode" "$threads"
            ./compare ref.bin gpu.bin
        done
    done
done

# Compare tiling with and without approximate reciprocal square root,
# and measure block sizes at both required problem sizes.
for n in 16384 131072; do
    for mode in tiled optimized; do
        for threads in 128 256 512; do
            echo "Warmup: $mode N=$n threads=$threads"
            ./nbody "$n" 10 - "$mode" "$threads"
            for run in 1 2 3; do
                echo "Measured run $run: $mode N=$n threads=$threads"
                ./nbody "$n" 10 - "$mode" "$threads"
            done
        done
    done
done

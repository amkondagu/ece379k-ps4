#!/usr/bin/env bash
# Original Q2 only. Run on a Vista compute node, from code/:
#   bash bench_q2.sh 2>&1 | tee -a results/q2.txt
set -euo pipefail
cd "$(dirname "$0")"
for tool in nsys ncu python3; do
    if ! command -v "$tool"; then
        echo "Missing $tool. Locate/load the required module before profiling."
        exit 1
    fi
done
mkdir -p results

date -Is
hostname
git rev-parse HEAD
git hash-object Q2_frame_pipeline_baseline.cu
module list
nvcc --version
nvidia-smi
nsys --version
ncu --version
make -B frame_pipeline_baseline

cat <<'PREDICTIONS'
Predictions recorded before implementing any Q2 fixes:
calibrate: consecutive warp lanes access different rows, 32 global
  sectors/request instead of 4. Mapping lanes to columns should reduce
  load/store sectors/request from 32 to 4; duration should decrease.
vignette_row: 4096 launches/frame, 32768 over 8 frames. A whole-frame
  kernel should reduce this to 8 launches, reducing host launch time
  and gaps. Total timed launches should fall from 32792 to 32.
rotate: a shared tile column has 32 distinct words in one bank, causing
  32 passes. Padding the row stride to 33 should reduce this to 1 pass,
  eliminate these bank conflicts, and reduce short-scoreboard/MIO stalls.
histogram: one global atomic update per pixel, with neighboring pixels
  usually sharing a sky bin. Block-local accumulation with a bounded
  grid should reduce global reductions and L2 reduction traffic; shared
  atomics increase and retain some contention. Duration should decrease.
These are mechanism predictions, not measured results.
PREDICTIONS

# Use unprofiled execution for the actual application timings.
for run in 0 1 2 3; do
    echo "Unprofiled baseline run $run (0 is warmup)"
    output=$(./frame_pipeline_baseline)
    printf '%s\n' "$output"
    if [[ "$output" != *" -> PASS"* ]]; then
        echo "Q2 baseline correctness check did not PASS."
        exit 1
    fi
done

nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    -o results/q2_baseline -f true ./frame_pipeline_baseline
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_size_sum \
    results/q2_baseline.nsys-rep
nsys export --type sqlite --force-overwrite=true \
    --output results/q2_baseline.sqlite results/q2_baseline.nsys-rep

# Ignore setup kernels and aggregate inter-kernel gaps in the single-stream
# timed pipeline. This keeps 32,000+ timeline entries out of the text log.
python3 - <<'PY'
import sqlite3
with sqlite3.connect('results/q2_baseline.sqlite') as db:
    rows = db.execute('''
        SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE s.value NOT LIKE '%make_frames%'
          AND s.value NOT LIKE '%make_calibration%'
        ORDER BY k.start
    ''').fetchall()
gaps = [max(0, b[0] - a[1]) for a, b in zip(rows, rows[1:])]
if not gaps:
    raise RuntimeError('No pipeline kernel gaps found in Nsight export')
print('Pipeline launches:', len(rows))
print('Inter-kernel gaps: total %.3f ms, mean %.3f us, maximum %.3f us'
      % (sum(gaps) / 1e6, sum(gaps) / len(gaps) / 1e3, max(gaps) / 1e3))
PY

# Verify available counter names on this installation. Full reports below
# collect the supported sections without hard-coding metric names.
ncu --query-metrics | awk '/gpu__time_duration|l1tex__t_sectors|l1tex__t_requests|bank_conflicts|wavefronts_mem_shared|op_global_red|op_shared_atom|lts__t_sectors_op_red|warps_issue_stalled/ {print}'
for kernel in calibrate vignette_row rotate histogram; do
    ncu --kernel-name-base function -k "$kernel" -c 1 --set full \
        --import-source yes -o "results/q2_baseline_$kernel" -f \
        ./frame_pipeline_baseline
    # Export all collected raw counters as text for terminal-only review.
    ncu --import "results/q2_baseline_$kernel.ncu-rep" --page raw --csv
done

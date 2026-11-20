#!/usr/bin/env bash
# Run on a compute node: bash bench_q2_fixed.sh 2>&1 | tee results/q2_fixed.txt
set -euo pipefail
cd "$(dirname "$0")"
for tool in nsys ncu python3; do
    command -v "$tool" || { echo "Missing $tool"; exit 1; }
done
mkdir -p results

date -Is
hostname
git rev-parse HEAD
module list
nvcc --version
nsys --version
ncu --version
nvidia-smi --query-gpu=name,driver_version --format=csv
make -B q2

cat <<'PREDICTIONS'
Predictions (individual fixes are compared against the unchanged baseline):
calibrate: global load/store sectors/request should approach 4 from about
  32; fewer sectors and shorter kernel time, possibly higher HBM utilization.
vignette: launches fall from 32768 to 8, and total timed launches from
  32792 to 32. Host launch time, total GPU stage time, and gaps should fall.
rotate: padding changes the shared column from 32 passes to 1; bank
  conflicts should approach zero and shared-load wavefronts drop about 32x.
histogram: at most 1024*256=262144 global atomic updates/frame, versus
  16777216 originally (at least 64x fewer). Global RED instructions and
  L2 reduction sectors should fall; shared ATOMS increase. Warp matching
  combines same-bin lanes and warp-private counters avoid cross-warp
  contention. Some local atomic and final-merge contention remains.
All: 32 timed launches; stage costs may shift after dominant costs vanish.
Use unprofiled medians for speedups, not Nsight replay-inflated timings.
PREDICTIONS

reference=''
for mode in baseline calibrate vignette rotate histogram all; do
    echo "=== Unprofiled: $mode ==="
    for run in 0 1 2 3; do
        if [[ "$mode" == baseline ]]; then
            output=$(./frame_pipeline_baseline)
        else
            output=$(./frame_pipeline "$mode")
        fi
        summary=$(printf '%s\n' "$output" | grep -E '^ *[0-7] +[0-9]+ +[0-9]+ +[0-9]+$')
        if [[ -z "$reference" ]]; then
            reference="$summary"
            printf '%s\n' "$summary"
        fi
        if [[ "$output" != *" -> PASS"* || "$summary" != "$reference" ]]; then
            printf '%s\n' "$output"
            echo "FAIL: $mode correctness or output summary differs from baseline"
            exit 1
        fi
        echo "run=$run (0=warmup); output matches baseline"
        printf '%s\n' "$output" | grep -E '^check|^pipeline:'
    done

    # Capture each independent fix's timeline, as required for comparison.
    report="results/q2_fixed_$mode"
    if [[ "$mode" == baseline ]]; then
        command=(./frame_pipeline_baseline)
    else
        command=(./frame_pipeline "$mode")
    fi
    echo "=== Systems: $mode ==="
    nsys profile --trace=cuda --sample=none --cpuctxsw=none \
        -o "$report" -f true "${command[@]}" > /tmp/ps4-q2-nsys.log 2>&1 || {
        cat /tmp/ps4-q2-nsys.log; exit 1;
    }
    nsys export --type sqlite --force-overwrite=true --output "$report.sqlite" \
        "$report.nsys-rep" > /tmp/ps4-q2-export.log 2>&1 || {
        cat /tmp/ps4-q2-export.log; exit 1;
    }
    python3 q2_report.py nsys "$report.sqlite"
done

# Check query support locally; keep the metric catalog out of the text log.
ncu --query-metrics > /tmp/ps4-q2-metrics.txt
for mode in baseline calibrate vignette rotate histogram all; do
    if [[ "$mode" == baseline ]]; then
        kernels=(calibrate vignette_row rotate histogram)
        command=(./frame_pipeline_baseline)
    elif [[ "$mode" == all ]]; then
        kernels=(calibrate_coalesced vignette_frame rotate_padded histogram_local)
        command=(./frame_pipeline all)
    else
        case "$mode" in
            calibrate) kernels=(calibrate_coalesced);;
            vignette) kernels=(vignette_frame);;
            rotate) kernels=(rotate_padded);;
            histogram) kernels=(histogram_local);;
        esac
        command=(./frame_pipeline "$mode")
    fi
    for kernel in "${kernels[@]}"; do
        echo "=== Compute: $mode / $kernel ==="
        report="results/q2_fixed_${mode}_$kernel"
        # Reuse your already collected, unchanged baseline counters.
        if [[ "$mode" == baseline && -f "results/q2_baseline_$kernel.ncu-rep" ]]; then
            report="results/q2_baseline_$kernel"
            echo "Reusing original baseline report"
        else
            ncu --kernel-name-base function -k "$kernel" -c 1 --set full \
                --import-source yes -o "$report" -f "${command[@]}" \
                > /tmp/ps4-q2-ncu.log 2>&1 || {
                cat /tmp/ps4-q2-ncu.log; exit 1;
            }
        fi
        python3 q2_report.py ncu "$report.ncu-rep"
    done
done
echo "Q2 correctness, individual fixes, combined timings, and profiles complete."

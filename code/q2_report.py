#!/usr/bin/env python3
# Compact terminal summaries of the native Nsight reports kept on Vista.
import csv
import io
import sqlite3
import subprocess
import sys


def ncu_summary(path):
    raw = subprocess.check_output(
        ['ncu', '--import', path, '--page', 'raw', '--csv'], text=True)
    rows = iter(csv.reader(io.StringIO(raw)))
    for header in rows:
        if header[:2] != ['ID', 'Process ID']:
            continue
        units = next(rows)
        values = next(rows)
        data = dict(zip(header, values))
        unit = dict(zip(header, units))
        print('Kernel:', data['Kernel Name'])
        metrics = [
            'gpu__time_duration.sum', 'launch__grid_size',
            'launch__waves_per_multiprocessor', 'launch__registers_per_thread',
            'sm__warps_active.avg.pct_of_peak_sustained_active',
            'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
            'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
            'dram__bytes_read.sum', 'dram__bytes_write.sum',
            'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
            'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
            'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
            'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
            'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
            'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
            'smsp__inst_executed_op_global_red.sum',
            'lts__t_sectors_srcunit_tex_op_red.sum',
            'smsp__inst_executed_op_shared_atom.sum',
        ]
        for reason in ['long_scoreboard', 'short_scoreboard', 'mio_throttle',
                       'lg_throttle', 'barrier', 'drain', 'math_pipe_throttle']:
            metrics.append('smsp__average_warps_issue_stalled_' + reason
                           + '_per_issue_active.ratio')
        for metric in metrics:
            print('  %s = %s %s' %
                  (metric, data.get(metric, 'NOT COLLECTED'), unit.get(metric, '')))
        for op in ['ld', 'st']:
            prefix = 'l1tex__t_'
            suffix = '_pipe_lsu_mem_global_op_' + op + '.sum'
            requests = float(data.get(prefix + 'requests' + suffix, '0').replace(',', ''))
            sectors = float(data.get(prefix + 'sectors' + suffix, '0').replace(',', ''))
            if requests:
                print('  Global %s sectors/request = %.3f' % (op, sectors / requests))
        return
    raise RuntimeError('No kernel counter row found in ' + path)


def nsys_summary(path):
    with sqlite3.connect(path) as db:
        kernels = db.execute('''
            SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start
        ''').fetchall()
        groups = {}
        pipeline = []
        for start, end, name in kernels:
            name = name.split('(')[0]
            if name in ['make_frames', 'make_calibration']:
                continue
            groups.setdefault(name, []).append(end - start)
            pipeline.append((start, end))
        for name, times in groups.items():
            print('%s: launches=%d, total=%.3f ms, mean=%.3f us' %
                  (name, len(times), sum(times) / 1e6, sum(times) / len(times) / 1e3))
        gaps = [max(0, b[0] - a[1]) for a, b in zip(pipeline, pipeline[1:])]
        if not gaps:
            raise RuntimeError('No pipeline gaps found')
        print('Pipeline launches=%d; gaps total=%.3f ms, mean=%.3f us, max=%.3f us' %
              (len(pipeline), sum(gaps) / 1e6, sum(gaps) / len(gaps) / 1e3, max(gaps) / 1e3))
        launch = db.execute('''
            SELECT COUNT(*), SUM(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaLaunchKernel%'
        ''').fetchone()
        print('cudaLaunchKernel, including setup: calls=%d, host total=%.3f ms' %
              (launch[0], launch[1] / 1e6))
        has_copies = db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMCPY'"
        ).fetchone()
        if has_copies:
            copies = db.execute('SELECT copyKind, SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').fetchall()
            print('Whole-program copies (CUPTI kind, bytes):', copies)
        else:
            print('Copy events were not captured by this profiler installation.')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ['ncu', 'nsys']:
        sys.exit('usage: python3 q2_report.py ncu|nsys report_path')
    if sys.argv[1] == 'ncu':
        ncu_summary(sys.argv[2])
    else:
        nsys_summary(sys.argv[2])

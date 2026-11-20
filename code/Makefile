# ECE 379K -- Problem Set 4 starter code.
#
# Q1 (any Linux machine):   make q1 ; ./nbody_serial 16384 10 ref.bin
# Q1 GPU (your nbody.cu):   make nbody ; ./nbody 16384 10 gpu.bin ; ./compare ref.bin gpu.bin
# Q2 (GPU node):            make q2 ; ./frame_pipeline
#
# On Vista: compile on a login node (module load gcc cuda ; make q2),
# then run on a GPU node (idev -p gh-dev -N 1 -n 1 -t 00:30:00).
# ARCH defaults to sm_90 (H100 / GH200).  Override with make ARCH=sm_xx.
CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra
NVCC     ?= nvcc
ARCH     ?= sm_90
NVFLAGS  ?= -O3 -arch=$(ARCH) -lineinfo -Xcompiler -Wall,-Wextra

all: q1

q1: nbody_serial compare
q2: frame_pipeline frame_pipeline_baseline

nbody_serial: Q1_nbody_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

compare: Q1_compare.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Q1 GPU versions: ./nbody N steps out.bin direct|tiled|optimized [threads]
nbody: nbody.cu check.h
	$(NVCC) $(NVFLAGS) -o $@ $<

frame_pipeline: Q2_frame_pipeline.cu check.h
	$(NVCC) $(NVFLAGS) -o $@ $<

frame_pipeline_baseline: Q2_frame_pipeline_baseline.cu check.h
	$(NVCC) $(NVFLAGS) -o $@ $<

# Whole-program timeline and per-kernel summary (Nsight Systems).
nsys: frame_pipeline
	nsys profile -o pipeline -f true --stats=true ./frame_pipeline

# Full Nsight Compute report for one launch of one kernel:
#   make ncu K=calibrate        (calibrate vignette_row rotate histogram)
K ?= calibrate
ncu: frame_pipeline
	ncu -k $(K) -c 1 --set full --import-source yes -o $(K) -f ./frame_pipeline

# Selected metrics for one kernel: make metrics K=rotate M=a,b,c
metrics: frame_pipeline
	ncu -k $(K) -c 1 --metrics $(M) ./frame_pipeline

clean:
	rm -f nbody_serial compare nbody frame_pipeline frame_pipeline_baseline *.ncu-rep *.nsys-rep *.sqlite

.PHONY: all q1 q2 nsys ncu metrics clean

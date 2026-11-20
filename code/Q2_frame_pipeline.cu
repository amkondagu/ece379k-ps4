// Q2_frame_pipeline.cu -- ECE 379K PS4, Question 2.
//
// A sky-survey camera delivers FRAMES raw frames of H x W pixels.  The
// GPU processes each frame in four stages:
//
//   1. calibrate   v = (raw - dark) * gain            per pixel
//   2. vignette    v *= vig[r]                        per row r
//   3. rotate      the camera is mounted on its side, so the frame is
//                  transposed into display orientation
//   4. histogram   256-bin histogram of the displayed frame
//
// The CPU only orchestrates: it generates nothing but the vignetting
// table, launches the kernels, and checks the last frame against a CPU
// reference at the end.  Raw frames, dark and gain maps are made on the
// GPU before the timed region.
//
// The program is correct: it prints the same histograms on every run
// and the last frame matches the CPU reference exactly.
//
// Build: make q2           (nvcc -O3 -arch=sm_90 -lineinfo)
// Run:   ./frame_pipeline
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "check.h"

constexpr int    W      = 4096;          // pixels per row
constexpr int    H      = 4096;          // rows
constexpr int    FRAMES = 8;
constexpr int    BINS   = 256;
constexpr int    TILE   = 32;
constexpr size_t NPIX   = (size_t)W * H; // 16.7 M pixels, 64 MB per frame
constexpr float  ADC    = 1.0f / 4096.0f;// one ADC count, in pixel units

// ---------------------------------------------------------------------
// Synthetic sensor.  Integer ADC counts scaled by an exact power of two,
// so the host and the device compute bit-identical inputs.
// ---------------------------------------------------------------------
__host__ __device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ inline int dark_counts(int r, int c) {       // 80..95
    return 80 + (int)(hash32((uint32_t)(r * W + c) ^ 0x0000D00Du) & 15u);
}

__host__ __device__ inline float dark_value(int r, int c) {
    return (float)dark_counts(r, c) * ADC;
}

__host__ __device__ inline float gain_value(int r, int c) {      // 1 +- 0.1%
    return (float)(4092u + hash32((uint32_t)(r * W + c) ^ 0x00006A1Eu) % 9u) * ADC;
}

// Dark sky (about 248 counts) with read noise, a star on about one pixel
// in a thousand, and the pixel's dark current.
__host__ __device__ inline float raw_value(int f, int r, int c) {
    uint32_t h = hash32((uint32_t)(r * W + c) * 2654435761u ^ (uint32_t)f * 0x9E3779B9u);
    int counts = 248 + (int)(h % 13u) - 6;
    if (((h >> 8) & 1023u) == 0)
        counts += 1000 + (int)((h >> 18) % 2800u);
    counts += dark_counts(r, c);
    return (float)counts * ADC;
}

__host__ __device__ inline int bin_of(float v) {
    int b = (int)(v * (float)BINS);
    return b < 0 ? 0 : (b >= BINS ? BINS - 1 : b);
}

// ---------------------------------------------------------------------
// Input generation (not timed)
// ---------------------------------------------------------------------
__global__ void make_frames(float *raw) {
    size_t total = (size_t)FRAMES * NPIX;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        int f = (int)(i / NPIX);
        int r = (int)((i % NPIX) / W);
        int c = (int)(i % W);
        raw[i] = raw_value(f, r, c);
    }
}

__global__ void make_calibration(float *dark, float *gain) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < NPIX;
         i += (size_t)gridDim.x * blockDim.x) {
        int r = (int)(i / W);
        int c = (int)(i % W);
        dark[i] = dark_value(r, c);
        gain[i] = gain_value(r, c);
    }
}

// ---------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------

// Stage 1: one thread per pixel.
__global__ void calibrate(const float *__restrict__ raw,
                          const float *__restrict__ dark,
                          const float *__restrict__ gain,
                          float *__restrict__ out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;      // row
    int c = blockIdx.y * blockDim.y + threadIdx.y;      // column
    if (r < H && c < W) {
        size_t i = (size_t)r * W + c;
        out[i] = (raw[i] - dark[i]) * gain[i];
    }
}

// Stage 2: scale one row by its vignetting factor.
__global__ void vignette_row(float *row, float v) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < W)
        row[c] *= v;
}

// Stage 3: out = transpose(in), through a shared-memory tile so that
// both the global read and the global write run along rows.
__global__ void rotate(const float *__restrict__ in, float *__restrict__ out) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[(size_t)y * W + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    out[(size_t)y * H + x] = tile[threadIdx.x][threadIdx.y];
}

// Stage 4: one thread per pixel.
__global__ void histogram(const float *__restrict__ img, unsigned *__restrict__ hist) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NPIX)
        atomicAdd(&hist[bin_of(img[i])], 1u);
}

// ---------------------------------------------------------------------
// CPU reference for one frame: the displayed image and its histogram.
// ---------------------------------------------------------------------
void reference(int f, const std::vector<float> &vig, std::vector<float> &disp,
               std::vector<unsigned> &hist) {
    std::fill(hist.begin(), hist.end(), 0u);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            float v = (raw_value(f, r, c) - dark_value(r, c)) * gain_value(r, c);
            v *= vig[r];
            disp[(size_t)c * H + r] = v;
            hist[bin_of(v)]++;
        }
}

int main() {
    // Vignetting: the optics dim the top and bottom rows of the sensor.
    std::vector<float> vig(H);
    for (int r = 0; r < H; r++) {
        float d = (r - H / 2 + 0.5f) / (H / 2);
        vig[r] = 1.0f / (1.0f + 0.3f * d * d);
    }

    float *d_raw, *d_dark, *d_gain, *d_work, *d_disp;
    unsigned *d_hist;
    CUDA_CHECK(cudaMalloc(&d_raw,  FRAMES * NPIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dark, NPIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gain, NPIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_work, NPIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_disp, NPIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hist, FRAMES * BINS * sizeof(unsigned)));

    make_frames<<<1024, 256>>>(d_raw);
    make_calibration<<<1024, 256>>>(d_dark, d_gain);
    CUDA_CHECK(cudaMemset(d_hist, 0, FRAMES * BINS * sizeof(unsigned)));
    KERNEL_CHECK();

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // ---------------------------------------------------- timed region
    CUDA_CHECK(cudaEventRecord(t0));
    for (int f = 0; f < FRAMES; f++) {
        const float *raw  = d_raw + f * NPIX;
        unsigned    *hist = d_hist + f * BINS;

        dim3 cal_block(32, 8);
        dim3 cal_grid(H / 32, W / 8);
        calibrate<<<cal_grid, cal_block>>>(raw, d_dark, d_gain, d_work);

        for (int r = 0; r < H; r++)
            vignette_row<<<W / 256, 256>>>(d_work + (size_t)r * W, vig[r]);

        dim3 rot_block(TILE, TILE);
        dim3 rot_grid(W / TILE, H / TILE);
        rotate<<<rot_grid, rot_block>>>(d_work, d_disp);

        histogram<<<(unsigned)(NPIX / 256), 256>>>(d_disp, hist);

        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    // ------------------------------------------------ end timed region

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<unsigned> hist(FRAMES * BINS);
    std::vector<float> disp(NPIX);
    CUDA_CHECK(cudaMemcpy(hist.data(), d_hist, hist.size() * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(disp.data(), d_disp, NPIX * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::printf("frame  sky bin  sky pixels  star pixels (bin >= 64)\n");
    for (int f = 0; f < FRAMES; f++) {
        const unsigned *h = &hist[f * BINS];
        int mode = 0;
        unsigned long stars = 0;
        for (int b = 0; b < BINS; b++) {
            if (h[b] > h[mode]) mode = b;
            if (b >= 64) stars += h[b];
        }
        std::printf("%5d  %7d  %10u  %11lu\n", f, mode, h[mode], stars);
    }

    // Check the last frame against the CPU.
    std::vector<float> ref_disp(NPIX);
    std::vector<unsigned> ref_hist(BINS);
    reference(FRAMES - 1, vig, ref_disp, ref_hist);
    size_t bad_px = 0, bad_bins = 0;
    for (size_t i = 0; i < NPIX; i++)
        if (disp[i] != ref_disp[i]) bad_px++;
    for (int b = 0; b < BINS; b++)
        if (hist[(FRAMES - 1) * BINS + b] != ref_hist[b]) bad_bins++;
    std::printf("check (last frame): %zu wrong pixels, %zu wrong bins -> %s\n",
                bad_px, bad_bins, (bad_px == 0 && bad_bins == 0) ? "PASS" : "FAIL");

    std::printf("pipeline: %.2f ms for %d frames (%.3f ms per frame)\n",
                ms, FRAMES, ms / FRAMES);

    cudaFree(d_raw); cudaFree(d_dark); cudaFree(d_gain);
    cudaFree(d_work); cudaFree(d_disp); cudaFree(d_hist);
    return 0;
}

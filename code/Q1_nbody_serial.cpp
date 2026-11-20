// Q1_nbody_serial.cpp -- ECE 379K PS4, Question 1.
//
// All-pairs gravitational N-body simulation, one CPU thread.
//
// Two clusters of N/2 bodies each fall toward one another.  Every step
// computes the acceleration of every body from every other body
// (N^2 interactions), then updates velocities and positions
// (semi-implicit Euler).  Units have G = 1.
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -o nbody_serial Q1_nbody_serial.cpp
// Run:   ./nbody_serial [N] [steps] [out.bin]       defaults: 16384 10
//
// If out.bin is given, the final state is written to it so that another
// implementation can be compared with Q1_compare.cpp.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr float DT   = 0.005f;   // time step
constexpr float EPS2 = 1e-4f;    // softening length squared (eps = 0.01)

struct Body {
    float x, y, z;      // position
    float vx, vy, vz;   // velocity
    float m;            // mass
};

// ---------------------------------------------------------------------
// Initial conditions.  A small xorshift generator makes them identical
// on every machine and compiler, so a GPU version can reproduce them by
// copying this function unchanged.
// ---------------------------------------------------------------------
static std::uint32_t rng_state = 2463534242u;

static float uniform01() {                    // [0, 1)
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (rng_state >> 8) * (1.0f / 16777216.0f);
}

void init_bodies(std::vector<Body> &b) {
    const int n = (int)b.size();
    for (int i = 0; i < n; i++) {
        // a point uniformly distributed in the unit ball (rejection)
        float x, y, z;
        do {
            x = 2.0f * uniform01() - 1.0f;
            y = 2.0f * uniform01() - 1.0f;
            z = 2.0f * uniform01() - 1.0f;
        } while (x * x + y * y + z * z > 1.0f);

        // first half: cluster at -1.5 moving right; second half: the mirror
        float side = (i < n / 2) ? -1.0f : 1.0f;
        b[i].x  = x + 1.5f * side;
        b[i].y  = y;
        b[i].z  = z;
        b[i].vx = -0.2f * side + 0.05f * (uniform01() - 0.5f);
        b[i].vy =  0.3f * z * side;           // some rotation about x
        b[i].vz = -0.3f * y * side;
        b[i].m  = 1.0f / n;
    }
}

// ---------------------------------------------------------------------
// One time step: O(N^2) accelerations, then O(N) updates.
// ---------------------------------------------------------------------
void step(std::vector<Body> &b, std::vector<float> &ax,
          std::vector<float> &ay, std::vector<float> &az) {
    const int n = (int)b.size();

    for (int i = 0; i < n; i++) {
        float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
        for (int j = 0; j < n; j++) {         // j == i adds exactly 0
            float dx = b[j].x - b[i].x;
            float dy = b[j].y - b[i].y;
            float dz = b[j].z - b[i].z;
            float r2 = dx * dx + dy * dy + dz * dz + EPS2;
            float inv_r  = 1.0f / std::sqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            float s = b[j].m * inv_r3;
            axi += s * dx;
            ayi += s * dy;
            azi += s * dz;
        }
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }

    for (int i = 0; i < n; i++) {             // kick, then drift
        b[i].vx += ax[i] * DT;
        b[i].vy += ay[i] * DT;
        b[i].vz += az[i] * DT;
        b[i].x  += b[i].vx * DT;
        b[i].y  += b[i].vy * DT;
        b[i].z  += b[i].vz * DT;
    }
}

int main(int argc, char **argv) {
    const int n     = (argc > 1) ? std::atoi(argv[1]) : 16384;
    const int steps = (argc > 2) ? std::atoi(argv[2]) : 10;
    const char *out = (argc > 3) ? argv[3] : nullptr;

    std::vector<Body> b(n);
    std::vector<float> ax(n), ay(n), az(n);
    init_bodies(b);

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++)
        step(b, ax, ay, az);
    auto t1 = std::chrono::steady_clock::now();

    double sec   = std::chrono::duration<double>(t1 - t0).count();
    double inter = (double)n * n * steps;

    // A few numbers to check against: centre of mass, mean distance
    // from it, and the largest speed.
    double cx = 0, cy = 0, cz = 0, mtot = 0;
    for (const Body &p : b) {
        cx += p.m * p.x; cy += p.m * p.y; cz += p.m * p.z; mtot += p.m;
    }
    cx /= mtot; cy /= mtot; cz /= mtot;
    double rmean = 0, vmax = 0;
    for (const Body &p : b) {
        rmean += std::sqrt((p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy) +
                           (p.z - cz) * (p.z - cz));
        double v = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
        if (v > vmax) vmax = v;
    }
    rmean /= n;

    std::printf("N = %d, steps = %d\n", n, steps);
    std::printf("centre of mass  = (%.6f, %.6f, %.6f)\n", cx, cy, cz);
    std::printf("mean radius     = %.6f\n", rmean);
    std::printf("max speed       = %.6f\n", vmax);
    std::printf("body 0          = (%.6f, %.6f, %.6f)\n", b[0].x, b[0].y, b[0].z);
    std::printf("body N-1        = (%.6f, %.6f, %.6f)\n",
                b[n - 1].x, b[n - 1].y, b[n - 1].z);
    std::printf("time            = %.3f s  (%.3f s/step)\n", sec, sec / steps);
    std::printf("throughput      = %.3f G interactions/s\n", inter / sec * 1e-9);

    if (out) {
        FILE *f = std::fopen(out, "wb");
        if (!f) { std::perror(out); return 1; }
        std::fwrite(&n, sizeof n, 1, f);
        std::fwrite(b.data(), sizeof(Body), n, f);
        std::fclose(f);
        std::printf("final state written to %s\n", out);
    }
    return 0;
}

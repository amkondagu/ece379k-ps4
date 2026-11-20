// Q1_compare.cpp -- compare two N-body final states written by
// Q1_nbody_serial (or by your GPU version, in the same format:
// an int N followed by N records of 7 floats x y z vx vy vz m).
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -o compare Q1_compare.cpp
// Run:   ./compare ref.bin gpu.bin
//
// PASS if every body's position differs by less than 1e-3 (absolute;
// the system is about 5 units across).
#include <cmath>
#include <cstdio>
#include <vector>

struct Body { float x, y, z, vx, vy, vz, m; };

static bool load(const char *path, std::vector<Body> &b) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::perror(path); return false; }
    int n = 0;
    if (std::fread(&n, sizeof n, 1, f) != 1 || n <= 0) {
        std::fprintf(stderr, "%s: bad header\n", path);
        std::fclose(f);
        return false;
    }
    b.resize(n);
    size_t got = std::fread(b.data(), sizeof(Body), n, f);
    std::fclose(f);
    if (got != (size_t)n) {
        std::fprintf(stderr, "%s: expected %d bodies, read %zu\n", path, n, got);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s ref.bin test.bin\n", argv[0]);
        return 2;
    }
    std::vector<Body> a, b;
    if (!load(argv[1], a) || !load(argv[2], b)) return 2;
    if (a.size() != b.size()) {
        std::fprintf(stderr, "different N: %zu vs %zu\n", a.size(), b.size());
        return 1;
    }
    double maxd = 0, sum2 = 0;
    size_t worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double dx = a[i].x - b[i].x, dy = a[i].y - b[i].y, dz = a[i].z - b[i].z;
        double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        sum2 += d * d;
        if (d > maxd) { maxd = d; worst = i; }
    }
    std::printf("N = %zu\n", a.size());
    std::printf("max position difference = %.3e (body %zu)\n", maxd, worst);
    std::printf("rms position difference = %.3e\n", std::sqrt(sum2 / a.size()));
    bool ok = maxd < 1e-3;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

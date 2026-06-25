// Microbenchmark: distance-kernel throughput for the per-dim + FFD byte-packed
// layout vs float32 baseline vs a uniform-width layout, using ADC (asymmetric
// distance computation: per-dim query*centroid LUT, accumulate by code).
//
// Measures the inner loop that dominates ANN search: score N database codes
// against one query. Single-threaded, per-core throughput. Not a full search
// (no IVF/pruning) -- it isolates "how fast can you decode+accumulate these codes".
//
// Finding (D=1024, avg 4bpd, one core): float32 dot ~2.6 M/s (bandwidth-bound,
// vectorizes), uniform-ADC ~2.4 M/s, FFD-per-dim-ADC ~2.1 M/s. Quantized ADC is
// GATHER-bound (1024 dependent per-dim LUT lookups/code), not bandwidth-bound, so
// 8x smaller codes buy FOOTPRINT not scoring speed with a naive kernel; none beat
// float32. The engine's real speedup is SIMD fastscan (uniform 4-bit + vpshufb),
// which FFD's VARIABLE widths preclude. => per-dim+FFD is a memory/accuracy play.
//
// build: g++ -O3 -march=native -funroll-loops bench/ffd_speed.cpp -o ffd_speed
// run:   ./ffd_speed [D=1024] [N=131072] [REPS=50] [AVG_BITS=4]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
using namespace std;
using clk = chrono::high_resolution_clock;

static double now_s() { return chrono::duration<double>(clk::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 1024;
    const long N = argc > 2 ? atol(argv[2]) : 131072;
    const int REPS = argc > 3 ? atoi(argv[3]) : 50;
    const double AVG_BITS = argc > 4 ? atof(argv[4]) : 4.0;
    mt19937 rng(0);

    // ---- per-dim bit allocation (~AVG_BITS), variable widths 1..8 ----
    normal_distribution<double> nd(AVG_BITS, 2.0);
    vector<int> bits(D);
    for (int d = 0; d < D; d++) bits[d] = min(8, max(1, (int)lround(nd(rng))));
    long total_bits = accumulate(bits.begin(), bits.end(), 0L);

    // ---- FFD layout (first-fit-decreasing; speed depends on layout, not optimality) ----
    vector<int> byte_idx(D), shift(D), order(D), mask(D);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){ return bits[a] > bits[b]; });
    vector<int> rem;
    for (int d : order) {
        int placed = -1;
        for (int b = 0; b < (int)rem.size(); b++) if (rem[b] >= bits[d]) { placed = b; break; }
        if (placed < 0) { placed = rem.size(); rem.push_back(8); }
        int off = 8 - rem[placed];
        byte_idx[d] = placed;
        shift[d] = 8 - off - bits[d];
        mask[d] = (1 << bits[d]) - 1;
        rem[placed] -= bits[d];
    }
    int n_bytes = rem.size();

    // ---- uniform-width layout at same avg (round to nearest bit) ----
    int bu = max(1, min(8, (int)lround(AVG_BITS)));
    int per_byte = 8 / bu;
    int u_bytes = (D + per_byte - 1) / per_byte;

    // ---- per-dim codebooks ----
    normal_distribution<float> cn(0.f, 1.f);
    vector<vector<float>> cb(D), cbu(D);
    for (int d = 0; d < D; d++) { cb[d].resize(1 << bits[d]); for (auto& v : cb[d]) v = cn(rng); }
    for (int d = 0; d < D; d++) { cbu[d].resize(1 << bu); for (auto& v : cbu[d]) v = cn(rng); }

    // ---- database: random codes, packed three ways ----
    vector<float> dbf((size_t)N * D);
    vector<uint8_t> ffd((size_t)N * n_bytes, 0);
    vector<uint8_t> uni((size_t)N * u_bytes, 0);
    for (long i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            int code = rng() & mask[d];
            dbf[(size_t)i * D + d] = cb[d][code];
            ffd[(size_t)i * n_bytes + byte_idx[d]] |= (uint8_t)(code << shift[d]);
            int ucode = rng() & ((1 << bu) - 1);
            int ub = d / per_byte, us = 8 - (d % per_byte + 1) * bu;
            uni[(size_t)i * u_bytes + ub] |= (uint8_t)(ucode << us);
        }
    }

    // ---- query + ADC LUTs ----
    vector<float> q(D); for (auto& v : q) v = cn(rng);
    vector<int> lut_off(D + 1, 0);
    for (int d = 0; d < D; d++) lut_off[d + 1] = lut_off[d] + (1 << bits[d]);
    vector<float> lut(lut_off[D]);
    for (int d = 0; d < D; d++) for (int c = 0; c < (1 << bits[d]); c++) lut[lut_off[d] + c] = q[d] * cb[d][c];
    vector<float> ulut((size_t)D * (1 << bu));
    for (int d = 0; d < D; d++) for (int c = 0; c < (1 << bu); c++) ulut[(size_t)d * (1 << bu) + c] = q[d] * cbu[d][c];
    // precompute uniform-layout per-dim byte index + shift (no division in hot loop)
    vector<int> u_bi(D), u_sh(D), u_lutoff(D);
    int umask = (1 << bu) - 1;
    for (int d = 0; d < D; d++) { u_bi[d] = d / per_byte; u_sh[d] = 8 - (d % per_byte + 1) * bu; u_lutoff[d] = d * (1 << bu); }

    volatile float sink = 0;
    auto bench = [&](const char* name, long bytes_per_code, auto fn) {
        double t0 = now_s(); float acc = 0;
        for (int r = 0; r < REPS; r++) for (long i = 0; i < N; i++) acc += fn(i);
        double dt = now_s() - t0; sink += acc;
        double codes = (double)N * REPS;
        printf("  %-22s  %7.1f M codes/s   %6.2f GB/s   %5ld B/code\n",
               name, codes / dt / 1e6, codes * bytes_per_code / dt / 1e9, bytes_per_code);
        return codes / dt;
    };

    printf("D=%d  N=%ld  reps=%d  avg_bits=%.1f  | FFD %d B/code (%ld bits), float32 %d B/code\n",
           D, N, REPS, AVG_BITS, n_bytes, total_bits, D * 4);

    bench("float32 dot", (long)D * 4, [&](long i) {
        const float* x = &dbf[(size_t)i * D]; float a = 0;
        for (int d = 0; d < D; d++) a += q[d] * x[d];
        return a;
    });
    bench("FFD per-dim ADC", n_bytes, [&](long i) {
        const uint8_t* c = &ffd[(size_t)i * n_bytes]; float a = 0;
        for (int d = 0; d < D; d++) a += lut[lut_off[d] + ((c[byte_idx[d]] >> shift[d]) & mask[d])];
        return a;
    });
    bench("uniform-width ADC", u_bytes, [&](long i) {
        const uint8_t* c = &uni[(size_t)i * u_bytes]; float a = 0;
        for (int d = 0; d < D; d++) a += ulut[u_lutoff[d] + ((c[u_bi[d]] >> u_sh[d]) & umask)];
        return a;
    });
    printf("  (quantized ADC is gather-bound, not bandwidth-bound; fastscan needs uniform widths)\n");
    return (int)sink & 0;
}

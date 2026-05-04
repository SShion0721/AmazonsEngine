/*==================================================================
 * AMAZONS ENGINE — nnue.cpp
 * Forward pass integer arithmetic with AVX2 SIMD acceleration.
 * Falls back to scalar code when AVX2 is unavailable.
 *==================================================================*/
#include "nnue.h"
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>

// ── SIMD detection ──────────────────────────────────────────────────
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#define NNUE_USE_AVX2 1
#elif defined(_MSC_VER)
// MSVC with /arch:AVX2 defines __AVX2__ since VS2017+
// but let's also check the compile flag via intrin.h availability
#include <intrin.h>
#if defined(__AVX2__)
#define NNUE_USE_AVX2 1
#endif
#endif

#ifdef NNUE_USE_AVX2
#include <immintrin.h>
// 12th Gen Intel (Alder Lake) supports AVX-VNNI but not AVX-512
// Enable this manually since the user confirmed a 12500H CPU
#define NNUE_USE_VNNI 1
#endif

namespace NNUE {

NetworkParameters g_weights;

// ── Dummy Weight Initialization ─────────────────────────────────────
// Fills weights with small random INT8/INT16 values to test inference
void init_random_weights() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int16_t> dist16(-100, 100);
    std::uniform_int_distribution<int8_t>  dist8(-20, 20);
    std::uniform_int_distribution<int32_t> dist32(-500, 500);

    // Layer 0
    for (int i = 0; i < ACC_SIZE; ++i) g_weights.l0_biases[i] = dist16(rng);
    for (int i = 0; i < FEATURE_SIZE; ++i)
        for (int j = 0; j < ACC_SIZE; ++j)
            g_weights.l0_weights[i][j] = dist16(rng);

    // Layer 1
    for (int i = 0; i < HIDDEN_SIZE; ++i) g_weights.l1_biases[i] = dist32(rng);
    for (int i = 0; i < ACC_SIZE * 2; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            g_weights.l1_weights[i][j] = dist8(rng);

    // Layer 2
    for (int i = 0; i < HIDDEN_SIZE; ++i) g_weights.l2_biases[i] = dist32(rng);
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            g_weights.l2_weights[i][j] = dist8(rng);

    // Layer 3
    g_weights.l3_biases[0] = 0;
    for (int i = 0; i < HIDDEN_SIZE; ++i) g_weights.l3_weights[i][0] = dist8(rng);
}

// ── Weight Loading ──────────────────────────────────────────────────
bool load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        init_random_weights();
        return false;
    }

    const std::size_t compact_size =
        sizeof(g_weights.l0_biases) +
        sizeof(g_weights.l0_weights) +
        sizeof(g_weights.l1_biases) +
        sizeof(g_weights.l1_weights) +
        sizeof(g_weights.l2_biases) +
        sizeof(g_weights.l2_weights) +
        sizeof(g_weights.l3_biases) +
        sizeof(g_weights.l3_weights);

    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size == static_cast<std::streamoff>(compact_size)) {
        // Preferred path: compact serialized layout from train/train.py exporter.
        file.read(reinterpret_cast<char*>(g_weights.l0_biases), sizeof(g_weights.l0_biases));
        file.read(reinterpret_cast<char*>(g_weights.l0_weights), sizeof(g_weights.l0_weights));
        file.read(reinterpret_cast<char*>(g_weights.l1_biases), sizeof(g_weights.l1_biases));
        file.read(reinterpret_cast<char*>(g_weights.l1_weights), sizeof(g_weights.l1_weights));
        file.read(reinterpret_cast<char*>(g_weights.l2_biases), sizeof(g_weights.l2_biases));
        file.read(reinterpret_cast<char*>(g_weights.l2_weights), sizeof(g_weights.l2_weights));
        file.read(reinterpret_cast<char*>(g_weights.l3_biases), sizeof(g_weights.l3_biases));
        file.read(reinterpret_cast<char*>(g_weights.l3_weights), sizeof(g_weights.l3_weights));
        return bool(file);
    }

    if (file_size == static_cast<std::streamoff>(sizeof(NetworkParameters))) {
        // Backward-compatible path for old raw-struct dumps.
        file.read(reinterpret_cast<char*>(&g_weights), sizeof(NetworkParameters));
        return bool(file);
    }

    init_random_weights();
    return false;
}

// ══════════════════════════════════════════════════════════════════════
// ── AVX2 SIMD Forward Pass ──────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════
#ifdef NNUE_USE_AVX2

// AVX2 Clipped ReLU for int16 accumulator → int8 [0..127]
// Processes 256 int16 values → 256 int8 values
static void avx2_clipped_relu_16to8(const int16_t* input, int8_t* output, int count) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max127 = _mm256_set1_epi8(127);
    // Process 32 int16 → 32 int8 per iteration (two 256-bit loads → one pack)
    for (int i = 0; i < count; i += 32) {
        __m256i lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i));
        __m256i hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i + 16));
        // Saturating pack: int16 → int8 with signed saturation
        __m256i packed = _mm256_packs_epi16(lo, hi);
        // Fix AVX2 lane interleave: packs works within 128-bit lanes
        packed = _mm256_permute4x64_epi64(packed, 0xD8); // 0b11_01_10_00
        // Clamp to [0, 127]
        packed = _mm256_max_epi8(packed, zero);
        packed = _mm256_min_epi8(packed, max127);
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + i), packed);
    }
}

// AVX2 matrix-vector multiply: int8 input × int8 weights → int32 output
// For small layers (32 neurons), processes one output at a time with dot products
static void avx2_linear_8x8_to_32(const int8_t* input, const int8_t weights[][HIDDEN_SIZE],
                                    const int32_t* biases, int32_t* output,
                                    int input_size, int output_size) {
    for (int o = 0; o < output_size; ++o) {
        __m256i sum = _mm256_setzero_si256();
        int j = 0;
        // Process 32 elements per iteration
        for (; j + 31 < input_size; j += 32) {
            __m256i in_vec = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(input + j));
            // Gather weights for output neuron o from each input j
            // Weights are stored as weights[j][o], so stride = HIDDEN_SIZE
            alignas(32) int8_t w_col[32];
            for (int k = 0; k < 32; ++k)
                w_col[k] = weights[j + k][o];
            __m256i w_vec = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(w_col));
#ifdef NNUE_USE_VNNI
            // ── AVX-VNNI (Alder Lake) ──────────────────────────────────────────
            // Single instruction: 32x INT8 dot product + 32-bit accumulate!
            // First operand must be uint8 (which it is, [0,127]), second int8
            sum = _mm256_dpbusd_epi32(sum, in_vec, w_vec);
#else
            // ── Standard AVX2 ──────────────────────────────────────────────────
            __m256i prod = _mm256_maddubs_epi16(
                _mm256_max_epu8(in_vec, _mm256_setzero_si256()), // ensure unsigned for maddubs
                w_vec);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
#endif
        }
        // Horizontal sum of 8 int32s
        __m128i lo128 = _mm256_castsi256_si128(sum);
        __m128i hi128 = _mm256_extracti128_si256(sum, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0xB1));
        int32_t result = _mm_cvtsi128_si32(sum128);
        // Handle remaining elements
        for (; j < input_size; ++j)
            result += static_cast<int32_t>(input[j]) * static_cast<int32_t>(weights[j][o]);
        output[o] = biases[o] + result;
    }
}

Score evaluate_nnue(Color side_to_move, const Accumulator& acc) {
    alignas(64) int8_t concat_layer[ACC_SIZE * 2];

    // 1. Clipped ReLU on accumulator → int8
    const int16_t* us   = acc.state[side_to_move];
    const int16_t* them = acc.state[side_to_move ^ 1];

    avx2_clipped_relu_16to8(us,   concat_layer, ACC_SIZE);
    avx2_clipped_relu_16to8(them, concat_layer + ACC_SIZE, ACC_SIZE);

    // 2. Hidden Layer 1 (512 → 32) — use scalar for now (small layer)
    int32_t hidden1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int32_t sum = g_weights.l1_biases[i];
        for (int j = 0; j < ACC_SIZE * 2; ++j)
            sum += static_cast<int32_t>(concat_layer[j]) *
                   static_cast<int32_t>(g_weights.l1_weights[j][i]);
        hidden1[i] = sum;
    }

    // Activate Hidden Layer 1
    int8_t hidden1_act[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        hidden1_act[i] = static_cast<int8_t>(std::clamp<int>(hidden1[i] >> 6, 0, 127));

    // 3. Hidden Layer 2 (32 → 32) — scalar (tiny)
    int32_t hidden2[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int32_t sum = g_weights.l2_biases[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            sum += static_cast<int32_t>(hidden1_act[j]) *
                   static_cast<int32_t>(g_weights.l2_weights[j][i]);
        hidden2[i] = sum;
    }

    // Activate Hidden Layer 2
    int8_t hidden2_act[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        hidden2_act[i] = static_cast<int8_t>(std::clamp<int>(hidden2[i] >> 6, 0, 127));

    // 4. Output Layer (32 → 1)
    int32_t output = g_weights.l3_biases[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        output += static_cast<int32_t>(hidden2_act[i]) *
                  static_cast<int32_t>(g_weights.l3_weights[i][0]);

    return output / 16;
}

// ── AVX2 accumulator add/sub (called from position.cpp via these helpers) ─
void acc_add_weights(int16_t* acc_row, const int16_t* weights, int count) {
    for (int i = 0; i < count; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_row + i));
        __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_row + i),
                           _mm256_add_epi16(a, w));
    }
}

void acc_sub_weights(int16_t* acc_row, const int16_t* weights, int count) {
    for (int i = 0; i < count; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_row + i));
        __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_row + i),
                           _mm256_sub_epi16(a, w));
    }
}

#else // ── Scalar fallback ───────────────────────────────────────────

static inline int8_t clipped_relu(int16_t x) {
    return std::clamp<int>(x, 0, 127);
}

static inline int8_t clipped_relu32(int32_t x) {
    return std::clamp<int>(x >> 6, 0, 127);
}

Score evaluate_nnue(Color side_to_move, const Accumulator& acc) {
    int8_t concat_layer[ACC_SIZE * 2];

    const int16_t* us   = acc.state[side_to_move];
    const int16_t* them = acc.state[side_to_move ^ 1];

    for (int i = 0; i < ACC_SIZE; ++i) {
        concat_layer[i]            = clipped_relu(us[i]);
        concat_layer[ACC_SIZE + i] = clipped_relu(them[i]);
    }

    // 2. Hidden Layer 1 (512 -> 32)
    int32_t hidden1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden1[i] = g_weights.l1_biases[i];
        for (int j = 0; j < ACC_SIZE * 2; ++j) {
            hidden1[i] += concat_layer[j] * g_weights.l1_weights[j][i];
        }
    }

    int8_t hidden1_act[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        hidden1_act[i] = clipped_relu32(hidden1[i]);

    // 3. Hidden Layer 2 (32 -> 32)
    int32_t hidden2[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden2[i] = g_weights.l2_biases[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            hidden2[i] += hidden1_act[j] * g_weights.l2_weights[j][i];
    }

    int8_t hidden2_act[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        hidden2_act[i] = clipped_relu32(hidden2[i]);

    // 4. Output Layer (32 -> 1)
    int32_t output = g_weights.l3_biases[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        output += hidden2_act[i] * g_weights.l3_weights[i][0];

    return output / 16;
}

// Scalar accumulator helpers
void acc_add_weights(int16_t* acc_row, const int16_t* weights, int count) {
    for (int i = 0; i < count; ++i)
        acc_row[i] += weights[i];
}

void acc_sub_weights(int16_t* acc_row, const int16_t* weights, int count) {
    for (int i = 0; i < count; ++i)
        acc_row[i] -= weights[i];
}

#endif // NNUE_USE_AVX2

} // namespace NNUE

/*==================================================================
 * AMAZONS ENGINE - nnue.cpp
 * AmazonsNNUE-v2 integer inference.
 *==================================================================*/
#include "nnue.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>

#if defined(__AVX2__)
#define NNUE_USE_AVX2 1
#include <immintrin.h>
#endif

#if defined(__AVXVNNI__) || defined(__AVX_VNNI__)
#define NNUE_USE_VNNI 1
#endif

namespace NNUE {

NetworkParameters g_weights;
bool g_accumulator_enabled = false;

namespace {

constexpr char WEIGHT_MAGIC[8] = {'A', 'M', 'Z', 'N', 'U', 'E', '2', '\0'};
const char* g_weight_format = "AMZNUE2";

#pragma pack(push, 1)
struct WeightHeader {
    char magic[8];
    uint32_t version;
    uint32_t base_feature_size;
    uint32_t base_acc_size;
    uint32_t line_feature_hash;
    uint32_t line_acc_size;
    uint32_t hidden_size;
    uint32_t global_feature_size;
    uint32_t phase_buckets;
    uint32_t sparse_input_size;
    uint32_t ft_quant_one;
    uint32_t hidden_weight_scale;
    uint32_t output_weight_scale;
    uint32_t line_seed_low;
    uint32_t line_seed_high;
};
#pragma pack(pop)

bool valid_header(const WeightHeader& h) {
    return std::memcmp(h.magic, WEIGHT_MAGIC, sizeof(WEIGHT_MAGIC)) == 0
        && h.version == WEIGHT_VERSION
        && h.base_feature_size == BASE_FEATURE_SIZE
        && h.base_acc_size == BASE_ACC_SIZE
        && h.line_feature_hash == LINE_FEATURE_HASH
        && h.line_acc_size == LINE_ACC_SIZE
        && h.hidden_size == HIDDEN_SIZE
        && h.global_feature_size == GLOBAL_FEATURE_SIZE
        && h.phase_buckets == PHASE_BUCKETS
        && h.sparse_input_size == SPARSE_INPUT_SIZE
        && h.ft_quant_one == FT_QUANT_ONE
        && h.hidden_weight_scale == HIDDEN_WEIGHT_SCALE
        && h.output_weight_scale == OUTPUT_WEIGHT_SCALE
        && h.line_seed_low == static_cast<uint32_t>(LINE_SEED_BASE)
        && h.line_seed_high == static_cast<uint32_t>(LINE_SEED_BASE >> 32);
}

std::streamoff expected_file_size() {
    const std::size_t compact_size =
        sizeof(g_weights.base_l0_biases) +
        sizeof(g_weights.base_l0_weights) +
        sizeof(g_weights.line_l0_biases) +
        sizeof(g_weights.line_l0_weights) +
        sizeof(g_weights.l1_biases) +
        sizeof(g_weights.l1_weights_t) +
        sizeof(g_weights.global_l1_weights) +
        sizeof(g_weights.phase_l1_biases) +
        sizeof(g_weights.l2_biases) +
        sizeof(g_weights.l2_weights_t) +
        sizeof(g_weights.l3_biases) +
        sizeof(g_weights.l3_weights);
    return static_cast<std::streamoff>(sizeof(WeightHeader) + compact_size);
}

inline int8_t clipped_relu32(int32_t x) {
    return static_cast<int8_t>(std::clamp<int>(x >> 6, 0, 127));
}

#ifdef NNUE_USE_AVX2

void clipped_relu_16to8(const int16_t* input, int8_t* output, int count) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max127 = _mm256_set1_epi8(127);
    for (int i = 0; i < count; i += 32) {
        __m256i lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i));
        __m256i hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i + 16));
        __m256i packed = _mm256_packs_epi16(lo, hi);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        packed = _mm256_max_epi8(packed, zero);
        packed = _mm256_min_epi8(packed, max127);
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + i), packed);
    }
}

inline int32_t horizontal_sum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
    return _mm_cvtsi128_si32(sum);
}

int32_t dot_u8_i8(const int8_t* input, const int8_t* weights, int count) {
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < count; i += 32) {
        const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i));
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
#ifdef NNUE_USE_VNNI
        sum = _mm256_dpbusd_epi32(sum, in, w);
#else
        const __m256i prod16 = _mm256_maddubs_epi16(in, w);
        const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
        sum = _mm256_add_epi32(sum, prod32);
#endif
    }
    return horizontal_sum_epi32(sum);
}

#else

void clipped_relu_16to8(const int16_t* input, int8_t* output, int count) {
    for (int i = 0; i < count; ++i)
        output[i] = static_cast<int8_t>(std::clamp<int>(input[i], 0, 127));
}

int32_t dot_u8_i8(const int8_t* input, const int8_t* weights, int count) {
    int32_t sum = 0;
    for (int i = 0; i < count; ++i)
        sum += static_cast<int32_t>(input[i]) * static_cast<int32_t>(weights[i]);
    return sum;
}

#endif

} // namespace

const char* weight_format() {
    return g_weight_format;
}

void init_random_weights() {
    std::memset(&g_weights, 0, sizeof(g_weights));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int16_t> dist16(-32, 32);
    std::uniform_int_distribution<int8_t> dist8(-8, 8);
    std::uniform_int_distribution<int32_t> dist32(-128, 128);

    for (int i = 0; i < BASE_ACC_SIZE; ++i)
        g_weights.base_l0_biases[i] = dist16(rng);
    for (int i = 0; i < BASE_FEATURE_SIZE; ++i)
        for (int j = 0; j < BASE_ACC_SIZE; ++j)
            g_weights.base_l0_weights[i][j] = dist16(rng);

    for (int i = 0; i < LINE_ACC_SIZE; ++i)
        g_weights.line_l0_biases[i] = dist16(rng);
    for (int i = 0; i < LINE_FEATURE_HASH; ++i)
        for (int j = 0; j < LINE_ACC_SIZE; ++j)
            g_weights.line_l0_weights[i][j] = dist16(rng);

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        g_weights.l1_biases[i] = dist32(rng);
        g_weights.l2_biases[i] = dist32(rng);
        g_weights.l3_weights[i] = dist8(rng);
        for (int j = 0; j < SPARSE_INPUT_SIZE; ++j)
            g_weights.l1_weights_t[i][j] = dist8(rng);
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            g_weights.l2_weights_t[i][j] = dist8(rng);
    }

    for (int i = 0; i < GLOBAL_FEATURE_SIZE; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            g_weights.global_l1_weights[i][j] = dist8(rng);
    for (int p = 0; p < PHASE_BUCKETS; ++p)
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            g_weights.phase_l1_biases[p][i] = dist32(rng);
}

bool load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        init_random_weights();
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size != expected_file_size()) {
        init_random_weights();
        return false;
    }

    WeightHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || !valid_header(header)) {
        init_random_weights();
        return false;
    }

    file.read(reinterpret_cast<char*>(g_weights.base_l0_biases), sizeof(g_weights.base_l0_biases));
    file.read(reinterpret_cast<char*>(g_weights.base_l0_weights), sizeof(g_weights.base_l0_weights));
    file.read(reinterpret_cast<char*>(g_weights.line_l0_biases), sizeof(g_weights.line_l0_biases));
    file.read(reinterpret_cast<char*>(g_weights.line_l0_weights), sizeof(g_weights.line_l0_weights));
    file.read(reinterpret_cast<char*>(g_weights.l1_biases), sizeof(g_weights.l1_biases));
    file.read(reinterpret_cast<char*>(g_weights.l1_weights_t), sizeof(g_weights.l1_weights_t));
    file.read(reinterpret_cast<char*>(g_weights.global_l1_weights), sizeof(g_weights.global_l1_weights));
    file.read(reinterpret_cast<char*>(g_weights.phase_l1_biases), sizeof(g_weights.phase_l1_biases));
    file.read(reinterpret_cast<char*>(g_weights.l2_biases), sizeof(g_weights.l2_biases));
    file.read(reinterpret_cast<char*>(g_weights.l2_weights_t), sizeof(g_weights.l2_weights_t));
    file.read(reinterpret_cast<char*>(g_weights.l3_biases), sizeof(g_weights.l3_biases));
    file.read(reinterpret_cast<char*>(g_weights.l3_weights), sizeof(g_weights.l3_weights));
    return bool(file);
}

void Accumulator::init() {
    for (int c = 0; c < 2; ++c) {
        std::memcpy(base[c], g_weights.base_l0_biases, sizeof(g_weights.base_l0_biases));
        std::memcpy(line[c], g_weights.line_l0_biases, sizeof(g_weights.line_l0_biases));
        std::fill(line_code[c], line_code[c] + NUM_LINES, 0u);
    }
}

Score evaluate_nnue(Color side_to_move,
                    const Accumulator& acc,
                    const int8_t global_features[GLOBAL_FEATURE_SIZE],
                    int phase_bucket) {
    alignas(64) int8_t sparse[SPARSE_INPUT_SIZE];
    int offset = 0;

    const Color us = side_to_move;
    const Color them = flip(us);
    clipped_relu_16to8(acc.base[us], sparse + offset, BASE_ACC_SIZE);
    offset += BASE_ACC_SIZE;
    clipped_relu_16to8(acc.base[them], sparse + offset, BASE_ACC_SIZE);
    offset += BASE_ACC_SIZE;
    clipped_relu_16to8(acc.line[us], sparse + offset, LINE_ACC_SIZE);
    offset += LINE_ACC_SIZE;
    clipped_relu_16to8(acc.line[them], sparse + offset, LINE_ACC_SIZE);

    phase_bucket = std::clamp(phase_bucket, 0, PHASE_BUCKETS - 1);

    alignas(64) int32_t hidden1[HIDDEN_SIZE];
    for (int o = 0; o < HIDDEN_SIZE; ++o) {
        int32_t sum = g_weights.l1_biases[o]
                    + g_weights.phase_l1_biases[phase_bucket][o]
                    + dot_u8_i8(sparse, g_weights.l1_weights_t[o], SPARSE_INPUT_SIZE);

        for (int j = 0; j < GLOBAL_FEATURE_SIZE; ++j)
            sum += static_cast<int32_t>(global_features[j])
                 * static_cast<int32_t>(g_weights.global_l1_weights[j][o]);
        hidden1[o] = sum;
    }

    alignas(64) int8_t hidden1_act[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        hidden1_act[i] = clipped_relu32(hidden1[i]);

    alignas(64) int32_t hidden2[HIDDEN_SIZE];
    for (int o = 0; o < HIDDEN_SIZE; ++o)
        hidden2[o] = g_weights.l2_biases[o]
                   + dot_u8_i8(hidden1_act, g_weights.l2_weights_t[o], HIDDEN_SIZE);

    alignas(64) int8_t hidden2_act[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        hidden2_act[i] = clipped_relu32(hidden2[i]);

    int32_t output = g_weights.l3_biases[0]
                   + dot_u8_i8(hidden2_act, g_weights.l3_weights, HIDDEN_SIZE);
    return output / 16;
}

void acc_add_weights(int16_t* acc_row, const int16_t* weights, int count) {
#ifdef NNUE_USE_AVX2
    for (int i = 0; i < count; i += 16) {
        const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_row + i));
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_row + i), _mm256_add_epi16(a, w));
    }
#else
    for (int i = 0; i < count; ++i)
        acc_row[i] += weights[i];
#endif
}

void acc_sub_weights(int16_t* acc_row, const int16_t* weights, int count) {
#ifdef NNUE_USE_AVX2
    for (int i = 0; i < count; i += 16) {
        const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_row + i));
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_row + i), _mm256_sub_epi16(a, w));
    }
#else
    for (int i = 0; i < count; ++i)
        acc_row[i] -= weights[i];
#endif
}

} // namespace NNUE

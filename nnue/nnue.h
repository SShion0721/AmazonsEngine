/*==================================================================
 * AMAZONS ENGINE - nnue.h
 * AmazonsNNUE-v2: base occupancy + line pattern + global fusion.
 *==================================================================*/
#pragma once

#include "../line_pattern.h"
#include "../types.h"
#include <cstdint>
#include <string>

namespace NNUE {

constexpr int BASE_FEATURE_SIZE   = 300;
constexpr int BASE_ACC_SIZE       = 512;
constexpr int LINE_ACC_SIZE       = 64;
constexpr int HIDDEN_SIZE         = 64;
constexpr int GLOBAL_FEATURE_SIZE = 32;
constexpr int PHASE_BUCKETS       = 8;
constexpr int SPARSE_INPUT_SIZE   = 2 * BASE_ACC_SIZE + 2 * LINE_ACC_SIZE;
constexpr int WEIGHT_VERSION      = 2;
constexpr int FT_QUANT_ONE        = 127;
constexpr int HIDDEN_WEIGHT_SCALE = 64;
constexpr int OUTPUT_WEIGHT_SCALE = 16;

struct NetworkParameters {
    alignas(64) int16_t base_l0_biases[BASE_ACC_SIZE];
    alignas(64) int16_t base_l0_weights[BASE_FEATURE_SIZE][BASE_ACC_SIZE];

    alignas(64) int16_t line_l0_biases[LINE_ACC_SIZE];
    alignas(64) int16_t line_l0_weights[LINE_FEATURE_HASH][LINE_ACC_SIZE];

    alignas(64) int32_t l1_biases[HIDDEN_SIZE];
    alignas(64) int8_t  l1_weights_t[HIDDEN_SIZE][SPARSE_INPUT_SIZE];
    alignas(64) int8_t  global_l1_weights[GLOBAL_FEATURE_SIZE][HIDDEN_SIZE];
    alignas(64) int32_t phase_l1_biases[PHASE_BUCKETS][HIDDEN_SIZE];

    alignas(64) int32_t l2_biases[HIDDEN_SIZE];
    alignas(64) int8_t  l2_weights_t[HIDDEN_SIZE][HIDDEN_SIZE];

    alignas(64) int32_t l3_biases[1];
    alignas(64) int8_t  l3_weights[HIDDEN_SIZE];
};

extern NetworkParameters g_weights;

void init_random_weights();
bool load_weights(const std::string& path);
const char* weight_format();

struct Accumulator {
    alignas(64) int16_t base[2][BASE_ACC_SIZE];
    alignas(64) int16_t line[2][LINE_ACC_SIZE];
    uint32_t line_code[2][NUM_LINES];

    void init();
};

inline int feature_index(Color perspective, int8_t cell_type, Square sq) {
    int offset = 0;
    if (cell_type == ARROW) {
        offset = 200;
    } else {
        const bool is_mine = (cell_type == WHITE_AMAZON && perspective == WHITE)
                          || (cell_type == BLACK_AMAZON && perspective == BLACK);
        offset = is_mine ? 0 : 100;
    }
    return offset + sq;
}

Score evaluate_nnue(Color side_to_move,
                    const Accumulator& acc,
                    const int8_t global_features[GLOBAL_FEATURE_SIZE],
                    int phase_bucket);

void acc_add_weights(int16_t* acc_row, const int16_t* weights, int count);
void acc_sub_weights(int16_t* acc_row, const int16_t* weights, int count);

} // namespace NNUE

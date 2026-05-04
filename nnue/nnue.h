/*==================================================================
 * AMAZONS ENGINE — nnue.h
 * Neural Network Updated Efficiently (NNUE) definitions.
 *
 * Architecture:
 *   Input: 300 binary features (My Amazons, Enemy Amazons, Arrows)
 *   Accumulator: 512 (per color perspective)
 *   Concat: 1024 (us + them)
 *   Hidden 1: 64
 *   Hidden 2: 64
 *   Output: 1 (score from side-to-move's perspective)
 *==================================================================*/
#pragma once
#include "../types.h"
#include <array>
#include <cstdint>
#include <vector>
#include <string>

namespace NNUE {

// ── Network Topology ──────────────────────────────────────────────────
constexpr int FEATURE_SIZE = 300;
constexpr int ACC_SIZE     = 512;
constexpr int HIDDEN_SIZE  = 64;

// ── Model Weights (Quantized) ─────────────────────────────────────────
// Stored in INT8 / INT16 for fast CPU inference.
struct NetworkParameters {
    // Layer 0: Sparse Input -> Accumulator
    // Biases are int16. Weights are int16.
    alignas(64) int16_t l0_biases[ACC_SIZE];
    alignas(64) int16_t l0_weights[FEATURE_SIZE][ACC_SIZE];

    // Layer 1: Concat (1024) -> Hidden (64)
    // Biases are int32. Weights are int8.
    alignas(64) int32_t l1_biases[HIDDEN_SIZE];
    alignas(64) int8_t  l1_weights[ACC_SIZE * 2][HIDDEN_SIZE]; // [1024][64]

    // Layer 2: Hidden (64) -> Hidden (64)
    alignas(64) int32_t l2_biases[HIDDEN_SIZE];
    alignas(64) int8_t  l2_weights[HIDDEN_SIZE][HIDDEN_SIZE];

    // Layer 3: Hidden (64) -> Output (1)
    alignas(64) int32_t l3_biases[1];
    alignas(64) int8_t  l3_weights[HIDDEN_SIZE][1];
};

extern NetworkParameters g_weights;

// Initialize weights to random/zero for testing until we train a real model
void init_random_weights();

// Load weights from a binary file
bool load_weights(const std::string& path);

// ── Accumulator ───────────────────────────────────────────────────────
// The continuously updated first layer.
struct Accumulator {
    alignas(64) int16_t state[2][ACC_SIZE]; // state[Color][512]

    // Fill with biases
    void init() {
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < ACC_SIZE; ++i) {
                state[c][i] = g_weights.l0_biases[i];
            }
        }
    }
};

// ── Feature Indices ───────────────────────────────────────────────────
// Maps a board piece to its 0..299 feature index from a given perspective.
inline int feature_index(Color perspective, int8_t cell_type, Square sq) {
    // Base offsets:
    // 0: My Amazons
    // 100: Enemy Amazons
    // 200: Arrows
    int offset = 0;
    if (cell_type == ARROW) {
        offset = 200;
    } else {
        bool is_mine = (cell_type == WHITE_AMAZON && perspective == WHITE) ||
                       (cell_type == BLACK_AMAZON && perspective == BLACK);
        offset = is_mine ? 0 : 100;
    }
    return offset + sq;
}

// ── Forward Pass ──────────────────────────────────────────────────────
// Returns the NNUE evaluation from the side_to_move's perspective
Score evaluate_nnue(Color side_to_move, const Accumulator& acc);

// ── SIMD-accelerated accumulator update helpers ──────────────────────
// Used by Position::do_move() for incremental feature updates.
// AVX2 path processes 16 int16s per cycle; scalar fallback otherwise.
void acc_add_weights(int16_t* acc_row, const int16_t* weights, int count);
void acc_sub_weights(int16_t* acc_row, const int16_t* weights, int count);

} // namespace NNUE

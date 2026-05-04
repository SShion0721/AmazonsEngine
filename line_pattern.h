#pragma once

#include "types.h"
#include <cstdint>

constexpr int NUM_LINES = 58;
constexpr int LINE_FEATURE_HASH = 1 << 16;
constexpr uint64_t LINE_SEED_BASE = 0xA17A5015BEE5C0DEULL;

struct LineRef {
    uint8_t line;
    uint8_t shift;
};

struct LineFeature {
    uint16_t index;
    int8_t sign;
};

extern LineRef SQ_LINES[BOARD_SQ][4];
extern uint32_t LINE_SEED[NUM_LINES];

void init_line_patterns();

inline uint32_t line_cell_code(Color perspective, int8_t cell) {
    if (cell == EMPTY)
        return 0;
    if (cell == ARROW)
        return 3;

    const bool own = (cell == WHITE_AMAZON && perspective == WHITE)
                  || (cell == BLACK_AMAZON && perspective == BLACK);
    return own ? 1u : 2u;
}

inline uint32_t mix_line_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline LineFeature line_feature(uint8_t line, uint32_t code) {
    const uint32_t h = mix_line_hash(code ^ LINE_SEED[line] ^ (uint32_t(line) * 0x9e3779b9u));
    return {static_cast<uint16_t>(h & (LINE_FEATURE_HASH - 1)), static_cast<int8_t>((h & 0x80000000u) ? 1 : -1)};
}

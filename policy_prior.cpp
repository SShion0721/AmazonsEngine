/*==================================================================
 * AMAZONS ENGINE - policy_prior.cpp
 * AMZPOL1: a tiny factorized linear policy prior.
 *==================================================================*/
#include "policy_prior.h"

#include <algorithm>
#include <cstring>
#include <fstream>

bool g_use_policy_prior = true;
int g_policy_prior_blend = 70;

namespace {

constexpr char POLICY_MAGIC[8] = {'A', 'M', 'Z', 'P', 'O', 'L', '1', '\0'};
constexpr uint32_t POLICY_VERSION = 1;
constexpr int POLICY_FEATURES = 300;

#pragma pack(push, 1)
struct PolicyHeader {
    char magic[8];
    uint32_t version;
    uint32_t board_squares;
    uint32_t feature_size;
};
#pragma pack(pop)

struct PolicyWeights {
    alignas(64) int16_t bias[3][BOARD_SQ]{};
    alignas(64) int8_t weights[3][POLICY_FEATURES][BOARD_SQ]{};
};

PolicyWeights g_policy_weights;
bool g_policy_loaded = false;
const char* g_policy_format = "none";

int feature_index(Color perspective, int8_t cell, Square sq) {
    int offset = 0;
    if (cell == ARROW) {
        offset = 200;
    } else {
        const bool mine = (cell == WHITE_AMAZON && perspective == WHITE)
                       || (cell == BLACK_AMAZON && perspective == BLACK);
        offset = mine ? 0 : 100;
    }
    return offset + sq;
}

std::streamoff expected_size() {
    return static_cast<std::streamoff>(sizeof(PolicyHeader)
        + 3 * BOARD_SQ * sizeof(int16_t)
        + 3 * POLICY_FEATURES * BOARD_SQ * sizeof(int8_t));
}

bool valid_header(const PolicyHeader& h) {
    return std::memcmp(h.magic, POLICY_MAGIC, sizeof(POLICY_MAGIC)) == 0
        && h.version == POLICY_VERSION
        && h.board_squares == BOARD_SQ
        && h.feature_size == POLICY_FEATURES;
}

} // namespace

bool load_policy_prior(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::memset(&g_policy_weights, 0, sizeof(g_policy_weights));
        g_policy_loaded = false;
        g_policy_format = "none";
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size != expected_size()) {
        std::memset(&g_policy_weights, 0, sizeof(g_policy_weights));
        g_policy_loaded = false;
        g_policy_format = "invalid";
        return false;
    }

    PolicyHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || !valid_header(header)) {
        std::memset(&g_policy_weights, 0, sizeof(g_policy_weights));
        g_policy_loaded = false;
        g_policy_format = "invalid";
        return false;
    }

    in.read(reinterpret_cast<char*>(g_policy_weights.bias), sizeof(g_policy_weights.bias));
    in.read(reinterpret_cast<char*>(g_policy_weights.weights), sizeof(g_policy_weights.weights));
    g_policy_loaded = bool(in);
    g_policy_format = g_policy_loaded ? "AMZPOL1" : "invalid";
    if (!g_policy_loaded)
        std::memset(&g_policy_weights, 0, sizeof(g_policy_weights));
    return g_policy_loaded;
}

bool policy_prior_loaded() {
    return g_policy_loaded;
}

const char* policy_prior_format() {
    return g_policy_format;
}

PolicyPrior evaluate_policy_prior(const Position& pos) {
    PolicyPrior out{};
    if (!g_use_policy_prior || !g_policy_loaded)
        return out;

    for (int sq = 0; sq < BOARD_SQ; ++sq) {
        out.from[sq] = g_policy_weights.bias[0][sq];
        out.to[sq] = g_policy_weights.bias[1][sq];
        out.arrow[sq] = g_policy_weights.bias[2][sq];
    }

    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        const int8_t cell = pos.board[sq];
        if (cell == EMPTY)
            continue;
        const int idx = feature_index(pos.side_to_move, cell, sq);
        for (int dst = 0; dst < BOARD_SQ; ++dst) {
            out.from[dst] = static_cast<int16_t>(std::clamp<int>(
                out.from[dst] + g_policy_weights.weights[0][idx][dst], -32768, 32767));
            out.to[dst] = static_cast<int16_t>(std::clamp<int>(
                out.to[dst] + g_policy_weights.weights[1][idx][dst], -32768, 32767));
            out.arrow[dst] = static_cast<int16_t>(std::clamp<int>(
                out.arrow[dst] + g_policy_weights.weights[2][idx][dst], -32768, 32767));
        }
    }

    return out;
}

int policy_move_prior(const PolicyPrior& prior, Move m) {
    if (m == MOVE_NONE)
        return 0;
    return int(prior.from[move_from(m)])
         + int(prior.to[move_to(m)])
         + int(prior.arrow[move_arrow(m)]);
}

#include "../bitboard.h"
#include "../evaluate.h"
#include "../line_pattern.h"
#include "../movegen.h"
#include "../position.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

void init_engine_tables() {
    Zobrist::init();
    init_rays();
    init_bitboards();
    init_attack_tables();
    init_between_line_bb();
    init_line_patterns();
    NNUE::init_random_weights();
    clear_eval_cache();
}

bool same_accumulator(const NNUE::Accumulator& a, const NNUE::Accumulator& b) {
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < NNUE::BASE_ACC_SIZE; ++i)
            if (a.base[c][i] != b.base[c][i])
                return false;
        for (int i = 0; i < NNUE::LINE_ACC_SIZE; ++i)
            if (a.line[c][i] != b.line[c][i])
                return false;
        for (int i = 0; i < NUM_LINES; ++i)
            if (a.line_code[c][i] != b.line_code[c][i])
                return false;
    }
    return true;
}

bool verify_accumulator(const Position& pos, const char* context) {
    auto full = std::make_unique<Position>(pos);
    full->compute_accumulator();
    if (same_accumulator(pos.history[pos.ply].accumulator,
                         full->history[full->ply].accumulator))
        return true;

    std::cerr << "Accumulator mismatch at " << context
              << " ply=" << pos.ply
              << " side=" << (pos.side_to_move == WHITE ? "white" : "black")
              << "\n";
    return false;
}

bool verify_generated_moves(const Position& pos, std::mt19937& rng, const char* context) {
    std::array<Move, MAX_LEGAL_MOVES> moves{};
    const int count = generate_moves(pos, moves.data(), MAX_LEGAL_MOVES);
    std::vector<uint8_t> seen(1 << 21, 0);

    for (int i = 0; i < count; ++i) {
        const Move m = moves[i];
        if (m <= 0 || m >= static_cast<int>(seen.size())) {
            std::cerr << "Generated out-of-range move at " << context << "\n";
            return false;
        }
        if (seen[m]) {
            std::cerr << "Duplicate move generated: " << move_to_str(m)
                      << " at " << context << "\n";
            return false;
        }
        seen[m] = 1;
        if (!is_pseudo_legal(pos, m)) {
            std::cerr << "Generated move rejected by is_pseudo_legal: "
                      << move_to_str(m) << " at " << context << "\n";
            return false;
        }
    }

    std::uniform_int_distribution<int> sq_dist(0, BOARD_SQ - 1);
    for (int i = 0; i < 2048; ++i) {
        const Move m = encode_move(sq_dist(rng), sq_dist(rng), sq_dist(rng));
        const bool in_generator = m > 0 && m < static_cast<int>(seen.size()) && seen[m];
        const bool legal = is_pseudo_legal(pos, m);
        if (in_generator != legal) {
            std::cerr << "Generator/pseudo mismatch for " << move_to_str(m)
                      << " at " << context
                      << " generated=" << in_generator
                      << " pseudo=" << legal << "\n";
            return false;
        }
    }

    return true;
}

bool verify_undo_roundtrip(Position& pos, Move m, const char* context) {
    const std::string board = pos.get_board_string();
    Square amazons[2][NUM_AMAZONS];
    std::memcpy(amazons, pos.amazons, sizeof(amazons));
    const Color side = pos.side_to_move;
    const uint64_t key = pos.key;
    const Bitboard128 white = pos.bb_white;
    const Bitboard128 black = pos.bb_black;
    const Bitboard128 arrows = pos.bb_arrow;
    const Bitboard128 occupied = pos.bb_occupied;
    const int ply = pos.ply;

    pos.do_move(m);
    pos.undo_move(m);

    if (pos.get_board_string() != board
        || pos.side_to_move != side
        || pos.key != key
        || !(pos.bb_white == white)
        || !(pos.bb_black == black)
        || !(pos.bb_arrow == arrows)
        || !(pos.bb_occupied == occupied)
        || pos.ply != ply
        || !std::equal(&pos.amazons[0][0], &pos.amazons[0][0] + 2 * NUM_AMAZONS, &amazons[0][0])) {
        std::cerr << "Undo roundtrip mismatch after " << move_to_str(m)
                  << " at " << context << "\n";
        return false;
    }

    return verify_accumulator(pos, context);
}

} // namespace

int main() {
    init_engine_tables();
    std::mt19937 rng(0xA5015);
    auto pos = std::make_unique<Position>();

    for (int game = 0; game < 8; ++game) {
        pos->set_startpos();
        for (int ply = 0; ply < 80; ++ply) {
            const std::string context = "game " + std::to_string(game)
                                      + ", ply " + std::to_string(ply);

            if (!verify_accumulator(*pos, context.c_str()))
                return 1;
            if (!verify_generated_moves(*pos, rng, context.c_str()))
                return 1;

            std::array<Move, MAX_LEGAL_MOVES> moves{};
            const int count = generate_moves(*pos, moves.data(), MAX_LEGAL_MOVES);
            if (count == 0)
                break;

            std::uniform_int_distribution<int> move_dist(0, count - 1);
            const Move m = moves[move_dist(rng)];
            if (!verify_undo_roundtrip(*pos, m, context.c_str()))
                return 1;

            pos->do_move(m);
        }
    }

    std::cout << "verify_refactor: accumulator, movegen, pseudo-legal, undo checks passed\n";
    return 0;
}

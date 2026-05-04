/*==================================================================
 * AMAZONS ENGINE — position.cpp
 * Implements Position::do_move, undo_move, set_startpos, print.
 *==================================================================*/
#include "position.h"
#include <iostream>
#include <random>
#include <cassert>

// ── Zobrist implementation ────────────────────────────────────────
namespace Zobrist {
    uint64_t piece_sq[3][BOARD_SQ];
    uint64_t side;

    void init() {
        std::mt19937_64 rng(0xDEADBEEFCAFEBABEULL);
        for (auto& row : piece_sq)
            for (auto& x : row)
                x = rng();
        side = rng();
    }
}

// Map board cell value (1/2/3) → Zobrist index (0/1/2)
static inline int piece_idx(int8_t cell) { return cell - 1; }

namespace {

void add_line_feature(NNUE::Accumulator& acc, Color p, uint8_t line, uint32_t code) {
    const LineFeature f = line_feature(line, code);
    const int16_t* weights = NNUE::g_weights.line_l0_weights[f.index];
    if (f.sign > 0)
        NNUE::acc_add_weights(acc.line[p], weights, NNUE::LINE_ACC_SIZE);
    else
        NNUE::acc_sub_weights(acc.line[p], weights, NNUE::LINE_ACC_SIZE);
}

void remove_line_feature(NNUE::Accumulator& acc, Color p, uint8_t line, uint32_t code) {
    const LineFeature f = line_feature(line, code);
    const int16_t* weights = NNUE::g_weights.line_l0_weights[f.index];
    if (f.sign > 0)
        NNUE::acc_sub_weights(acc.line[p], weights, NNUE::LINE_ACC_SIZE);
    else
        NNUE::acc_add_weights(acc.line[p], weights, NNUE::LINE_ACC_SIZE);
}

void update_line_square(NNUE::Accumulator& acc,
                        Color p,
                        Square sq,
                        int8_t old_cell,
                        int8_t new_cell) {
    const uint32_t old_piece = line_cell_code(p, old_cell);
    const uint32_t new_piece = line_cell_code(p, new_cell);
    if (old_piece == new_piece)
        return;

    for (int k = 0; k < 4; ++k) {
        const LineRef lr = SQ_LINES[sq][k];
        uint32_t& code = acc.line_code[p][lr.line];

        remove_line_feature(acc, p, lr.line, code);
        code &= ~(3u << lr.shift);
        code |= (new_piece << lr.shift);
        add_line_feature(acc, p, lr.line, code);
    }
}

} // namespace

// ── set_startpos ─────────────────────────────────────────────────
/*  Standard Game of the Amazons starting position:
 *
 *    . . . B . . B . . .    ← row 9 (rank 10): d10=93, g10=96
 *    . . . . . . . . . .
 *    B . . . . . . . . B    ← row 6 (rank 7):  a7=60, j7=69
 *    . . . . . . . . . .
 *    . . . . . . . . . .
 *    . . . . . . . . . .
 *    W . . . . . . . . W    ← row 3 (rank 4):  a4=30, j4=39
 *    . . . . . . . . . .
 *    . . . W . . W . . .    ← row 0 (rank 1):  d1=3,  g1=6
 */
void Position::set_startpos() {
    memset(board, EMPTY, sizeof(board));
    side_to_move = WHITE;
    ply = 0;

    // White amazons: a4, d1, g1, j4
    amazons[WHITE][0] = 30; // a4
    amazons[WHITE][1] =  3; // d1
    amazons[WHITE][2] =  6; // g1
    amazons[WHITE][3] = 39; // j4

    // Black amazons: a7, d10, g10, j7
    amazons[BLACK][0] = 60; // a7
    amazons[BLACK][1] = 93; // d10
    amazons[BLACK][2] = 96; // g10
    amazons[BLACK][3] = 69; // j7

    // Place on board
    for (int i = 0; i < NUM_AMAZONS; i++) {
        board[amazons[WHITE][i]] = WHITE_AMAZON;
        board[amazons[BLACK][i]] = BLACK_AMAZON;
    }

    // Compute Zobrist key from scratch
    key = 0;
    for (int i = 0; i < NUM_AMAZONS; i++) {
        key ^= Zobrist::piece_sq[piece_idx(WHITE_AMAZON)][amazons[WHITE][i]];
        key ^= Zobrist::piece_sq[piece_idx(BLACK_AMAZON)][amazons[BLACK][i]];
    }
    // WHITE to move → don't XOR Zobrist::side

    // Initialize bitboards
    bb_white = Bitboard128::zero();
    bb_black = Bitboard128::zero();
    bb_arrow = Bitboard128::zero();
    for (int i = 0; i < NUM_AMAZONS; i++) {
        bb_white.set(amazons[WHITE][i]);
        bb_black.set(amazons[BLACK][i]);
    }
    bb_occupied = bb_white | bb_black | bb_arrow;
    
    compute_accumulator();
}

// ── compute_accumulator ──────────────────────────────────────────
void Position::compute_accumulator() {
    history[ply].accumulator.init();
    auto& acc = history[ply].accumulator;

    for (int sq = 0; sq < BOARD_SQ; ++sq) {
        if (board[sq] != EMPTY) {
            int idx_w = NNUE::feature_index(WHITE, board[sq], sq);
            int idx_b = NNUE::feature_index(BLACK, board[sq], sq);
            NNUE::acc_add_weights(acc.base[WHITE], NNUE::g_weights.base_l0_weights[idx_w], NNUE::BASE_ACC_SIZE);
            NNUE::acc_add_weights(acc.base[BLACK], NNUE::g_weights.base_l0_weights[idx_b], NNUE::BASE_ACC_SIZE);

            for (Color p : {WHITE, BLACK}) {
                const uint32_t piece = line_cell_code(p, board[sq]);
                for (int k = 0; k < 4; ++k) {
                    const LineRef lr = SQ_LINES[sq][k];
                    acc.line_code[p][lr.line] &= ~(3u << lr.shift);
                    acc.line_code[p][lr.line] |= (piece << lr.shift);
                }
            }
        }
    }

    for (Color p : {WHITE, BLACK})
        for (int line = 0; line < NUM_LINES; ++line)
            add_line_feature(acc, p, static_cast<uint8_t>(line), acc.line_code[p][line]);
}

// ── do_move ──────────────────────────────────────────────────────
// Execute a move and update all state incrementally.
// This is the Amazons equivalent of SF's Position::do_move().
void Position::do_move(Move m) {
    assert(ply < MAX_PLY - 1);

    const Square from  = move_from(m);
    const Square to    = move_to(m);
    const Square arrow = move_arrow(m);
    const Color  us    = side_to_move;
    const int8_t amazon_val = (us == WHITE) ? WHITE_AMAZON : BLACK_AMAZON;

    // Save state for undo
    history[ply].key = key;

    // --- 0. Update NNUE Accumulator incrementally (SIMD-accelerated) ---
    history[ply + 1].accumulator = history[ply].accumulator;
    auto& acc = history[ply + 1].accumulator;
    
    auto update_feat = [&](int8_t type, Square sq, int sign) {
        int idx_w = NNUE::feature_index(WHITE, type, sq);
        int idx_b = NNUE::feature_index(BLACK, type, sq);
        if (sign == 1) {
            NNUE::acc_add_weights(acc.base[WHITE], NNUE::g_weights.base_l0_weights[idx_w], NNUE::BASE_ACC_SIZE);
            NNUE::acc_add_weights(acc.base[BLACK], NNUE::g_weights.base_l0_weights[idx_b], NNUE::BASE_ACC_SIZE);
        } else {
            NNUE::acc_sub_weights(acc.base[WHITE], NNUE::g_weights.base_l0_weights[idx_w], NNUE::BASE_ACC_SIZE);
            NNUE::acc_sub_weights(acc.base[BLACK], NNUE::g_weights.base_l0_weights[idx_b], NNUE::BASE_ACC_SIZE);
        }
    };

    update_feat(amazon_val, from, -1); // remove from
    update_feat(amazon_val, to,    1); // place to
    update_feat(ARROW,      arrow, 1); // place arrow

    for (Color p : {WHITE, BLACK}) {
        update_line_square(acc, p, from, amazon_val, EMPTY);
        update_line_square(acc, p, to, EMPTY, amazon_val);
        update_line_square(acc, p, arrow, EMPTY, ARROW);
    }

    // --- 1. Update Zobrist key (incremental, like SF) ---
    key ^= Zobrist::piece_sq[piece_idx(amazon_val)][from];   // remove from 'from'
    key ^= Zobrist::piece_sq[piece_idx(amazon_val)][to];     // place at 'to'
    key ^= Zobrist::piece_sq[piece_idx(ARROW)][arrow];       // place arrow
    key ^= Zobrist::side;                                     // flip side

    // --- 2. Update board array ---
    board[from]  = EMPTY;
    board[to]    = amazon_val;
    board[arrow] = ARROW;

    // --- 3. Update bitboards (3 XOR/OR ops) ---
    Bitboard128& us_bb = (us == WHITE) ? bb_white : bb_black;
    us_bb ^= SQUARE_BB[from] | SQUARE_BB[to];   // move amazon
    bb_arrow |= SQUARE_BB[arrow];                // place arrow
    bb_occupied = bb_white | bb_black | bb_arrow; // recompute

    // --- 4. Update amazon square list ---
    int idx = amazon_index(us, from);
    assert(idx != -1);
    amazons[us][idx] = to;

    // --- 4. Flip side ---
    side_to_move = flip(us);
    ply++;
}

// ── undo_move ────────────────────────────────────────────────────
// Reverse a move. Because there are no captures in Amazons, undo
// only needs to: remove the arrow, move the amazon back, flip side.
void Position::undo_move(Move m) {
    assert(ply > 0);
    ply--;

    const Square from  = move_from(m);
    const Square to    = move_to(m);
    const Square arrow = move_arrow(m);
    const Color  us    = flip(side_to_move); // who just moved
    const int8_t amazon_val = (us == WHITE) ? WHITE_AMAZON : BLACK_AMAZON;

    // Restore board
    board[arrow] = EMPTY;      // remove arrow
    board[to]    = EMPTY;      // remove amazon from destination
    board[from]  = amazon_val; // restore amazon to origin

    // Restore bitboards
    Bitboard128& us_bb = (us == WHITE) ? bb_white : bb_black;
    bb_arrow &= ~SQUARE_BB[arrow];
    us_bb ^= SQUARE_BB[from] | SQUARE_BB[to];
    bb_occupied = bb_white | bb_black | bb_arrow;

    // Restore amazon list
    int idx = amazon_index(us, to);
    assert(idx != -1);
    amazons[us][idx] = from;

    // Restore Zobrist key and side
    key = history[ply].key;
    side_to_move = us;
}

// ── do_null_move ─────────────────────────────────────────────────
// For Null Move Pruning: just flip side_to_move, no piece movement.
// Mirrors SF's Position::do_null_move().
void Position::do_null_move() {
    assert(ply < MAX_PLY - 1);

    // Save state for undo
    history[ply].key = key;

    // Copy accumulator forward (no features change)
    history[ply + 1].accumulator = history[ply].accumulator;

    // Flip side and update hash
    key ^= Zobrist::side;
    side_to_move = flip(side_to_move);
    ply++;
}

// ── undo_null_move ───────────────────────────────────────────────
void Position::undo_null_move() {
    assert(ply > 0);
    ply--;
    key = history[ply].key;
    side_to_move = flip(side_to_move);
}

// ── print ────────────────────────────────────────────────────────
void Position::print() const {
    const char* sym = ".WBX"; // EMPTY, WHITE_AMAZON, BLACK_AMAZON, ARROW
    std::cout << "\n  a b c d e f g h i j\n";
    for (int r = 9; r >= 0; r--) {
        std::cout << (r + 1) << " ";
        for (int c = 0; c < 10; c++)
            std::cout << sym[(int)board[make_sq(r, c)]] << " ";
        std::cout << "\n";
    }
    std::cout << (side_to_move == WHITE ? "  White to move\n" : "  Black to move\n");
}

std::string Position::get_board_string() const {
    std::string s(BOARD_SQ, '0');
    // We print a string of length 100.
    // Index i corresponds to sq `i`
    for (int i = 0; i < BOARD_SQ; ++i) {
        if (board[i] == WHITE_AMAZON) s[i] = '1';
        else if (board[i] == BLACK_AMAZON) s[i] = '2';
        else if (board[i] == ARROW) s[i] = '3';
        else s[i] = '0';
    }
    return s;
}

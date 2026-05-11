/*==================================================================
 * AMAZONS ENGINE — position.h
 * Board state, Zobrist hashing, do_move / undo_move.
 * Mirrors Stockfish's position.h / position.cpp
 *==================================================================*/
#pragma once
#include "types.h"
#include "rays.h"
#include "bitboard.h"
#include "nnue/nnue.h"
#include <array>
#include <cstring>

// ── Zobrist hash tables ───────────────────────────────────────────
// Mirrors Stockfish's Zobrist namespace.
// One 64-bit random number per (piece_type, square) combination.
namespace Zobrist {
    // Index 0 = WHITE_AMAZON (1), 1 = BLACK_AMAZON (2), 2 = ARROW (3)
    // We shift piece value by 1 to use as index.
    extern uint64_t piece_sq[3][BOARD_SQ];
    extern uint64_t side;    // XOR'd in when it's BLACK's turn
    void init();             // call once at startup
}

// ── UndoInfo ─────────────────────────────────────────────────────
// Stores everything needed to undo a move (like SF's StateInfo).
struct UndoInfo {
    uint64_t key;   // Zobrist key before the move
    NNUE::Accumulator accumulator; // NNUE layer 0 state
};

// ── Position ─────────────────────────────────────────────────────
// The complete board state. Mirrors Stockfish's Position class.
struct Position {
    // Board: 100 cells, each is EMPTY / WHITE_AMAZON / BLACK_AMAZON / ARROW
    int8_t  board[BOARD_SQ];

    // Explicit amazon square lists for fast iteration
    // amazons[color][0..3] = squares of that color's four amazons
    Square  amazons[2][NUM_AMAZONS];

    Color   side_to_move;
    uint64_t key;       // running Zobrist hash
    int      ply;       // half-moves played from root

    // ── Bitboards (maintained alongside board[] for fast bulk ops) ─
    Bitboard128 bb_white;    // white amazon positions
    Bitboard128 bb_black;    // black amazon positions
    Bitboard128 bb_arrow;    // arrow positions
    Bitboard128 bb_occupied; // bb_white | bb_black | bb_arrow

    // Undo stack
    // Amazons games can run much longer than chess-like search plies.
    static constexpr int MAX_PLY = 256;
    UndoInfo history[MAX_PLY];

    // ── Setup ────────────────────────────────────────────────────
    void set_startpos();
    void set_from_fen(const std::string& fen); // optional extension
    void compute_accumulator();              // recompute NNUE acc from scratch

    // ── Queries ──────────────────────────────────────────────────
    bool is_occupied(Square sq) const { return board[sq] != EMPTY; }
    bool is_empty   (Square sq) const { return board[sq] == EMPTY;  }
    
    // Returns a 100-character compact string representing the board:
    // '0'=EMPTY, '1'=WHITE_AMAZON, '2'=BLACK_AMAZON, '3'=ARROW
    std::string get_board_string() const;

    // Count queen-reachable squares from sq (used in evaluation)
    int queen_reach(Square sq) const {
        return (get_queen_attacks(sq, bb_occupied) & ~bb_occupied).popcount();
    }

    // ── Move execution ───────────────────────────────────────────
    void do_move  (Move m);
    void undo_move(Move m);

    // ── Null move (for NMP: just flip side, no piece movement) ───
    void do_null_move();
    void undo_null_move();

    // ── Debug ────────────────────────────────────────────────────
    void print() const;

private:
    // Find the index (0..3) of the amazon at square sq for color c.
    // Called during do_move to update the amazon[c] array.
    int amazon_index(Color c, Square sq) const {
        for (int i = 0; i < NUM_AMAZONS; i++)
            if (amazons[c][i] == sq) return i;
        return -1; // should not happen
    }
};

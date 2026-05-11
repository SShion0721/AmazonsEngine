/*==================================================================
 * AMAZONS ENGINE — types.h
 * Core types, constants, and board coordinate utilities.
 * Mirrors Stockfish's types.h
 *==================================================================*/
#pragma once
#include <cstdint>
#include <cctype>
#include <string>

// ── Basic types ──────────────────────────────────────────────────
using Square = int;    // 0..99, flat index on the 10×10 board
using Score  = int;    // evaluation score ("centipawns equivalent")
using Move   = int;    // 21-bit encoded move
using Color  = int;    // 0 = WHITE, 1 = BLACK
using Depth  = int;

// ── Colors ───────────────────────────────────────────────────────
constexpr Color WHITE = 0, BLACK = 1;
inline Color    flip(Color c) { return c ^ 1; }

// ── Board constants ───────────────────────────────────────────────
constexpr int NUM_AMAZONS = 4;    // each side has 4 amazons
constexpr int BOARD_SQ    = 100;  // 10×10 = 100 squares

// ── Score limits ─────────────────────────────────────────────────
constexpr Score SCORE_INF  =     32000;
constexpr Score SCORE_MATE =     30000;  // Must fit in TT's int16 score storage
constexpr Score SCORE_ZERO =          0;
constexpr Score SCORE_NONE =    -32001;  // sentinel outside the search window

// ── Board cell values ────────────────────────────────────────────
constexpr int8_t EMPTY        = 0;
constexpr int8_t WHITE_AMAZON = 1;
constexpr int8_t BLACK_AMAZON = 2;
constexpr int8_t ARROW        = 3;

// ── Move encoding: [from:7][to:7][arrow:7] = 21 bits ────────────
//   from  = source square of the amazon being moved
//   to    = destination square of the amazon
//   arrow = square where the fired arrow lands
inline Move   encode_move (Square from, Square to, Square arrow)
              { return (from << 14) | (to << 7) | arrow; }
inline Square move_from   (Move m) { return (m >> 14) & 0x7F; }
inline Square move_to     (Move m) { return (m >>  7) & 0x7F; }
inline Square move_arrow  (Move m) { return  m        & 0x7F; }
constexpr Move MOVE_NONE = 0;

// ── Board coordinate helpers ─────────────────────────────────────
/*  Square layout (row-major, bottom-left = 0):
 *
 *   90 91 92 93 94 95 96 97 98 99   — rank 10 (top)
 *   80 81 82 83 84 85 86 87 88 89
 *   ...
 *    0  1  2  3  4  5  6  7  8  9   — rank 1 (bottom)
 *    a  b  c  d  e  f  g  h  i  j
 */
inline int    sq_row   (Square sq)        { return sq / 10; }
inline int    sq_col   (Square sq)        { return sq % 10; }
inline Square make_sq  (int r, int c)     { return r * 10 + c; }
inline bool   valid_rc (int r, int c)     { return r >= 0 && r < 10 && c >= 0 && c < 10; }
inline bool   valid_sq (Square sq)        { return sq >= 0 && sq < 100; }

// Convert square to algebraic notation (e.g. 0 — "a1", 99 — "j10")
inline std::string sq_to_str(Square sq) {
    return std::string(1, 'a' + sq_col(sq)) + std::to_string(sq_row(sq) + 1);
}

// Convert algebraic notation to square (e.g. "a1" — 0)
inline Square str_to_sq(const std::string& s) {
    if (s.size() < 2) return -1;
    int c = s[0] - 'a';
    if (c < 0 || c >= 10) return -1;

    int r = 0;
    for (size_t i = 1; i < s.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(s[i]);
        if (!std::isdigit(ch)) return -1;
        r = r * 10 + (ch - '0');
    }
    r -= 1;
    return valid_rc(r, c) ? make_sq(r, c) : -1;
}

// Convert move to string (e.g. "a4-d7/d3")
inline std::string move_to_str(Move m) {
    if (m == MOVE_NONE) return "0000";
    return sq_to_str(move_from(m)) + "-" +
           sq_to_str(move_to(m))   + "/" +
           sq_to_str(move_arrow(m));
}

// ── 8 Queen directions: N S E W NE SW NW SE ─────────────────────
constexpr int DR[8] = { 1, -1,  0,  0,  1, -1,  1, -1 };
constexpr int DC[8] = { 0,  0,  1, -1,  1, -1, -1,  1 };

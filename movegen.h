/*==================================================================
 * AMAZONS ENGINE — movegen.h / movegen.cpp
 * Legal move generation for the Game of the Amazons.
 *
 * Each move = (amazon_from, amazon_to, arrow_square).
 * Both the amazon move and the arrow shot follow queen-sliding rules.
 *
 * Algorithm (mirrors SF's generate_all() with magic bitboards,
 * but uses precomputed RAYS instead):
 *
 *   For each friendly amazon at 'from':
 *     1. Temporarily lift the amazon (board[from] = EMPTY)
 *     2. Slide in all 8 directions to find legal 'to' squares
 *     3. For each 'to': temporarily place amazon, then slide again
 *        from 'to' to find legal 'arrow' squares
 *     4. Emit move(from, to, arrow) for each combination
 *     5. Restore board state
 *
 * The temporary lift in step 1 correctly allows the arrow to land
 * on the amazon's original square (which is now empty).
 *==================================================================*/
#pragma once
#include "position.h"
#include <vector>

// Generate all legal moves for the side to move.
// Appends to 'moves' (caller must pass an empty or pre-sized vector).
void generate_moves(const Position& pos, std::vector<Move>& moves);

// Returns true if the side to move has at least one legal move.
// Slightly faster than generate_moves() for the terminal check.
bool has_legal_move(const Position& pos);

// Returns true if 'm' is a valid pseudo-legal/legal move on 'pos'.
bool is_pseudo_legal(const Position& pos, Move m);

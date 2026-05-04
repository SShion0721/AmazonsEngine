/*==================================================================
 * AMAZONS ENGINE — rays.h
 * Precomputed sliding-piece attack rays for the 10×10 board.
 * Mirrors Stockfish's precomputed bishop/rook attack tables.
 *==================================================================*/
#pragma once
#include "types.h"
#include <array>
#include <vector>

// RAYS[sq][d] = ordered list of squares reachable from sq in direction d,
// stopping at the board edge (not including sq itself).
// During move generation, we walk this list and stop at the first blocker.
extern std::array<std::array<std::vector<Square>, 8>, BOARD_SQ> RAYS;

// Must be called once at startup (equivalent to Stockfish's Bitboards::init())
void init_rays();

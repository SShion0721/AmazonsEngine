/*==================================================================
 * AMAZONS ENGINE — movegen.cpp
 * Bitboard-based legal move generation using PEXT attack tables.
 *
 * All ray-scanning loops have been replaced with O(1) bitboard
 * attack lookups via get_queen_attacks(). Move generation now
 * iterates over set bits in attack bitboards instead of walking
 * ray arrays.
 *==================================================================*/
#include "movegen.h"
#include <cassert>

void generate_moves(const Position& pos, std::vector<Move>& moves) {
    moves.clear();
    moves.reserve(2048);

    const Color  us = pos.side_to_move;
    Bitboard128  occ = pos.bb_occupied;

    for (int i = 0; i < NUM_AMAZONS; i++) {
        const Square from = pos.amazons[us][i];

        // Step 1: Remove the amazon from occupancy so it doesn't block itself
        //         and so the arrow can land on 'from'.
        occ.clear(from);

        // Step 2: Get all queen-reachable squares from 'from' (O(1) via PEXT)
        Bitboard128 queen_atk = get_queen_attacks(from, occ);
        // Can only move to empty squares (exclude all occupied)
        Bitboard128 to_set = queen_atk & ~occ;

        // Step 3: For each destination 'to', compute arrow destinations
        Bitboard128 to_iter = to_set;
        while (to_iter) {
            Square to = to_iter.pop_lsb();

            // Temporarily place amazon at 'to'
            occ.set(to);

            // Get arrow attacks from 'to' (O(1) via PEXT)
            Bitboard128 arrow_atk = get_queen_attacks(to, occ);
            Bitboard128 arrow_set = arrow_atk & ~occ;

            // Emit all (from, to, arrow) combinations
            Bitboard128 arrow_iter = arrow_set;
            while (arrow_iter) {
                Square arrow = arrow_iter.pop_lsb();
                moves.push_back(encode_move(from, to, arrow));
            }

            // Remove amazon from 'to' for next iteration
            occ.clear(to);
        }

        // Step 4: Restore amazon at 'from'
        occ.set(from);
    }
}

bool has_legal_move(const Position& pos) {
    const Color us = pos.side_to_move;
    Bitboard128 occ = pos.bb_occupied;

    for (int i = 0; i < NUM_AMAZONS; i++) {
        const Square from = pos.amazons[us][i];
        occ.clear(from);

        Bitboard128 queen_atk = get_queen_attacks(from, occ);
        Bitboard128 to_set = queen_atk & ~occ;

        // If there's at least one 'to' square, check for arrow
        Bitboard128 to_iter = to_set;
        while (to_iter) {
            Square to = to_iter.pop_lsb();
            occ.set(to);

            Bitboard128 arrow_atk = get_queen_attacks(to, occ);
            Bitboard128 arrow_set = arrow_atk & ~occ;

            if (arrow_set) {
                occ.clear(to);
                occ.set(from);
                return true;
            }

            occ.clear(to);
        }

        occ.set(from);
    }
    return false;
}

bool is_pseudo_legal(const Position& pos, Move m) {
    if (m == MOVE_NONE) return false;

    Square from  = move_from(m);
    Square to    = move_to(m);
    Square arrow = move_arrow(m);

    if (!valid_sq(from) || !valid_sq(to) || !valid_sq(arrow)) return false;

    // Ensure the piece at 'from' belongs to the current side
    int8_t us_piece = (pos.side_to_move == WHITE) ? WHITE_AMAZON : BLACK_AMAZON;
    if (pos.board[from] != us_piece) return false;

    // Check 'from' and 'to' are on the same queen line
    Bitboard128 occ = pos.bb_occupied;
    occ.clear(from);  // lift the amazon

    // 'to' must be reachable from 'from'
    Bitboard128 from_atk = get_queen_attacks(from, occ);
    if (!from_atk.test(to)) return false;
    if (occ.test(to)) return false;  // 'to' must be empty

    // Check arrow: place amazon at 'to', then check 'arrow' is reachable
    occ.set(to);
    Bitboard128 to_atk = get_queen_attacks(to, occ);
    if (!to_atk.test(arrow)) return false;
    if (occ.test(arrow)) return false;  // 'arrow' must be empty

    return true;
}

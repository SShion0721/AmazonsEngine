/*==================================================================
 * AMAZONS ENGINE — bitboard.cpp
 * Precomputed bitboard table initialization.
 * Includes PEXT attack tables, BETWEEN_BB, and LINE_BB.
 *==================================================================*/
#include "bitboard.h"
#include "types.h"
#include <cstring>
#include <vector>

Bitboard128 SQUARE_BB[100];
Bitboard128 NEIGHBOR_BB[100];
Bitboard128 FILE_A_BB;
Bitboard128 FILE_J_BB;
Bitboard128 BOARD_MASK_BB;

// ── PEXT Attack Tables ──────────────────────────────────────────
Bitboard128 LINE_MASK[100][4];
int LINE_MASK_LO_POPCNT[100][4];
Bitboard128 ATTACK_TABLE[100][4][512];

// ── BETWEEN_BB and LINE_BB ──────────────────────────────────────
Bitboard128 BETWEEN_BB[100][100];
Bitboard128 LINE_BB[100][100];

void init_bitboards() {
    // BOARD_MASK: all 100 valid squares
    BOARD_MASK_BB = Bitboard128::zero();
    for (int sq = 0; sq < 100; ++sq) {
        SQUARE_BB[sq] = Bitboard128::from_sq(sq);
        BOARD_MASK_BB |= SQUARE_BB[sq];
    }

    // File masks
    FILE_A_BB = Bitboard128::zero();
    FILE_J_BB = Bitboard128::zero();
    for (int r = 0; r < 10; ++r) {
        FILE_A_BB.set(r * 10);       // column 0
        FILE_J_BB.set(r * 10 + 9);   // column 9
    }

    // Neighbor (king attack) bitboards per square
    for (int sq = 0; sq < 100; ++sq) {
        NEIGHBOR_BB[sq] = Bitboard128::zero();
        int r = sq / 10, c = sq % 10;
        for (int d = 0; d < 8; ++d) {
            int nr = r + DR[d], nc = c + DC[d];
            if (nr >= 0 && nr < 10 && nc >= 0 && nc < 10)
                NEIGHBOR_BB[sq].set(nr * 10 + nc);
        }
    }
}

// ── Helper: get all squares on a line through sq in a direction ──
// Returns ordered list of squares (along the line, including sq)
static void get_line_squares(int sq, int dir, std::vector<int>& line) {
    line.clear();
    int r = sq / 10, c = sq % 10;

    // Direction vectors for 4 line types
    // DIR_RANK=0: horizontal (dr=0, dc=+1)
    // DIR_FILE=1: vertical   (dr=+1, dc=0)
    // DIR_DIAG=2: diagonal   (dr=+1, dc=+1)
    // DIR_ADIAG=3: anti-diag (dr=+1, dc=-1)
    static const int dr4[4] = {0, 1, 1,  1};
    static const int dc4[4] = {1, 0, 1, -1};

    int dr = dr4[dir], dc = dc4[dir];

    // Go to the start of the line (negative direction)
    int sr = r, sc = c;
    while (sr - dr >= 0 && sr - dr < 10 && sc - dc >= 0 && sc - dc < 10) {
        sr -= dr;
        sc -= dc;
    }

    // Walk the entire line
    int cr = sr, cc = sc;
    while (cr >= 0 && cr < 10 && cc >= 0 && cc < 10) {
        line.push_back(cr * 10 + cc);
        cr += dr;
        cc += dc;
    }
}

// ── Helper: compute attacks from sq on a given line with occupancy ──
static Bitboard128 compute_line_attacks(int sq, const std::vector<int>& line, unsigned occ_bits) {
    Bitboard128 result = Bitboard128::zero();

    // Find sq's position in the line
    int sq_pos = -1;
    for (int i = 0; i < (int)line.size(); ++i) {
        if (line[i] == sq) { sq_pos = i; break; }
    }
    if (sq_pos < 0) return result;

    // Build occupancy array for the line (excluding sq itself)
    // occ_bits is indexed by position in the line (minus sq)
    // We need to map occ_bits back to which line positions are occupied
    int bit_idx = 0;
    bool occupied[10] = {};
    for (int i = 0; i < (int)line.size(); ++i) {
        if (line[i] == sq) continue;  // sq is excluded from mask
        occupied[i] = (occ_bits >> bit_idx) & 1;
        bit_idx++;
    }

    // Slide in positive direction (higher indices in line)
    for (int i = sq_pos + 1; i < (int)line.size(); ++i) {
        result.set(line[i]);
        if (occupied[i]) break;  // blocked
    }

    // Slide in negative direction (lower indices in line)
    for (int i = sq_pos - 1; i >= 0; --i) {
        result.set(line[i]);
        if (occupied[i]) break;  // blocked
    }

    return result;
}

void init_attack_tables() {
    std::vector<int> line;

    for (int sq = 0; sq < 100; ++sq) {
        for (int dir = 0; dir < 4; ++dir) {
            get_line_squares(sq, dir, line);

            // Build line mask (all squares on line EXCEPT sq)
            LINE_MASK[sq][dir] = Bitboard128::zero();
            for (int s : line) {
                if (s != sq)
                    LINE_MASK[sq][dir].set(s);
            }

            // Precompute lo popcount for PEXT split
            LINE_MASK_LO_POPCNT[sq][dir] = 0;
            {
                uint64_t m = LINE_MASK[sq][dir].lo;
                while (m) { LINE_MASK_LO_POPCNT[sq][dir]++; m &= m - 1; }
            }

            // Number of mask bits (line_length - 1)
            int mask_bits = LINE_MASK[sq][dir].popcount();

            // For each possible occupancy pattern, compute the attack set
            unsigned max_idx = 1u << mask_bits;
            for (unsigned idx = 0; idx < max_idx && idx < 512; ++idx) {
                ATTACK_TABLE[sq][dir][idx] = compute_line_attacks(sq, line, idx);
            }
        }
    }
}

void init_between_line_bb() {
    // Initialize to zero
    for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 100; ++j) {
            BETWEEN_BB[i][j] = Bitboard128::zero();
            LINE_BB[i][j] = Bitboard128::zero();
        }

    for (int sq1 = 0; sq1 < 100; ++sq1) {
        int r1 = sq1 / 10, c1 = sq1 % 10;

        for (int sq2 = sq1 + 1; sq2 < 100; ++sq2) {
            int r2 = sq2 / 10, c2 = sq2 % 10;
            int dr = r2 - r1, dc = c2 - c1;

            // Check if on same rank, file, or diagonal
            bool same_line = false;
            int step_r = 0, step_c = 0;

            if (dr == 0 && dc != 0) {
                // Same rank
                same_line = true;
                step_c = (dc > 0) ? 1 : -1;
            } else if (dc == 0 && dr != 0) {
                // Same file
                same_line = true;
                step_r = (dr > 0) ? 1 : -1;
            } else if (dr != 0 && dc != 0 && abs(dr) == abs(dc)) {
                // Same diagonal
                same_line = true;
                step_r = (dr > 0) ? 1 : -1;
                step_c = (dc > 0) ? 1 : -1;
            }

            if (!same_line) continue;

            // BETWEEN_BB: squares strictly between sq1 and sq2
            Bitboard128 between = Bitboard128::zero();
            int cr = r1 + step_r, cc = c1 + step_c;
            while (cr * 10 + cc != sq2) {
                between.set(cr * 10 + cc);
                cr += step_r;
                cc += step_c;
            }
            BETWEEN_BB[sq1][sq2] = between;
            BETWEEN_BB[sq2][sq1] = between;

            // LINE_BB: entire line through sq1 and sq2
            Bitboard128 full_line = SQUARE_BB[sq1] | between | SQUARE_BB[sq2];
            // Extend in both directions to board edges
            int er = r1 - step_r, ec = c1 - step_c;
            while (er >= 0 && er < 10 && ec >= 0 && ec < 10) {
                full_line.set(er * 10 + ec);
                er -= step_r;
                ec -= step_c;
            }
            er = r2 + step_r; ec = c2 + step_c;
            while (er >= 0 && er < 10 && ec >= 0 && ec < 10) {
                full_line.set(er * 10 + ec);
                er += step_r;
                ec += step_c;
            }
            LINE_BB[sq1][sq2] = full_line;
            LINE_BB[sq2][sq1] = full_line;
        }
    }
}

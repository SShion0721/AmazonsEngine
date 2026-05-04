#include "line_pattern.h"

#include <array>

LineRef SQ_LINES[BOARD_SQ][4];
uint32_t LINE_SEED[NUM_LINES];

namespace {

uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void set_ref(Square sq, int slot, int line, int pos) {
    SQ_LINES[sq][slot] = {
        static_cast<uint8_t>(line),
        static_cast<uint8_t>(2 * pos)
    };
}

} // namespace

void init_line_patterns() {
    uint64_t seed = LINE_SEED_BASE;
    for (int i = 0; i < NUM_LINES; ++i)
        LINE_SEED[i] = static_cast<uint32_t>(splitmix64(seed));

    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        const int r = sq_row(sq);
        const int c = sq_col(sq);

        set_ref(sq, 0, r, c);          // ranks 0..9
        set_ref(sq, 1, 10 + c, r);     // files 10..19

        const int diag = r - c + 9;    // diagonals 20..38
        int dr = r;
        int dc = c;
        while (dr > 0 && dc > 0) {
            --dr;
            --dc;
        }
        set_ref(sq, 2, 20 + diag, r - dr);

        const int anti = r + c;        // anti-diagonals 39..57
        dr = r;
        dc = c;
        while (dr > 0 && dc < 9) {
            --dr;
            ++dc;
        }
        set_ref(sq, 3, 39 + anti, r - dr);
    }
}

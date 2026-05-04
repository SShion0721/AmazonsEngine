/*==================================================================
 * AMAZONS ENGINE — rays.cpp
 *==================================================================*/
#include "rays.h"

std::array<std::array<std::vector<Square>, 8>, BOARD_SQ> RAYS;

void init_rays() {
    for (int sq = 0; sq < BOARD_SQ; sq++) {
        for (int d = 0; d < 8; d++) {
            RAYS[sq][d].clear();
            int r = sq_row(sq), c = sq_col(sq);
            int nr = r + DR[d], nc = c + DC[d];
            while (valid_rc(nr, nc)) {
                RAYS[sq][d].push_back(make_sq(nr, nc));
                nr += DR[d];
                nc += DC[d];
            }
        }
    }
}

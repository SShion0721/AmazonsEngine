/*==================================================================
 * AMAZONS ENGINE — movepicker.cpp
 *==================================================================*/
#include "movepicker.h"
#include "search.h"
#include "bitboard.h"
#include <cstdlib>

bool g_use_bounded_movegen = true;
int g_bounded_dest_cap = 16;
int g_bounded_arrow_cap = 12;

namespace {

struct ScoredSquare {
    Square sq;
    int score;
};

void insert_top_square(ScoredSquare* out, int& count, int cap, ScoredSquare value) {
    if (cap <= 0)
        return;
    if (count == cap && value.score <= out[count - 1].score)
        return;

    int pos = std::min(count, cap - 1);
    if (count < cap)
        ++count;

    while (pos > 0 && out[pos - 1].score < value.score) {
        out[pos] = out[pos - 1];
        --pos;
    }
    out[pos] = value;
}

int queen_mobility_from(Square sq, Bitboard128 occ) {
    return (get_queen_attacks(sq, occ) & ~occ).popcount();
}

int blocks_opponent_with_square(const Position& pos, Square blocker, Bitboard128 occ_without_blocker) {
    const Color them = flip(pos.side_to_move);
    int score = 0;

    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square enemy = pos.amazons[them][i];
        if (!get_queen_attacks(enemy, occ_without_blocker).test(blocker))
            continue;

        Bitboard128 occ_with_blocker = occ_without_blocker;
        occ_with_blocker.set(blocker);
        const int before = queen_mobility_from(enemy, occ_without_blocker);
        const int after = queen_mobility_from(enemy, occ_with_blocker);
        score += 600 + 120 * std::max(0, before - after);
    }

    return score;
}

bool adjacent_to_side(const Position& pos, Square sq, Color c) {
    const Bitboard128 side = c == WHITE ? pos.bb_white : pos.bb_black;
    return bool(NEIGHBOR_BB[sq] & side);
}

int cheap_category_score(const Position& pos, Move m) {
    const Color them = flip(pos.side_to_move);
    const Bitboard128 them_bb = them == WHITE ? pos.bb_white : pos.bb_black;
    const Square arrow = move_arrow(m);
    return (NEIGHBOR_BB[arrow] & them_bb) ? 1200 : 0;
}

int center_score(Square sq) {
    const int dr = std::abs(2 * sq_row(sq) - 9);
    const int dc = std::abs(2 * sq_col(sq) - 9);
    return 72 - 4 * (dr + dc);
}

int cheap_destination_score(const Position& pos,
                            Color us,
                            Square from,
                            Square to,
                            Bitboard128 occ_without_from,
                            const int (&history)[2][BOARD_SQ][BOARD_SQ]) {
    Bitboard128 occ_after = occ_without_from;
    occ_after.set(to);
    const Color them = flip(us);
    const Bitboard128 them_bb = them == WHITE ? pos.bb_white : pos.bb_black;

    int score = history[us][from][to];
    score += center_score(to);
    score += 18 * queen_mobility_from(to, occ_after);
    if (NEIGHBOR_BB[to] & them_bb)
        score += 700;
    return score;
}

int cheap_arrow_score(const Position& pos,
                      Color us,
                      Square from,
                      Square to,
                      Square arrow,
                      const int (&arrow_history)[2][BOARD_SQ][BOARD_SQ],
                      const int (&from_arrow_history)[2][BOARD_SQ][BOARD_SQ]) {
    const Color them = flip(us);
    const Bitboard128 them_bb = them == WHITE ? pos.bb_white : pos.bb_black;

    int score = arrow_history[us][to][arrow] + from_arrow_history[us][from][arrow];
    score += center_score(arrow) / 2;
    if (NEIGHBOR_BB[arrow] & them_bb)
        score += 1200;
    if (NEIGHBOR_BB[arrow] & (us == WHITE ? pos.bb_white : pos.bb_black))
        score -= 150;
    return score;
}

int generate_bounded_moves(const Position& pos,
                           ScoredMove* out,
                           int max_moves,
                           const int (&history)[2][BOARD_SQ][BOARD_SQ],
                           const int (&arrow_history)[2][BOARD_SQ][BOARD_SQ],
                           const int (&from_arrow_history)[2][BOARD_SQ][BOARD_SQ]) {
    const Color us = pos.side_to_move;
    Bitboard128 occ = pos.bb_occupied;
    int count = 0;

    const int dest_cap = std::clamp(g_bounded_dest_cap, 1, 64);
    const int arrow_cap = std::clamp(g_bounded_arrow_cap, 1, 64);

    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square from = pos.amazons[us][i];
        occ.clear(from);

        ScoredSquare dests[64];
        int dest_count = 0;
        Bitboard128 tos = get_queen_attacks(from, occ) & ~occ;
        while (tos) {
            const Square to = tos.pop_lsb();
            Bitboard128 occ_after = occ;
            occ_after.set(to);
            if (!(get_queen_attacks(to, occ_after) & ~occ_after))
                continue;
            const int score = cheap_destination_score(pos, us, from, to, occ, history);
            insert_top_square(dests, dest_count, dest_cap, {to, score});
        }

        for (int d = 0; d < dest_count; ++d) {
            const Square to = dests[d].sq;
            occ.set(to);

            ScoredSquare arrows[64];
            int arrow_count = 0;
            Bitboard128 arrow_set = get_queen_attacks(to, occ) & ~occ;
            while (arrow_set) {
                const Square arrow = arrow_set.pop_lsb();
                const int score = dests[d].score
                                + cheap_arrow_score(pos, us, from, to, arrow,
                                                    arrow_history, from_arrow_history);
                insert_top_square(arrows, arrow_count, arrow_cap, {arrow, score});
            }

            for (int a = 0; a < arrow_count && count < max_moves; ++a) {
                const Move m = encode_move(from, to, arrows[a].sq);
                int score = arrows[a].score;
                if (g_use_move_categories)
                    score += cheap_category_score(pos, m);
                out[count++] = {m, score, 0};
            }

            occ.clear(to);
            if (count >= max_moves)
                break;
        }

        occ.set(from);
        if (count >= max_moves)
            break;
    }

    return count;
}

} // namespace

int amazon_move_category_score(const Position& pos, Move m, const Stack* ss) {
    if (m == MOVE_NONE)
        return 0;

    const Color us = pos.side_to_move;
    const Color them = flip(us);
    const Square from = move_from(m);
    const Square to = move_to(m);
    const Square arrow = move_arrow(m);

    Bitboard128 occ_after_move = pos.bb_occupied;
    occ_after_move.clear(from);
    occ_after_move.set(to);

    int score = 0;

    score += blocks_opponent_with_square(pos, to, occ_after_move & ~SQUARE_BB[to]);
    score += blocks_opponent_with_square(pos, arrow, occ_after_move);

    if (adjacent_to_side(pos, arrow, them))
        score += 1400;
    if (adjacent_to_side(pos, to, them))
        score += 700;

    const Move prev = ss ? (ss - 1)->current_move : MOVE_NONE;
    if (prev != MOVE_NONE) {
        const Square last_to = move_to(prev);
        if (get_queen_attacks(last_to, occ_after_move).test(arrow))
            score += 900;
        if (NEIGHBOR_BB[arrow].test(last_to))
            score += 500;
    }

    const int from_mob = queen_mobility_from(from, pos.bb_occupied);
    const int to_mob_before_arrow = queen_mobility_from(to, occ_after_move);

    if (from_mob <= 3 && to_mob_before_arrow > from_mob + 2)
        score += 900 + 120 * (to_mob_before_arrow - from_mob);

    Bitboard128 occ_after_arrow = occ_after_move;
    occ_after_arrow.set(arrow);
    const int to_mob_after_arrow = queen_mobility_from(to, occ_after_arrow);

    if (to_mob_after_arrow == 0)
        score -= 5000;
    else if (to_mob_after_arrow <= 2)
        score -= 1800;
    else if (to_mob_after_arrow <= 4)
        score -= 600;

    const Bitboard128 own = us == WHITE ? pos.bb_white : pos.bb_black;
    if (adjacent_to_side(pos, arrow, us) && !(NEIGHBOR_BB[arrow] & (own & ~SQUARE_BB[from] | SQUARE_BB[to])))
        score -= 300;

    return score;
}

Move MovePicker::next_move() {
    switch (stage_) {
    case STAGE_TT:
        stage_ = STAGE_KILLER_1;
        if (tt_move_ != MOVE_NONE && is_pseudo_legal(pos_, tt_move_)) {
            remember_special(tt_move_);
            return tt_move_;
        }
        [[fallthrough]];

    case STAGE_KILLER_1:
        stage_ = STAGE_KILLER_2;
        if (ss_->killers[0] != MOVE_NONE &&
            !is_excluded(ss_->killers[0]) &&
            is_pseudo_legal(pos_, ss_->killers[0])) {
            remember_special(ss_->killers[0]);
            return ss_->killers[0];
        }
        [[fallthrough]];

    case STAGE_KILLER_2:
        stage_ = STAGE_COUNTER;
        if (ss_->killers[1] != MOVE_NONE &&
            !is_excluded(ss_->killers[1]) &&
            is_pseudo_legal(pos_, ss_->killers[1])) {
            remember_special(ss_->killers[1]);
            return ss_->killers[1];
        }
        [[fallthrough]];

    case STAGE_COUNTER:
        stage_ = STAGE_GENERATE;
        if (counter_ != MOVE_NONE &&
            !is_excluded(counter_) &&
            is_pseudo_legal(pos_, counter_)) {
            remember_special(counter_);
            return counter_;
        }
        [[fallthrough]];

    case STAGE_GENERATE: {
        // Only now do we generate all moves (lazy!)
        const Color us = pos_.side_to_move;
        move_count_ = 0;
        const bool bounded = g_use_bounded_movegen
                          && g_use_lmp_pruning
                          && !g_teacher_safe_search
                          && top_k_ < 1000000;

        if (bounded) {
            const int max_candidates = std::min(MAX_LEGAL_MOVES, std::max(top_k_ * 3, top_k_ + 64));
            move_count_ = generate_bounded_moves(pos_, buffer_.moves, max_candidates,
                                                 history_, arrow_history_, from_arrow_history_);
        } else {
            const int raw_count = generate_moves(pos_, buffer_.raw, MAX_LEGAL_MOVES);
            for (int i = 0; i < raw_count; ++i) {
                const Move m = buffer_.raw[i];
                int s = history_[us][move_from(m)][move_to(m)]
                      + arrow_history_[us][move_to(m)][move_arrow(m)]
                      + from_arrow_history_[us][move_from(m)][move_arrow(m)];
                if (g_use_move_categories)
                    s += cheap_category_score(pos_, m);
                buffer_.moves[move_count_++] = {m, s, 0};
            }
        }

        const auto better = [](const ScoredMove& a, const ScoredMove& b) {
            return a.score > b.score;
        };
        if (top_k_ > 0 && move_count_ > top_k_) {
            std::nth_element(buffer_.moves, buffer_.moves + top_k_, buffer_.moves + move_count_, better);
            move_count_ = top_k_;
        }

        if (g_use_move_categories) {
            for (int i = 0; i < move_count_; ++i) {
                const int category = amazon_move_category_score(pos_, buffer_.moves[i].move, ss_);
                buffer_.moves[i].category = category;
                buffer_.moves[i].score += category;
            }
        }

        std::sort(buffer_.moves, buffer_.moves + move_count_, better);

        move_idx_ = 0;
        stage_ = STAGE_YIELD;
        [[fallthrough]];
    }

    case STAGE_YIELD:
        while (move_idx_ < move_count_) {
            const ScoredMove& sm = buffer_.moves[move_idx_++];
            Move m = sm.move;
            if (is_excluded(m)) continue;
            last_category_score_ = sm.category;
            last_category_known_ = true;
            remember_tried(m);
            return m;
        }
        stage_ = STAGE_DONE;
        return MOVE_NONE;

    case STAGE_DONE:
        return MOVE_NONE;
    }
    return MOVE_NONE;
}

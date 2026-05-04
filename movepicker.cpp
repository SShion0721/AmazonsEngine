/*==================================================================
 * AMAZONS ENGINE — movepicker.cpp
 *==================================================================*/
#include "movepicker.h"
#include "search.h"
#include "bitboard.h"

namespace {

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
            tried_.push_back(tt_move_);
            return tt_move_;
        }
        [[fallthrough]];

    case STAGE_KILLER_1:
        stage_ = STAGE_KILLER_2;
        if (ss_->killers[0] != MOVE_NONE &&
            !already_tried(ss_->killers[0]) &&
            is_pseudo_legal(pos_, ss_->killers[0])) {
            tried_.push_back(ss_->killers[0]);
            return ss_->killers[0];
        }
        [[fallthrough]];

    case STAGE_KILLER_2:
        stage_ = STAGE_COUNTER;
        if (ss_->killers[1] != MOVE_NONE &&
            !already_tried(ss_->killers[1]) &&
            is_pseudo_legal(pos_, ss_->killers[1])) {
            tried_.push_back(ss_->killers[1]);
            return ss_->killers[1];
        }
        [[fallthrough]];

    case STAGE_COUNTER:
        stage_ = STAGE_GENERATE;
        if (counter_ != MOVE_NONE &&
            !already_tried(counter_) &&
            is_pseudo_legal(pos_, counter_)) {
            tried_.push_back(counter_);
            return counter_;
        }
        [[fallthrough]];

    case STAGE_GENERATE: {
        // Only now do we generate all moves (lazy!)
        std::vector<Move> raw;
        generate_moves(pos_, raw);
        const Color us = pos_.side_to_move;
        moves_.reserve(raw.size());
        for (Move m : raw) {
            int s = history_[us][move_from(m)][move_to(m)]
                  + arrow_history_[us][move_to(m)][move_arrow(m)]
                  + amazon_move_category_score(pos_, m, ss_);
            moves_.push_back({m, s});
        }
        std::sort(moves_.begin(), moves_.end(),
                  [](const ScoredMove& a, const ScoredMove& b) {
                      return a.score > b.score;
                  });
        move_idx_ = 0;
        stage_ = STAGE_YIELD;
        [[fallthrough]];
    }

    case STAGE_YIELD:
        while (move_idx_ < moves_.size()) {
            Move m = moves_[move_idx_++].move;
            if (already_tried(m)) continue;
            tried_.push_back(m);
            return m;
        }
        stage_ = STAGE_DONE;
        return MOVE_NONE;

    case STAGE_DONE:
        return MOVE_NONE;
    }
    return MOVE_NONE;
}

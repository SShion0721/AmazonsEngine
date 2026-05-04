/*==================================================================
 * AMAZONS ENGINE - search.cpp
 * Full Negamax PVS implementation with Stockfish-inspired pruning,
 * ordering, and TT handling adapted for Amazons.
 *==================================================================*/
#include "search.h"
#include "movegen.h"
#include "evaluate.h"
#include "movepicker.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

bool g_use_null_move_pruning = false;
bool g_use_lmp_pruning = true;
bool g_use_move_categories = true;
bool g_teacher_safe_search = false;

namespace {

constexpr int MAX_SEARCH_PLY = 120;

inline Score mate_in(int ply) {
    return SCORE_MATE - ply;
}

inline Score mated_in(int ply) {
    return -SCORE_MATE + ply;
}

inline bool is_decisive_score(Score score) {
    return std::abs(score) >= SCORE_MATE - MAX_SEARCH_PLY;
}

inline Score score_to_tt(Score score, int ply) {
    if (score >= SCORE_MATE - MAX_SEARCH_PLY)
        return score + ply;
    if (score <= -SCORE_MATE + MAX_SEARCH_PLY)
        return score - ply;
    return score;
}

inline Score score_from_tt(Score score, int ply) {
    if (score >= SCORE_MATE - MAX_SEARCH_PLY)
        return score - ply;
    if (score <= -SCORE_MATE + MAX_SEARCH_PLY)
        return score + ply;
    return score;
}

inline void add_history_score(int& entry, int delta) {
    entry = std::clamp(entry + delta, -32768, 32767);
}

int candidate_cap(int phase, int depth, bool is_root, bool is_pv) {
    if (is_root || is_pv || phase >= 56)
        return 1000000;

    if (phase < 14) {
        if (depth <= 2) return 80;
        if (depth <= 4) return 160;
        return 256;
    }

    if (phase < 40) {
        if (depth <= 2) return 120;
        if (depth <= 4) return 220;
        return 360;
    }

    if (depth <= 2) return 180;
    if (depth <= 4) return 320;
    return 520;
}

} // namespace

void Searcher::init_tables() {
    for (int d = 1; d < 64; ++d) {
        for (int m = 1; m < 64; ++m) {
            lmr_[d][m] = static_cast<int>(
                std::max(0.0, 1.0 + std::log(d) * std::log(m) / 2.0));
        }
    }
}

void Searcher::check_time() {
    if (time_man.hard_stop())
        stop_.store(true, std::memory_order_relaxed);
}

void Searcher::update_histories(Move m, int depth, Stack* ss, Color us) {
    if (m != ss->killers[0]) {
        ss->killers[1] = ss->killers[0];
        ss->killers[0] = m;
    }

    const int bonus = depth * depth * 64;
    add_history_score(history_[us][move_from(m)][move_to(m)], bonus);
    add_history_score(arrow_history_[us][move_to(m)][move_arrow(m)], bonus);

    const Move prev = (ss - 1)->current_move;
    if (prev != MOVE_NONE)
        countermoves_[us][move_from(prev)][move_to(prev)] = m;
}

Score Searcher::negamax(Position& pos, Score alpha, Score beta, int depth, Stack* ss, Move* best_move_out) {
    if ((++nodes & 0x1FFF) == 0)
        check_time();
    if (stop_.load(std::memory_order_relaxed))
        return SCORE_ZERO;

    const bool is_pv = (beta - alpha > 1);
    const bool is_root = (ss->ply == 0);

    if (ss->ply >= MAX_SEARCH_PLY)
        return evaluate(pos);

    tt_.prefetch(pos.key);

    if (!has_legal_move(pos))
        return mated_in(ss->ply);

    if (depth <= 0)
        return evaluate(pos);

    auto [hit, tte] = tt_.probe(pos.key);
    Move tt_move = hit ? tte->best_move() : MOVE_NONE;
    if (tt_move != MOVE_NONE && !is_pseudo_legal(pos, tt_move)) {
        hit = false;
        tt_move = MOVE_NONE;
    }

    const Score tt_score = hit ? score_from_tt(tte->score(), ss->ply) : SCORE_NONE;
    Score tt_eval = SCORE_NONE;

    if (hit) {
        tt_eval = tte->eval();

        if (!is_pv && tte->depth() >= depth) {
            const TTBound bound = tte->bound();
            if (bound == TT_EXACT)
                return tt_score;
            if (bound == TT_LOWER && tt_score >= beta)
                return tt_score;
            if (bound == TT_UPPER && tt_score <= alpha)
                return tt_score;
        }
    }

    ss->static_eval = (hit && tt_eval != SCORE_NONE) ? tt_eval : evaluate(pos);

    const bool improving = (ss - 2)->static_eval != SCORE_NONE
                        && ss->static_eval > (ss - 2)->static_eval;

    if (!is_root && !is_pv && !g_teacher_safe_search
        && depth <= 4 && ss->static_eval - 60 * depth + 30 * int(improving) >= beta)
        return ss->static_eval;

    if (!g_teacher_safe_search && depth >= 4 && tt_move == MOVE_NONE)
        --depth;

    if (g_use_null_move_pruning
        && !is_root
        && !is_pv
        && !g_teacher_safe_search
        && depth >= 3
        && ss->static_eval >= beta
        && !ss->in_null_move) {
        const int reduction = 3 + depth / 6 + !improving;
        pos.do_null_move();
        (ss + 1)->in_null_move = true;
        (ss + 1)->ply = ss->ply + 1;
        const Score null_score = -negamax(pos, -beta, -beta + 1, depth - reduction, ss + 1);
        (ss + 1)->in_null_move = false;
        pos.undo_null_move();

        if (stop_.load(std::memory_order_relaxed))
            return SCORE_ZERO;

        if (null_score >= beta)
            return is_decisive_score(null_score) ? beta : null_score;
    }

    Score best_score = -SCORE_INF;
    Move best_move = MOVE_NONE;
    const Score orig_alpha = alpha;
    int move_count = 0;

    const bool futility_ok = !is_root
                          && !is_pv
                          && !g_teacher_safe_search
                          && depth <= 6
                          && ss->static_eval + 80 * depth + 40 * int(improving) < alpha;

    const int lmp_threshold = (!g_use_lmp_pruning || g_teacher_safe_search)
                            ? 1000000
                            : candidate_cap(pos.bb_arrow.popcount(), depth, is_root, is_pv);

    Move countermove = MOVE_NONE;
    const Move prev = (ss - 1)->current_move;
    if (prev != MOVE_NONE)
        countermove = countermoves_[pos.side_to_move][move_from(prev)][move_to(prev)];

    MovePicker picker(pos, tt_move, ss, history_, arrow_history_, countermove);
    Move m;
    while ((m = picker.next_move()) != MOVE_NONE) {
        ++move_count;
        const int category_score = g_use_move_categories
                                 ? amazon_move_category_score(pos, m, ss)
                                 : 0;

        if (futility_ok && category_score < 2500
            && move_count > 1 && best_score > mated_in(MAX_SEARCH_PLY))
            continue;

        if (!is_root && !is_pv
            && move_count > lmp_threshold
            && best_score > mated_in(MAX_SEARCH_PLY))
            break;

        const Color us = pos.side_to_move;
        pos.do_move(m);
        ss->move_count = move_count;
        (ss + 1)->current_move = m;
        (ss + 1)->ply = ss->ply + 1;

        Score score;

        if (move_count == 1) {
            score = -negamax(pos, -beta, -alpha, depth - 1, ss + 1);
        } else {
            int reduction = 0;
            if (depth >= 3 && move_count >= 4) {
                reduction = lmr_[std::min(depth - 1, 63)][std::min(move_count, 63)];

                if (m == ss->killers[0] || m == ss->killers[1])
                    reduction = std::max(0, reduction - 1);

                const int h_score = history_[us][move_from(m)][move_to(m)]
                                  + arrow_history_[us][move_to(m)][move_arrow(m)];
                if (h_score > 4000 || category_score > 3000)
                    reduction = std::max(0, reduction - 1);
                else if (h_score < -4000 || category_score < -1500)
                    reduction += 1;

                if (is_pv)
                    reduction = std::max(0, reduction - 1);

                reduction = std::min(reduction, depth - 2);
            }

            score = -negamax(pos, -alpha - 1, -alpha, depth - 1 - reduction, ss + 1);

            if (score > alpha && reduction > 0)
                score = -negamax(pos, -alpha - 1, -alpha, depth - 1, ss + 1);

            if (score > alpha && score < beta)
                score = -negamax(pos, -beta, -alpha, depth - 1, ss + 1);
        }

        pos.undo_move(m);

        if (stop_.load(std::memory_order_relaxed))
            return SCORE_ZERO;

        if (best_move == MOVE_NONE || score > best_score) {
            best_score = score;
            best_move = m;
            if (best_move_out)
                *best_move_out = m;

            if (score > alpha) {
                alpha = score;

                if (alpha >= beta) {
                    update_histories(m, depth, ss, pos.side_to_move);

                    const int malus = -(depth * depth);
                    for (Move tried : picker.tried_moves()) {
                        if (tried == m)
                            break;
                        add_history_score(history_[pos.side_to_move][move_from(tried)][move_to(tried)],
                                          malus * 64);
                        add_history_score(arrow_history_[pos.side_to_move][move_to(tried)][move_arrow(tried)],
                                          malus * 64);
                    }
                    break;
                }
            }
        }
    }

    TTBound bound = TT_EXACT;
    if (best_score <= orig_alpha)
        bound = TT_UPPER;
    else if (best_score >= beta)
        bound = TT_LOWER;

    tte->save(pos.key, score_to_tt(best_score, ss->ply), is_pv, bound, depth,
              best_move, ss->static_eval, tt_.generation());

    return best_score;
}

Move Searcher::search(Position& pos, int max_depth, Score* out_score, int thread_id) {
    stop_.store(false, std::memory_order_relaxed);
    nodes = 0;
    killers_ = {};
    std::fill(&history_[0][0][0], &history_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::fill(&arrow_history_[0][0][0], &arrow_history_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::memset(countermoves_, 0, sizeof(countermoves_));

    if (thread_id == 0)
        tt_.new_search();

    std::vector<Move> root_moves;
    generate_moves(pos, root_moves);
    Move best_move = root_moves.empty() ? MOVE_NONE : root_moves.front();
    Score best_score = SCORE_ZERO;

    std::array<Stack, 128> stack{};
    Stack* ss = stack.data() + 4;
    ss->ply = 0;

    max_depth = std::min(max_depth, MAX_SEARCH_PLY);

    for (int depth = 1; depth <= max_depth; ++depth) {
        if (thread_id > 0 && depth > 3 && (depth + thread_id) % 2 == 0)
            continue;

        Score alpha;
        Score beta;

        if (depth >= 5) {
            const int delta = 30;
            alpha = std::max(-SCORE_INF, best_score - delta);
            beta = std::min(SCORE_INF, best_score + delta);
        } else {
            alpha = -SCORE_INF;
            beta = SCORE_INF;
        }

        Score score = SCORE_NONE;
        while (true) {
            Move current_root_move = MOVE_NONE;
            score = negamax(pos, alpha, beta, depth, ss, &current_root_move);

            if (current_root_move != MOVE_NONE)
                best_move = current_root_move;

            if (stop_.load(std::memory_order_relaxed))
                goto done;

            if (score <= alpha)
                alpha = std::max(-SCORE_INF, alpha - 50);
            else if (score >= beta)
                beta = std::min(SCORE_INF, beta + 50);
            else
                break;
        }

        best_score = score;

        if (!silent) {
            std::cout << "info depth " << depth
                      << " score cp " << best_score
                      << " nodes " << nodes
                      << " time " << time_man.elapsed_ms()
                      << " hashfull " << tt_.hashfull()
                      << " pv " << move_to_str(best_move)
                      << std::endl;
        }

        if (time_man.soft_stop())
            break;

        if (is_decisive_score(best_score))
            break;
    }

done:
    if (out_score)
        *out_score = best_score;
    if (!silent)
        std::cout << "bestmove " << move_to_str(best_move) << std::endl;
    return best_move;
}

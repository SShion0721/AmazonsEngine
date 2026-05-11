/*==================================================================
 * AMAZONS ENGINE - search.cpp
 * Full Negamax PVS implementation with Stockfish-inspired pruning,
 * ordering, and TT handling adapted for Amazons.
 *==================================================================*/
#include "search.h"
#include "movegen.h"
#include "evaluate.h"
#include "movepicker.h"
#include "mcts.h"
#include "policy_prior.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

bool g_use_null_move_pruning = false;
bool g_use_lmp_pruning = true;
bool g_use_move_categories = true;
bool g_teacher_safe_search = false;
bool g_use_soft_tail_search = true;
bool g_use_mcts_root = true;
int g_mcts_time_share = 75;
int g_ab_verify_topn = 6;
int g_mcts_min_time_ms = 80;
int g_mcts_cpuct = 120;
int g_mcts_threads = 1;
bool g_use_tree_reuse = true;
int g_search_mode = SEARCH_MODE_MATCH;

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

constexpr int HISTORY_MAX = 16384;

inline void update_history_score(int& entry, int delta) {
    delta = std::clamp(delta, -HISTORY_MAX, HISTORY_MAX);
    entry += delta - entry * std::abs(delta) / HISTORY_MAX;
    entry = std::clamp(entry, -HISTORY_MAX, HISTORY_MAX);
}

int candidate_cap(int phase, int depth, bool is_root, bool is_pv) {
    if (g_teacher_safe_search || g_search_mode == SEARCH_MODE_TEACHER_SAFE)
        return 1000000;
    if (g_search_mode != SEARCH_MODE_FAST_SELFPLAY && (is_root || is_pv))
        return 1000000;
    if (phase >= 56)
        return 1000000;

    const int pv_bonus = (is_root || is_pv) ? 96 : 0;

    if (phase < 14) {
        if (depth <= 2) return 80 + pv_bonus;
        if (depth <= 4) return 160 + pv_bonus;
        return 256 + pv_bonus;
    }

    if (phase < 40) {
        if (depth <= 2) return 120 + pv_bonus;
        if (depth <= 4) return 220 + pv_bonus;
        return 360 + pv_bonus;
    }

    if (depth <= 2) return 180 + pv_bonus;
    if (depth <= 4) return 320 + pv_bonus;
    return 520 + pv_bonus;
}

bool should_use_mcts_root(const Position& pos, int max_depth, int thread_id, const TimeManager& tm, uint64_t node_limit) {
    if (thread_id != 0)
        return false;
    if (!g_use_mcts_root || g_search_mode != SEARCH_MODE_MATCH || g_teacher_safe_search)
        return false;
    if (node_limit != 0)
        return false;
    if (max_depth < 8)
        return false;
    if (pos.bb_arrow.popcount() >= 56)
        return false;
    if (tm.soft_limit_ms() > 30000)
        return false;

    const int remaining = tm.soft_limit_ms() - tm.elapsed_ms();
    return remaining >= g_mcts_min_time_ms;
}

struct AdaptiveMctsParams {
    MctsConfig config;
    int time_share = 75;
    int verify_topn = 6;
};

AdaptiveMctsParams adaptive_mcts_params(const Position& pos, int remaining_ms) {
    AdaptiveMctsParams params;
    const int arrows = pos.bb_arrow.popcount();
    params.time_share = std::clamp(g_mcts_time_share, 1, 100);
    params.verify_topn = std::max(0, g_ab_verify_topn);
    params.config.cpuct = std::clamp(g_mcts_cpuct, 1, 1000) / 100.0;
    params.config.threads = std::max(1, g_mcts_threads);
    params.config.use_tree_reuse = g_use_tree_reuse;
    params.config.use_policy_prior = g_use_policy_prior && policy_prior_loaded();
    params.config.policy_blend = std::clamp(g_policy_prior_blend, 0, 100);

    if (arrows < 20) {
        params.time_share = std::max(params.time_share, 85);
        if (params.verify_topn > 0)
            params.verify_topn = std::min(params.verify_topn, 4);
        params.config.root_initial_width = 192;
        params.config.root_max_candidates = 512;
        params.config.node_initial_width = 8;
        params.config.node_max_candidates = 160;
        params.config.rollout_plies = 6;
        params.config.uct_c = 0.50;
    } else if (arrows < 45) {
        params.config.root_initial_width = 160;
        params.config.root_max_candidates = 448;
        params.config.node_initial_width = 8;
        params.config.node_max_candidates = 192;
        params.config.rollout_plies = 6;
        params.config.uct_c = 0.45;
    } else {
        params.time_share = std::min(params.time_share, 60);
        if (params.verify_topn > 0)
            params.verify_topn = std::max(params.verify_topn, 8);
        params.config.root_initial_width = 96;
        params.config.root_max_candidates = 256;
        params.config.node_initial_width = 6;
        params.config.node_max_candidates = 128;
        params.config.rollout_plies = 4;
        params.config.uct_c = 0.40;
    }

    if (remaining_ms < 300) {
        params.time_share = std::min(params.time_share, 70);
        if (params.verify_topn > 0)
            params.verify_topn = std::min(params.verify_topn, 3);
        params.config.root_initial_width = std::min(params.config.root_initial_width, 96);
        params.config.root_max_candidates = std::min(params.config.root_max_candidates, 256);
        params.config.node_max_candidates = std::min(params.config.node_max_candidates, 96);
        params.config.rollout_plies = std::min(params.config.rollout_plies, 4);
    } else if (remaining_ms >= 1500 && arrows < 20) {
        params.config.root_initial_width = 256;
        params.config.root_max_candidates = 640;
    }

    return params;
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

void Searcher::new_game() {
    std::fill(&history_[0][0][0], &history_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::fill(&arrow_history_[0][0][0], &arrow_history_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::fill(&from_arrow_history_[0][0][0], &from_arrow_history_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::fill(&butterfly_from_to_[0][0][0], &butterfly_from_to_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::fill(&butterfly_to_arrow_[0][0][0], &butterfly_to_arrow_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::fill(&butterfly_from_arrow_[0][0][0], &butterfly_from_arrow_[0][0][0] + 2 * BOARD_SQ * BOARD_SQ, 0);
    std::memset(countermoves_, 0, sizeof(countermoves_));
}

void Searcher::update_histories(Move m, int depth, Stack* ss, Color us) {
    if (m != ss->killers[0]) {
        ss->killers[1] = ss->killers[0];
        ss->killers[0] = m;
    }

    const int bonus = depth * depth * 64;
    update_history_score(history_[us][move_from(m)][move_to(m)], bonus);
    update_history_score(arrow_history_[us][move_to(m)][move_arrow(m)], bonus);
    update_history_score(from_arrow_history_[us][move_from(m)][move_arrow(m)], bonus);

    const Move prev = (ss - 1)->current_move;
    if (prev != MOVE_NONE)
        countermoves_[us][move_from(prev)][move_to(prev)] = m;
}

void Searcher::update_butterflies(Move m, Color us) {
    auto bump = [](int& entry) {
        if (entry < (1 << 24))
            ++entry;
        else
            entry = (entry >> 1) + 1;
    };

    bump(butterfly_from_to_[us][move_from(m)][move_to(m)]);
    bump(butterfly_to_arrow_[us][move_to(m)][move_arrow(m)]);
    bump(butterfly_from_arrow_[us][move_from(m)][move_arrow(m)]);
}

Score Searcher::negamax(Position& pos, Score alpha, Score beta, int depth, Stack* ss, Move* best_move_out) {
    const uint64_t node = ++nodes;
    if (node_limit && node >= node_limit)
        stop_.store(true, std::memory_order_relaxed);
    if ((node & 0x1FFF) == 0)
        check_time();
    if (stop_.load(std::memory_order_relaxed))
        return SCORE_ZERO;

    const bool is_pv = (beta - alpha > 1);
    const bool is_root = (ss->ply == 0);

    if (ss->ply >= MAX_SEARCH_PLY)
        return evaluate_search(pos, is_pv, is_root, depth, alpha, beta);

    tt_.prefetch(pos.key);

    if (depth <= 0) {
        if (pos.bb_arrow.popcount() >= 72 && !has_legal_move(pos))
            return mated_in(ss->ply);
        return evaluate_search(pos, is_pv, is_root, depth, alpha, beta);
    }

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

    ss->static_eval = (hit && tt_eval != SCORE_NONE)
                    ? tt_eval
                    : evaluate_search(pos, is_pv, is_root, depth, alpha, beta);

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

    MovePicker picker(pos,
                      tt_move,
                      ss,
                      history_,
                      arrow_history_,
                      from_arrow_history_,
                      butterfly_from_to_,
                      butterfly_to_arrow_,
                      butterfly_from_arrow_,
                      countermove,
                      move_buffers_[std::min(ss->ply, 127)],
                      lmp_threshold);
    Move m;
    while ((m = picker.next_move()) != MOVE_NONE) {
        ++move_count;
        int category_score = 0;
        if (g_use_move_categories) {
            category_score = picker.last_category_known()
                           ? picker.last_category_score()
                           : amazon_move_category_score(pos, m, ss);
        }

        if (futility_ok && category_score < 2500
            && move_count > 1 && best_score > mated_in(MAX_SEARCH_PLY))
            continue;

        if (!is_root && !is_pv
            && move_count > lmp_threshold
            && best_score > mated_in(MAX_SEARCH_PLY))
            if (!g_use_soft_tail_search)
                break;

        const Color us = pos.side_to_move;
        const bool is_tail = picker.last_tail_known() && picker.last_tail();
        update_butterflies(m, us);
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
                                  + arrow_history_[us][move_to(m)][move_arrow(m)]
                                  + from_arrow_history_[us][move_from(m)][move_arrow(m)];
                if (h_score > 4000 || category_score > 3000)
                    reduction = std::max(0, reduction - 1);
                else if (h_score < -4000 || category_score < -1500)
                    reduction += 1;

                if (is_tail && !is_root && !is_pv)
                    reduction += 2 + depth / 4;

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
                    const Move* tried_moves = picker.tried_moves();
                    for (int i = 0; i < picker.tried_count(); ++i) {
                        const Move tried = tried_moves[i];
                        if (tried == m)
                            break;
                        update_history_score(history_[pos.side_to_move][move_from(tried)][move_to(tried)],
                                             malus * 64);
                        update_history_score(arrow_history_[pos.side_to_move][move_to(tried)][move_arrow(tried)],
                                             malus * 64);
                        update_history_score(from_arrow_history_[pos.side_to_move][move_from(tried)][move_arrow(tried)],
                                             malus * 64);
                    }
                    break;
                }
            }
        }
    }

    if (move_count == 0)
        return mated_in(ss->ply);

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
    last_mcts_valid_ = false;
    last_mcts_result_ = MctsResult{};

    if (thread_id == 0)
        tt_.new_search();

    max_depth = std::min(max_depth, MAX_SEARCH_PLY);

    if (should_use_mcts_root(pos, max_depth, thread_id, time_man, node_limit)) {
        const int remaining = std::max(1, time_man.soft_limit_ms() - time_man.elapsed_ms());
        const AdaptiveMctsParams adaptive = adaptive_mcts_params(pos, remaining);
        const int budget = std::clamp(remaining * adaptive.time_share / 100,
                                      g_mcts_min_time_ms,
                                      remaining);
        MctsResult mcts = run_mcts_root(pos, budget, adaptive.config, &stop_);

        if (mcts.best_move != MOVE_NONE && is_pseudo_legal(pos, mcts.best_move)) {
            last_mcts_result_ = mcts;
            last_mcts_valid_ = true;
            Move best_move = mcts.best_move;
            Score best_score = mcts.score;

            if (!silent) {
                std::cout << "info string mcts playouts " << mcts.playouts
                          << " rootChildren " << mcts.root_children
                          << " rootCandidates " << mcts.root_candidates
                          << " budgetMs " << budget
                          << " share " << adaptive.time_share
                          << " rollout " << adaptive.config.rollout_plies
                          << " best " << move_to_str(best_move)
                          << std::endl;
            }

            const int verify_count = std::min<int>(adaptive.verify_topn,
                                                   mcts.root_moves.size());
            const int verify_remaining = time_man.soft_limit_ms() - time_man.elapsed_ms();
            const int verify_depth = verify_remaining < 500 ? 2
                                   : verify_remaining < 1000 ? 3
                                   : std::clamp(max_depth / 3, 2, 4);
            const int verify_limit = verify_remaining < 500 ? std::min(verify_count, 3)
                                   : verify_remaining < 1000 ? std::min(verify_count, 5)
                                   : verify_count;
            if (verify_limit > 0 && verify_remaining >= 80 && verify_depth >= 2 && !time_man.soft_stop()) {
                std::array<Stack, 128> stack{};
                Stack* ss = stack.data() + 4;
                ss->ply = 0;

                for (int i = 0; i < verify_limit; ++i) {
                    const Move m = mcts.root_moves[i].move;
                    if (time_man.soft_stop() || stop_.load(std::memory_order_relaxed))
                        break;
                    if (!is_pseudo_legal(pos, m))
                        continue;

                    pos.do_move(m);
                    ss->current_move = m;
                    (ss + 1)->current_move = m;
                    (ss + 1)->ply = 1;
                    Score score = -negamax(pos, -SCORE_INF, SCORE_INF, verify_depth - 1, ss + 1);
                    pos.undo_move(m);

                    if (stop_.load(std::memory_order_relaxed))
                        break;
                    if (m == best_move || score > best_score) {
                        best_score = score;
                        best_move = m;
                    }
                }
            }

            if (out_score)
                *out_score = best_score;
            if (!silent) {
                std::cout << "info depth mcts score cp " << best_score
                          << " nodes " << nodes
                          << " time " << time_man.elapsed_ms()
                          << " pv " << move_to_str(best_move)
                          << std::endl;
                std::cout << "bestmove " << move_to_str(best_move) << std::endl;
            }
            return best_move;
        }
    }

    std::array<Move, MAX_LEGAL_MOVES> root_moves{};
    const int root_count = generate_moves(pos, root_moves.data(), MAX_LEGAL_MOVES);
    Move best_move = root_count == 0 ? MOVE_NONE : root_moves[0];
    Score best_score = SCORE_ZERO;

    std::array<Stack, 128> stack{};
    Stack* ss = stack.data() + 4;
    ss->ply = 0;

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

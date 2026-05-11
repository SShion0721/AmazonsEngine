/*==================================================================
 * AMAZONS ENGINE - mcts.h
 * Root-only UCT/MCTS search for high-branching Amazons openings.
 *==================================================================*/
#pragma once

#include "position.h"
#include <atomic>
#include <cstdint>
#include <vector>

struct MctsRootMove {
    Move move = MOVE_NONE;
    int visits = 0;
    double mean = 0.0;
    int prior = 0;
    double prior_prob = 0.0;
};

struct MctsResult {
    Move best_move = MOVE_NONE;
    Score score = SCORE_ZERO;
    int playouts = 0;
    int root_children = 0;
    int root_candidates = 0;
    std::vector<MctsRootMove> root_moves;
};

struct MctsConfig {
    int root_initial_width = 128;
    int root_width_step = 8;
    int width_visit_step = 512;
    int root_max_candidates = 384;
    int node_initial_width = 8;
    int node_max_candidates = 192;
    int rollout_plies = 6;
    double uct_c = 0.45;
    double cpuct = 1.20;
    int policy_blend = 70;
    int threads = 1;
    bool use_policy_prior = true;
    bool use_tree_reuse = true;
};

MctsResult run_mcts_root(const Position& root,
                         int max_ms,
                         const MctsConfig& config = MctsConfig{},
                         const std::atomic<bool>* stop = nullptr);

void clear_mcts_tree();

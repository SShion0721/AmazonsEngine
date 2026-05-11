/*==================================================================
 * AMAZONS ENGINE -- search.h
 * Negamax Alpha-Beta search engine.
 * Mirrors Stockfish's search.h / search.cpp
 *
 * Techniques implemented (all with SF-style annotations):
 *   * Negamax with Alpha-Beta pruning
 *   * Principal Variation Search (PVS / NegaScout)
 *   * Iterative Deepening (ID)
 *   * Aspiration Windows
 *   * Transposition Table (TT) cutoffs
 *   * Reverse Futility Pruning (RFP)
 *   * Late Move Reduction (LMR) with a precomputed log table
 *   * Killer Move Heuristic (2 per depth)
 *   * History Heuristic for move ordering
 *   * Time management (soft + hard limits)
 *==================================================================*/
#pragma once
#include "position.h"
#include "tt.h"
#include "movepicker.h"
#include "mcts.h"
#include <atomic>
#include <chrono>
#include <array>
#include <memory>

extern bool g_use_null_move_pruning;
extern bool g_use_lmp_pruning;
extern bool g_use_move_categories;
extern bool g_teacher_safe_search;
extern bool g_use_soft_tail_search;
extern bool g_use_mcts_root;
extern int g_mcts_time_share;
extern int g_ab_verify_topn;
extern int g_mcts_min_time_ms;
extern int g_mcts_cpuct;
extern int g_mcts_threads;
extern bool g_use_tree_reuse;

enum SearchMode {
    SEARCH_MODE_FAST_SELFPLAY = 0,
    SEARCH_MODE_TEACHER_SAFE = 1,
    SEARCH_MODE_MATCH = 2
};

extern int g_search_mode;

// ==== Search stack (one entry per depth) ====
// Mirrors SF's Search::Stack.
struct Stack {
    Move  killers[2]   = { MOVE_NONE, MOVE_NONE };
    Score static_eval  = SCORE_NONE;
    Move  current_move = MOVE_NONE;
    int   move_count   = 0;  // number of moves searched at this node
    int   ply          = 0;  // search ply from root
    bool  in_null_move = false; // true if this node is inside NMP
    Move  excluded     = MOVE_NONE; // for Singular Extensions
};

// ==== TimeManager ====
// Decides when to stop the search (mirrors SF's TimeManagement).
class TimeManager {
public:
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::milliseconds;

    void init(int time_ms, int inc_ms, int movestogo, int move_overhead_ms = 0) {
        start_ = Clock::now();
        int moves = movestogo > 0 ? std::min(movestogo, 50) : 50;
        if (time_ms < 1000)
            moves = std::max(1, time_ms / 20);

        int time_left = std::max(1, time_ms + inc_ms * (moves - 1)
                                    - move_overhead_ms * (2 + moves));
        soft_limit_ = std::max(1, time_left / std::max(1, moves));
        hard_limit_ = std::max(soft_limit_,
                               std::min(std::max(1, time_ms - move_overhead_ms), soft_limit_ * 5));
    }

    void set_movetime(int ms, int move_overhead_ms = 0) {
        start_ = Clock::now();
        int budget = std::max(1, ms - move_overhead_ms);
        soft_limit_ = budget;
        hard_limit_ = budget;
    }

    int elapsed_ms() const {
        return static_cast<int>(
            std::chrono::duration_cast<Ms>(Clock::now() - start_).count());
    }

    bool soft_stop() const { return elapsed_ms() >= soft_limit_; }
    bool hard_stop() const { return elapsed_ms() >= hard_limit_; }
    int soft_limit_ms() const { return soft_limit_; }
    int hard_limit_ms() const { return hard_limit_; }

private:
    Clock::time_point start_;
    int soft_limit_ = 5000;
    int hard_limit_ = 10000;
};

// ==== Searcher ====
class Searcher {
public:
    explicit Searcher(TranspositionTable& tt)
        : tt_(tt),
          move_buffers_(std::make_unique<MovePickerBuffer[]>(128)) {
        init_tables();
    }

    // Main entry point: run iterative deepening and return the best move.
    Move search(Position& pos, int max_depth = 64, Score* out_score = nullptr, int thread_id = 0);
    void new_game();
    bool last_mcts_valid() const { return last_mcts_valid_; }
    const MctsResult& last_mcts_result() const { return last_mcts_result_; }

    // Stop the search at the next opportunity.
    void request_stop() { stop_.store(true, std::memory_order_relaxed); }

    // Silent mode: suppress all stdout output (for selfplay)
    bool silent = false;

    // Time manager (configure before calling search())
    TimeManager time_man;

    // Node counter (informational)
    uint64_t nodes = 0;
    uint64_t node_limit = 0;

private:
    TranspositionTable& tt_;
    std::atomic<bool>   stop_{ false };

    // ==== Killer moves [depth][0..1] ====
    // "Killer" moves are quiet moves that caused a beta cutoff at the
    // same depth in another branch. They are tried early in move ordering.
    std::array<std::array<Move, 2>, 128> killers_{};

    // ---- History heuristic [color][from][to] ----
    // Bonus accumulated each time move (from->to) causes a beta cutoff.
    // Bigger bonus = move is tried earlier in sibling nodes.
    int history_[2][BOARD_SQ][BOARD_SQ]{};

    // Arrow history [color][amazon_to][arrow]. In Amazons the arrow square
    // often carries more tactical meaning than the amazon destination.
    int arrow_history_[2][BOARD_SQ][BOARD_SQ]{};
    int from_arrow_history_[2][BOARD_SQ][BOARD_SQ]{};
    int butterfly_from_to_[2][BOARD_SQ][BOARD_SQ]{};
    int butterfly_to_arrow_[2][BOARD_SQ][BOARD_SQ]{};
    int butterfly_from_arrow_[2][BOARD_SQ][BOARD_SQ]{};

    // ---- LMR table [depth][move_index] ----
    // Precomputed reduction amounts (mirrors SF's lmr[][]).
    int lmr_[64][64]{};

    // ---- Countermove table [color][from][to] -> Move ----
    // The "best reply" to a given previous move. Tried right after
    // TT move and killers. Mirrors SF's Countermoves.
    Move countermoves_[2][BOARD_SQ][BOARD_SQ]{};
    std::unique_ptr<MovePickerBuffer[]> move_buffers_;
    MctsResult last_mcts_result_{};
    bool last_mcts_valid_ = false;

    void init_tables();

    // Core recursive search (Negamax PVS)
    Score negamax(Position& pos, Score alpha, Score beta,
                  int depth, Stack* ss, Move* best_move_out = nullptr);

    // Update killer + history after a beta cutoff
    void update_histories(Move m, int depth, Stack* ss, Color us);
    void update_butterflies(Move m, Color us);

    // Periodically check time; sets stop_ flag if over hard limit
    void check_time();
};


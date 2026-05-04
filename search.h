/*==================================================================
 * AMAZONS ENGINE 鈥?search.h
 * Negamax Alpha-Beta search engine.
 * Mirrors Stockfish's search.h / search.cpp
 *
 * Techniques implemented (all with SF-style annotations):
 *   鉁?Negamax with Alpha-Beta pruning
 *   鉁?Principal Variation Search (PVS / NegaScout)
 *   鉁?Iterative Deepening (ID)
 *   鉁?Aspiration Windows
 *   鉁?Transposition Table (TT) cutoffs
 *   鉁?Reverse Futility Pruning (RFP)
 *   鉁?Late Move Reduction (LMR) with a precomputed log table
 *   鉁?Killer Move Heuristic (2 per depth)
 *   鉁?History Heuristic for move ordering
 *   鉁?Time management (soft + hard limits)
 *==================================================================*/
#pragma once
#include "position.h"
#include "tt.h"
#include "movepicker.h"
#include <atomic>
#include <chrono>
#include <array>
#include <memory>

extern bool g_use_null_move_pruning;
extern bool g_use_lmp_pruning;
extern bool g_use_move_categories;
extern bool g_teacher_safe_search;

// 鈹€鈹€ Search stack (one entry per depth) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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

// 鈹€鈹€ TimeManager 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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

private:
    Clock::time_point start_;
    int soft_limit_ = 5000;
    int hard_limit_ = 10000;
};

// 鈹€鈹€ Searcher 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class Searcher {
public:
    explicit Searcher(TranspositionTable& tt)
        : tt_(tt),
          move_buffers_(std::make_unique<MovePickerBuffer[]>(128)) {
        init_tables();
    }

    // Main entry point: run iterative deepening and return the best move.
    Move search(Position& pos, int max_depth = 64, Score* out_score = nullptr, int thread_id = 0);

    // Stop the search at the next opportunity.
    void request_stop() { stop_.store(true, std::memory_order_relaxed); }

    // Silent mode: suppress all stdout output (for selfplay)
    bool silent = false;

    // Time manager (configure before calling search())
    TimeManager time_man;

    // Node counter (informational)
    uint64_t nodes = 0;

private:
    TranspositionTable& tt_;
    std::atomic<bool>   stop_{ false };

    // 鈹€鈹€ Killer moves [depth][0..1] 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // "Killer" moves are quiet moves that caused a beta cutoff at the
    // same depth in another branch. They are tried early in move ordering.
    std::array<std::array<Move, 2>, 128> killers_{};

    // 鈹€鈹€ History heuristic [color][from][to] 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // Bonus accumulated each time move (from鈫抰o) causes a beta cutoff.
    // Bigger bonus = move is tried earlier in sibling nodes.
    int history_[2][BOARD_SQ][BOARD_SQ]{};

    // Arrow history [color][amazon_to][arrow]. In Amazons the arrow square
    // often carries more tactical meaning than the amazon destination.
    int arrow_history_[2][BOARD_SQ][BOARD_SQ]{};
    int from_arrow_history_[2][BOARD_SQ][BOARD_SQ]{};

    // 鈹€鈹€ LMR table [depth][move_index] 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // Precomputed reduction amounts (mirrors SF's lmr[][]).
    int lmr_[64][64]{};

    // 鈹€鈹€ Countermove table [color][from][to] 鈫?Move 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // The "best reply" to a given previous move. Tried right after
    // TT move and killers. Mirrors SF's Countermoves.
    Move countermoves_[2][BOARD_SQ][BOARD_SQ]{};
    std::unique_ptr<MovePickerBuffer[]> move_buffers_;

    void init_tables();

    // Core recursive search (Negamax PVS)
    Score negamax(Position& pos, Score alpha, Score beta,
                  int depth, Stack* ss, Move* best_move_out = nullptr);

    // Update killer + history after a beta cutoff
    void update_histories(Move m, int depth, Stack* ss, Color us);

    // Periodically check time; sets stop_ flag if over hard limit
    void check_time();
};


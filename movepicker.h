/*==================================================================
 * AMAZONS ENGINE — movepicker.h
 * Lazy move generation: yields moves on demand.
 * Mirrors Stockfish's MovePicker — TT/Killer/Counter first,
 * then generate+sort remaining only if needed.
 *==================================================================*/
#pragma once
#include "types.h"
#include "position.h"
#include "movegen.h"
#include <vector>
#include <algorithm>

// Forward declare Stack to avoid circular include
struct Stack;

class MovePicker {
public:
    MovePicker(const Position& pos, Move tt_move, Stack* ss,
               const int (&history)[2][BOARD_SQ][BOARD_SQ],
               const int (&arrow_history)[2][BOARD_SQ][BOARD_SQ],
               Move countermove)
        : pos_(pos), tt_move_(tt_move), ss_(ss),
          history_(history), arrow_history_(arrow_history), counter_(countermove),
          stage_(STAGE_TT), move_idx_(0) {}

    Move next_move();

    // Access generated moves (for history malus after cutoff)
    const std::vector<Move>& tried_moves() const { return tried_; }

private:
    enum Stage {
        STAGE_TT, STAGE_KILLER_1, STAGE_KILLER_2,
        STAGE_COUNTER, STAGE_GENERATE, STAGE_YIELD, STAGE_DONE
    };

    const Position& pos_;
    Move            tt_move_;
    Stack*          ss_;
    const int     (&history_)[2][BOARD_SQ][BOARD_SQ];
    const int     (&arrow_history_)[2][BOARD_SQ][BOARD_SQ];
    Move            counter_;
    Stage           stage_;

    struct ScoredMove { Move move; int score; };
    std::vector<ScoredMove> moves_;
    std::vector<Move>       tried_;
    size_t                  move_idx_;

    bool already_tried(Move m) const {
        for (Move t : tried_)
            if (t == m) return true;
        return false;
    }
};

int amazon_move_category_score(const Position& pos, Move m, const Stack* ss);

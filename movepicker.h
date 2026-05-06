/*==================================================================
 * AMAZONS ENGINE - movepicker.h
 * Fixed-buffer move picker for high-branching Amazons search.
 *==================================================================*/
#pragma once

#include "movegen.h"
#include "position.h"
#include "types.h"
#include <algorithm>

struct Stack;

extern bool g_use_bounded_movegen;
extern int g_bounded_dest_cap;
extern int g_bounded_arrow_cap;

struct ScoredMove {
    Move move;
    int score;
    int category;
};

struct MovePickerBuffer {
    alignas(64) Move raw[MAX_LEGAL_MOVES];
    alignas(64) ScoredMove moves[MAX_LEGAL_MOVES];
    alignas(64) Move tried[MAX_LEGAL_MOVES];
};

class MovePicker {
public:
    MovePicker(const Position& pos,
               Move tt_move,
               Stack* ss,
               const int (&history)[2][BOARD_SQ][BOARD_SQ],
               const int (&arrow_history)[2][BOARD_SQ][BOARD_SQ],
               const int (&from_arrow_history)[2][BOARD_SQ][BOARD_SQ],
               Move countermove,
               MovePickerBuffer& buffer,
               int top_k)
        : pos_(pos),
          tt_move_(tt_move),
          ss_(ss),
          history_(history),
          arrow_history_(arrow_history),
          from_arrow_history_(from_arrow_history),
          counter_(countermove),
          buffer_(buffer),
          top_k_(top_k),
          stage_(STAGE_TT) {}

    Move next_move();

    const Move* tried_moves() const { return buffer_.tried; }
    int tried_count() const { return tried_count_; }
    int last_category_score() const { return last_category_score_; }
    bool last_category_known() const { return last_category_known_; }

private:
    enum Stage {
        STAGE_TT,
        STAGE_KILLER_1,
        STAGE_KILLER_2,
        STAGE_COUNTER,
        STAGE_GENERATE,
        STAGE_YIELD,
        STAGE_DONE
    };

    const Position& pos_;
    Move tt_move_;
    Stack* ss_;
    const int (&history_)[2][BOARD_SQ][BOARD_SQ];
    const int (&arrow_history_)[2][BOARD_SQ][BOARD_SQ];
    const int (&from_arrow_history_)[2][BOARD_SQ][BOARD_SQ];
    Move counter_;
    MovePickerBuffer& buffer_;
    int top_k_;
    Stage stage_;
    int move_count_ = 0;
    int move_idx_ = 0;
    int tried_count_ = 0;
    int excluded_count_ = 0;
    Move excluded_[4] = {MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_NONE};
    int last_category_score_ = 0;
    bool last_category_known_ = false;

    bool is_excluded(Move m) const {
        for (int i = 0; i < excluded_count_; ++i)
            if (excluded_[i] == m)
                return true;
        return false;
    }

    void remember_special(Move m) {
        if (excluded_count_ < 4)
            excluded_[excluded_count_++] = m;
        remember_tried(m);
        last_category_known_ = false;
        last_category_score_ = 0;
    }

    void remember_tried(Move m) {
        if (tried_count_ < MAX_LEGAL_MOVES)
            buffer_.tried[tried_count_++] = m;
    }
};

int amazon_move_category_score(const Position& pos, Move m, const Stack* ss);

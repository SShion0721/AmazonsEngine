/*==================================================================
 * AMAZONS ENGINE - evaluate.h
 * Static position evaluation.
 *==================================================================*/
#pragma once
#include "position.h"

// Runtime-tunable legacy evaluation weights.
extern int g_territory_weight;
extern int g_mobility_weight;

void set_territory_weight(int value);
void set_mobility_weight(int value);

struct EvalBreakdown {
    int phase = 0;
    int active_areas = 0;
    Score t1 = 0;        // queen-distance territory, white POV
    Score t2 = 0;        // king-distance territory, white POV
    Score p1 = 0;        // queen-distance closeness, white POV
    Score p2 = 0;        // king-distance closeness, white POV
    Score m = 0;         // per-amazon mobility/enclosure, white POV
    Score partition = 0; // separated territory, white POV
    Score raw_white = 0;
    Score classical = 0; // side-to-move POV
};

// Returns the static evaluation from the side-to-move's perspective.
Score evaluate(const Position& pos);
Score evaluate_classical(const Position& pos);
Score evaluate_classical2(const Position& pos);

// Flag: only use NNUE when real weights are loaded.
extern bool g_nnue_loaded;
extern bool g_use_nnue;
extern bool g_use_residual_nnue;
extern bool g_use_pure_nnue;
extern bool g_use_strong_classical;

// Sub-components exposed for debug/tuning.
Score eval_territory(const Position& pos); // legacy king flood territory, white POV
Score eval_mobility (const Position& pos); // queen mobility, white POV
bool get_eval_breakdown(const Position& pos, EvalBreakdown& out);

// Clear the eval cache (call on ucinewgame / option changes).
void clear_eval_cache();

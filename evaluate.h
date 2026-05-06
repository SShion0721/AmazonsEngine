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
    int phase_bucket = 0;
    int active_areas = 0;
    Score t1 = 0;        // queen-distance territory, white POV
    Score t2 = 0;        // king-distance territory, white POV
    Score p1 = 0;        // queen-distance closeness, white POV
    Score p2 = 0;        // king-distance closeness, white POV
    Score m = 0;         // per-amazon mobility/enclosure, white POV
    Score partition = 0; // separated territory, white POV
    Score mobility = 0;  // queen mobility, white POV
    Score raw_white = 0;
    Score classical = 0; // side-to-move POV
};

struct GlobalFeatures {
    int8_t v[NNUE::GLOBAL_FEATURE_SIZE]{};
};

struct EvalInfo {
    EvalBreakdown breakdown{};
    GlobalFeatures global{};
    int phase_bucket = 0;
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

enum EvalTier {
    EVAL_TIER_FAST = 0,
    EVAL_TIER_MIXED = 1,
    EVAL_TIER_STRONG = 2
};

extern int g_eval_tier;

// Sub-components exposed for debug/tuning.
Score eval_territory(const Position& pos); // legacy king flood territory, white POV
Score eval_mobility (const Position& pos); // queen mobility, white POV
bool get_eval_breakdown(const Position& pos, EvalBreakdown& out);
bool get_eval_info(const Position& pos, EvalInfo& out);
bool partition_fast_track(const Position& pos, Score& out);
Score evaluate_with_tier(const Position& pos, int tier);
Score evaluate_search(const Position& pos, bool is_pv, bool is_root, int depth, Score alpha, Score beta);

// Clear the eval cache (call on ucinewgame / option changes).
void clear_eval_cache();

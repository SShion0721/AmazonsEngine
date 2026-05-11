/*==================================================================
 * AMAZONS ENGINE - policy_prior.h
 * Optional CPU policy prior used by root PUCT.
 *==================================================================*/
#pragma once

#include "position.h"
#include <string>

struct PolicyPrior {
    alignas(64) int16_t from[BOARD_SQ]{};
    alignas(64) int16_t to[BOARD_SQ]{};
    alignas(64) int16_t arrow[BOARD_SQ]{};
};

extern bool g_use_policy_prior;
extern int g_policy_prior_blend;

bool load_policy_prior(const std::string& path);
bool policy_prior_loaded();
const char* policy_prior_format();
PolicyPrior evaluate_policy_prior(const Position& pos);
int policy_move_prior(const PolicyPrior& prior, Move m);

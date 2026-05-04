/*==================================================================
 * AMAZONS ENGINE - tt.h
 * Stockfish-inspired transposition table with cache-line clusters.
 *==================================================================*/
#pragma once

#include "types.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <xmmintrin.h>
#endif

enum TTBound : uint8_t {
    TT_NONE  = 0,
    TT_EXACT = 1,
    TT_LOWER = 2,
    TT_UPPER = 3,
};

// genBound8 layout: [PV:1][BOUND:2][GENERATION:5]
static constexpr uint8_t GEN_BITS = 5;
static constexpr uint8_t GEN_MASK = (1 << GEN_BITS) - 1;
static constexpr uint8_t BOUND_SHIFT = GEN_BITS;
static constexpr uint8_t BOUND_MASK = 0b11 << BOUND_SHIFT;
static constexpr uint8_t PV_SHIFT = BOUND_SHIFT + 2;
static constexpr uint8_t PV_MASK = 1 << PV_SHIFT;

// 16-byte entries give us 4 entries per 64-byte cache line while still storing
// a 32-bit Amazons move and a 32-bit partial key.
struct TTEntry {
    uint32_t key32 = 0;
    int32_t  move32 = MOVE_NONE;
    int16_t  value16 = 0;
    int16_t  eval16 = 0;
    uint8_t  depth8 = 0;
    uint8_t  genBound8 = 0;
    uint16_t padding = 0;

    bool is_occupied() const { return depth8 != 0; }

    TTBound bound() const {
        return TTBound((genBound8 & BOUND_MASK) >> BOUND_SHIFT);
    }

    bool is_pv() const { return (genBound8 & PV_MASK) != 0; }

    int depth() const { return int(depth8) - 1; }

    Score score() const { return Score(value16); }
    Score eval() const { return Score(eval16); }
    Move best_move() const { return Move(move32); }

    uint8_t relative_age(uint8_t curr_gen) const {
        return (curr_gen - genBound8) & GEN_MASK;
    }

    void save(uint64_t key, Score value, bool pv, TTBound bound_type, int depth_value,
              Move move, Score static_eval, uint8_t curr_gen) {
        const uint32_t key32_value = uint32_t(key);

        if (move != MOVE_NONE || key32_value != key32)
            move32 = int32_t(move);

        if (bound_type == TT_EXACT || key32_value != key32
            || depth_value + 2 * int(pv) > int(depth8) - 4
            || relative_age(curr_gen)) {
            key32 = key32_value;
            depth8 = uint8_t(depth_value + 1);
            genBound8 = uint8_t(curr_gen | (bound_type << BOUND_SHIFT) | (uint8_t(pv) << PV_SHIFT));
            value16 = int16_t(value);
            eval16 = int16_t(static_eval);
        }
    }
};

static_assert(sizeof(TTEntry) == 16, "TTEntry must stay 16 bytes");

static constexpr int ClusterSize = 4;

struct TTCluster {
    TTEntry entry[ClusterSize];
};

static_assert(sizeof(TTCluster) == 64, "TTCluster must be 64 bytes");

class TranspositionTable {
public:
    explicit TranspositionTable(size_t mb = 64) { resize(mb); }
    ~TranspositionTable() { free(table_); }

    void resize(size_t mb) {
        free(table_);
        cluster_count_ = mb * 1024 * 1024 / sizeof(TTCluster);
        if (cluster_count_ == 0)
            cluster_count_ = 1;
        table_ = static_cast<TTCluster*>(calloc(cluster_count_, sizeof(TTCluster)));
        generation8_ = 0;
    }

    void clear() {
        std::memset(table_, 0, cluster_count_ * sizeof(TTCluster));
        generation8_ = 0;
    }

    void new_search() {
        generation8_ = (generation8_ + 1) & GEN_MASK;
    }

    uint8_t generation() const { return generation8_; }

    struct ProbeResult {
        bool hit;
        TTEntry* entry;
    };

    ProbeResult probe(uint64_t key) const {
        TTEntry* tte = first_entry(key);
        const uint32_t key32_value = uint32_t(key);

        for (int i = 0; i < ClusterSize; ++i) {
            if (tte[i].key32 == key32_value && tte[i].is_occupied())
                return {true, &tte[i]};
        }

        TTEntry* replace = &tte[0];
        for (int i = 1; i < ClusterSize; ++i) {
            if (replace->depth8 - 8 * replace->relative_age(generation8_)
                > tte[i].depth8 - 8 * tte[i].relative_age(generation8_)) {
                replace = &tte[i];
            }
        }

        return {false, replace};
    }

    void store(uint64_t key, Score score, bool pv, TTBound bound,
               int depth, Move best_move, Score static_eval) {
        auto [hit, tte] = probe(key);
        (void)hit;
        tte->save(key, score, pv, bound, depth, best_move, static_eval, generation8_);
    }

    void prefetch(uint64_t key) const {
#ifdef _MSC_VER
        _mm_prefetch(reinterpret_cast<const char*>(first_entry(key)), _MM_HINT_T0);
#else
        __builtin_prefetch(first_entry(key));
#endif
    }

    int hashfull() const {
        int cnt = 0;
        int sample = std::min<size_t>(1000, cluster_count_);
        for (int i = 0; i < sample; ++i) {
            for (int j = 0; j < ClusterSize; ++j) {
                cnt += table_[i].entry[j].is_occupied()
                    && table_[i].entry[j].relative_age(generation8_) == 0;
            }
        }
        return cnt / ClusterSize;
    }

private:
    TTCluster* table_ = nullptr;
    size_t cluster_count_ = 0;
    uint8_t generation8_ = 0;

    TTEntry* first_entry(uint64_t key) const {
#ifdef _MSC_VER
        uint64_t hi;
        _umul128(key, uint64_t(cluster_count_), &hi);
        return const_cast<TTEntry*>(&table_[hi].entry[0]);
#else
        size_t idx = size_t(((unsigned __int128)key * (unsigned __int128)cluster_count_) >> 64);
        return &table_[idx].entry[0];
#endif
    }
};

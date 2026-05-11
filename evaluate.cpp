/*==================================================================
 * AMAZONS ENGINE - evaluate.cpp
 * Static position evaluation.
 *==================================================================*/
#include "evaluate.h"
#include "nnue/nnue.h"
#include "bitboard.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

int g_territory_weight = 10;
int g_mobility_weight = 2;

bool g_nnue_loaded = false;
bool g_use_nnue = false;
bool g_use_residual_nnue = true;
bool g_use_pure_nnue = false;
bool g_use_strong_classical = true;
int g_eval_tier = EVAL_TIER_MIXED;

namespace {

constexpr int DIST_INF = 127;
constexpr int EVAL_CACHE_SIZE = 1 << 15;
constexpr int EVAL_CACHE_MASK = EVAL_CACHE_SIZE - 1;

struct DistanceInfo {
    int qdist[2][BOARD_SQ];
    int kdist[2][BOARD_SQ];
};

struct EvalCacheEntry {
    uint64_t key = 0;
    uint32_t epoch = 0;
    EvalInfo info{};
};

std::atomic<uint32_t> g_eval_cache_epoch{1};
thread_local std::array<EvalCacheEntry, EVAL_CACHE_SIZE> t_eval_cache;

static Score clamp_static_eval(Score value) {
    return std::clamp(value, -SCORE_INF + 1, SCORE_INF - 1);
}

static int phase_bucket(int arrows) {
    if (arrows <= 14) return 0;
    if (arrows <= 25) return 1;
    if (arrows <= 40) return 2;
    if (arrows <= 55) return 3;
    if (arrows <= 63) return 4;
    return 5;
}

static int fusion_phase_bucket(int arrows) {
    return std::clamp(arrows / 12, 0, NNUE::PHASE_BUCKETS - 1);
}

static int8_t scaled_i8(int value, int divisor = 1) {
    if (divisor > 1)
        value /= divisor;
    return static_cast<int8_t>(std::clamp(value, -127, 127));
}

struct EvalWeights {
    int t1;
    int t2;
    int p1;
    int p2;
    int m;
    int part;
};

static EvalWeights weights_by_phase(int arrows) {
    static constexpr EvalWeights W[6] = {
        {10, 2, 2, 1, 1,  1},
        { 9, 3, 2, 2, 1,  2},
        { 8, 4, 2, 2, 2,  3},
        { 7, 5, 1, 2, 3,  5},
        { 6, 6, 1, 1, 4,  8},
        { 5, 6, 0, 1, 5, 12},
    };
    return W[phase_bucket(arrows)];
}

static bool should_try_partition_fast_track(const Position& pos) {
    return pos.bb_arrow.popcount() >= 56;
}

static Bitboard128 side_sources(const Position& pos, Color c) {
    return c == WHITE ? pos.bb_white : pos.bb_black;
}

static void compute_queen_distance_one(const Position& pos, Color c, int dist[BOARD_SQ]) {
    std::fill(dist, dist + BOARD_SQ, DIST_INF);

    Square queue[BOARD_SQ];
    int head = 0;
    int tail = 0;

    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square sq = pos.amazons[c][i];
        dist[sq] = 0;
        queue[tail++] = sq;
    }

    const Bitboard128 occ = pos.bb_occupied;
    while (head < tail) {
        const Square sq = queue[head++];
        const int d = dist[sq];
        Bitboard128 next = get_queen_attacks(sq, occ) & ~occ;

        while (next) {
            const Square to = next.pop_lsb();
            if (dist[to] <= d + 1)
                continue;
            dist[to] = d + 1;
            queue[tail++] = to;
        }
    }
}

static void compute_king_distance_one(const Position& pos, Color c, int dist[BOARD_SQ]) {
    std::fill(dist, dist + BOARD_SQ, DIST_INF);

    Bitboard128 visited = side_sources(pos, c);
    Bitboard128 front = visited;
    for (int i = 0; i < NUM_AMAZONS; ++i)
        dist[pos.amazons[c][i]] = 0;

    const Bitboard128 empty = ~pos.bb_occupied & BOARD_MASK_BB;
    int d = 0;
    while (front) {
        ++d;
        Bitboard128 next = king_attacks_all(front) & empty & ~visited;
        if (!next)
            break;

        visited |= next;
        Bitboard128 iter = next;
        while (iter)
            dist[iter.pop_lsb()] = d;
        front = next;
    }
}

static DistanceInfo compute_distances(const Position& pos) {
    DistanceInfo info;
    compute_queen_distance_one(pos, WHITE, info.qdist[WHITE]);
    compute_queen_distance_one(pos, BLACK, info.qdist[BLACK]);
    compute_king_distance_one(pos, WHITE, info.kdist[WHITE]);
    compute_king_distance_one(pos, BLACK, info.kdist[BLACK]);
    return info;
}

static Score eval_t1_queen_territory(const Position& pos, const DistanceInfo& info) {
    Score score = 0;
    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        if (pos.board[sq] != EMPTY)
            continue;

        const int w = info.qdist[WHITE][sq];
        const int b = info.qdist[BLACK][sq];
        if (w == b)
            continue;
        if (w < b)
            ++score;
        else if (b < w)
            --score;
    }
    return score;
}

static Score eval_t2_king_territory(const Position& pos, const DistanceInfo& info) {
    Score score = 0;
    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        if (pos.board[sq] != EMPTY)
            continue;

        const int w = info.kdist[WHITE][sq];
        const int b = info.kdist[BLACK][sq];
        if (w == b)
            continue;
        if (w < b)
            ++score;
        else if (b < w)
            --score;
    }
    return score;
}

static Score eval_p1_queen_position(const Position& pos, const DistanceInfo& info) {
    static constexpr int closeness[9] = {16, 8, 4, 2, 1, 1, 1, 1, 1};
    Score score = 0;

    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        if (pos.board[sq] != EMPTY)
            continue;

        const int w = info.qdist[WHITE][sq];
        const int b = info.qdist[BLACK][sq];
        if (w < 9)
            score += closeness[w];
        if (b < 9)
            score -= closeness[b];
    }
    return score;
}

static Score eval_p2_king_position(const Position& pos, const DistanceInfo& info) {
    Score score = 0;
    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        if (pos.board[sq] != EMPTY)
            continue;

        const int w = info.kdist[WHITE][sq];
        const int b = info.kdist[BLACK][sq];
        if (w >= DIST_INF && b >= DIST_INF)
            continue;
        if (w >= DIST_INF) {
            score -= 6;
        } else if (b >= DIST_INF) {
            score += 6;
        } else {
            score += std::clamp(b - w, -6, 6);
        }
    }
    return score;
}

static int local_power_from(const Position& pos, Square sq, Bitboard128 occ) {
    Bitboard128 reach = get_queen_attacks(sq, occ) & ~occ;
    int power = 0;

    while (reach) {
        const Square to = reach.pop_lsb();
        const int dist = std::max(std::abs(sq_row(to) - sq_row(sq)),
                                  std::abs(sq_col(to) - sq_col(sq)));
        const int empty_neighbors = (NEIGHBOR_BB[to] & ~occ & BOARD_MASK_BB).popcount();
        power += (empty_neighbors * 16) / std::max(1, dist);
    }
    return power;
}

static Score side_enclosure_score(const Position& pos, Color c) {
    Score score = 0;
    const Bitboard128 occ = pos.bb_occupied;

    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square sq = pos.amazons[c][i];
        const int ray_mobility = (get_queen_attacks(sq, occ) & ~occ).popcount();
        int power = local_power_from(pos, sq, occ);

        if (ray_mobility == 0)
            power -= 120;
        else if (ray_mobility <= 2)
            power -= 60;
        else if (ray_mobility <= 4)
            power -= 25;

        if (power <= 4)
            power -= 30;
        else if (power <= 12)
            power -= 12;

        score += power;
    }

    return score;
}

static Score eval_enclosure_m(const Position& pos) {
    return (side_enclosure_score(pos, WHITE) - side_enclosure_score(pos, BLACK)) / 16;
}

static Score eval_mobility_fast(const Position& pos) {
    const Bitboard128 occ = pos.bb_occupied;
    int white_mob = 0;
    int black_mob = 0;

    for (int i = 0; i < NUM_AMAZONS; i++) {
        const Square wsq = pos.amazons[WHITE][i];
        white_mob += (get_queen_attacks(wsq, occ) & ~occ).popcount();

        const Square bsq = pos.amazons[BLACK][i];
        black_mob += (get_queen_attacks(bsq, occ) & ~occ).popcount();
    }

    return white_mob - black_mob;
}

static int simple_component_value(int empty_count, int amazon_count) {
    if (amazon_count <= 0)
        return 0;
    return empty_count + std::min(empty_count, amazon_count * 2);
}

struct LocalSolverKey {
    uint64_t occ_lo = 0;
    uint64_t occ_hi = 0;
    uint8_t amazon_count = 0;
    uint8_t amazons[NUM_AMAZONS] = {};

    bool operator==(const LocalSolverKey& other) const {
        if (occ_lo != other.occ_lo || occ_hi != other.occ_hi
            || amazon_count != other.amazon_count)
            return false;
        for (int i = 0; i < amazon_count; ++i)
            if (amazons[i] != other.amazons[i])
                return false;
        return true;
    }
};

struct LocalSolverKeyHash {
    std::size_t operator()(const LocalSolverKey& k) const {
        uint64_t h = k.occ_lo ^ (k.occ_hi + 0x9e3779b97f4a7c15ULL + (k.occ_lo << 6) + (k.occ_lo >> 2));
        h ^= uint64_t(k.amazon_count) * 0xbf58476d1ce4e5b9ULL;
        for (int i = 0; i < k.amazon_count; ++i) {
            h ^= uint64_t(k.amazons[i] + 1) * 0x94d049bb133111ebULL;
            h ^= h >> 27;
        }
        return static_cast<std::size_t>(h);
    }
};

static constexpr int MIXED_SOLVER_UNKNOWN = 32767;

struct LocalFillSolver {
    static constexpr int MAX_EMPTY = 14;
    static constexpr int NODE_BUDGET = 200000;

    Bitboard128 component;
    Bitboard128 outside_occ;
    std::unordered_map<LocalSolverKey, uint8_t, LocalSolverKeyHash> memo;
    int nodes = 0;
    bool aborted = false;

    LocalSolverKey make_key(Bitboard128 local_occ, const Square* amazons, int amazon_count) const {
        LocalSolverKey key;
        key.occ_lo = local_occ.lo;
        key.occ_hi = local_occ.hi;
        const int count = std::min(amazon_count, NUM_AMAZONS);
        key.amazon_count = static_cast<uint8_t>(count);
        for (int i = 0; i < count; ++i)
            key.amazons[i] = static_cast<uint8_t>(amazons[i]);
        std::sort(key.amazons, key.amazons + count);
        return key;
    }

    int solve(Bitboard128 local_occ, Square* amazons, int amazon_count, int empty_left) {
        if (empty_left <= 0)
            return 0;
        if (++nodes > NODE_BUDGET) {
            aborted = true;
            return 0;
        }

        const LocalSolverKey key = make_key(local_occ, amazons, amazon_count);
        auto it = memo.find(key);
        if (it != memo.end())
            return it->second;

        int best = 0;
        const Bitboard128 occ = outside_occ | local_occ;

        for (int i = 0; i < amazon_count; ++i) {
            const Square from = amazons[i];
            Bitboard128 occ_lifted = occ;
            occ_lifted.clear(from);

            Bitboard128 tos = get_queen_attacks(from, occ_lifted) & ~occ_lifted & component;
            while (tos) {
                const Square to = tos.pop_lsb();
                Bitboard128 occ_after_to = occ_lifted;
                occ_after_to.set(to);

                Bitboard128 arrows = get_queen_attacks(to, occ_after_to) & ~occ_after_to & component;
                while (arrows) {
                    const Square arrow = arrows.pop_lsb();

                    Bitboard128 next_local_occ = local_occ;
                    next_local_occ.clear(from);
                    next_local_occ.set(to);
                    next_local_occ.set(arrow);

                    Square next_amazons[NUM_AMAZONS];
                    for (int j = 0; j < amazon_count; ++j)
                        next_amazons[j] = amazons[j];
                    next_amazons[i] = to;

                    const int value = 1 + solve(next_local_occ, next_amazons, amazon_count, empty_left - 1);
                    best = std::max(best, value);
                    if (best >= empty_left) {
                        memo[key] = static_cast<uint8_t>(best);
                        return best;
                    }
                    if (aborted)
                        return best;
                }
            }
        }

        memo[key] = static_cast<uint8_t>(best);
        return best;
    }
};

struct LocalMixedSolverKey {
    uint64_t occ_lo = 0;
    uint64_t occ_hi = 0;
    uint8_t side = WHITE;
    uint8_t white_count = 0;
    uint8_t black_count = 0;
    uint8_t white[NUM_AMAZONS] = {};
    uint8_t black[NUM_AMAZONS] = {};

    bool operator==(const LocalMixedSolverKey& other) const {
        if (occ_lo != other.occ_lo || occ_hi != other.occ_hi || side != other.side
            || white_count != other.white_count || black_count != other.black_count)
            return false;
        for (int i = 0; i < white_count; ++i)
            if (white[i] != other.white[i])
                return false;
        for (int i = 0; i < black_count; ++i)
            if (black[i] != other.black[i])
                return false;
        return true;
    }
};

struct LocalMixedSolverKeyHash {
    std::size_t operator()(const LocalMixedSolverKey& k) const {
        uint64_t h = k.occ_lo ^ (k.occ_hi + 0x9e3779b97f4a7c15ULL + (k.occ_lo << 6) + (k.occ_lo >> 2));
        h ^= uint64_t(k.side + 1) * 0xbf58476d1ce4e5b9ULL;
        h ^= uint64_t(k.white_count + 7 * k.black_count) * 0x94d049bb133111ebULL;
        for (int i = 0; i < k.white_count; ++i) {
            h ^= uint64_t(k.white[i] + 1) * 0x9ddfea08eb382d69ULL;
            h ^= h >> 29;
        }
        for (int i = 0; i < k.black_count; ++i) {
            h ^= uint64_t(k.black[i] + 101) * 0xc2b2ae3d27d4eb4fULL;
            h ^= h >> 31;
        }
        return static_cast<std::size_t>(h);
    }
};

struct LocalMixedSolver {
    static constexpr int MAX_EMPTY = 16;
    static constexpr int MAX_AMAZONS_TOTAL = 4;
    static constexpr int NODE_BUDGET = 250000;

    Bitboard128 component;
    Bitboard128 outside_occ;
    std::unordered_map<LocalMixedSolverKey, int8_t, LocalMixedSolverKeyHash> memo;
    int nodes = 0;
    bool aborted = false;

    LocalMixedSolverKey make_key(Bitboard128 local_occ,
                                 const Square* white,
                                 int white_count,
                                 const Square* black,
                                 int black_count,
                                 Color side) const {
        LocalMixedSolverKey key;
        key.occ_lo = local_occ.lo;
        key.occ_hi = local_occ.hi;
        key.side = static_cast<uint8_t>(side);
        const int wc = std::min(white_count, NUM_AMAZONS);
        const int bc = std::min(black_count, NUM_AMAZONS);
        key.white_count = static_cast<uint8_t>(wc);
        key.black_count = static_cast<uint8_t>(bc);
        for (int i = 0; i < wc; ++i)
            key.white[i] = static_cast<uint8_t>(white[i]);
        for (int i = 0; i < bc; ++i)
            key.black[i] = static_cast<uint8_t>(black[i]);
        std::sort(key.white, key.white + wc);
        std::sort(key.black, key.black + bc);
        return key;
    }

    int solve(Bitboard128 local_occ,
              Square* white,
              int white_count,
              Square* black,
              int black_count,
              Color side,
              int empty_left) {
        if (empty_left <= 0)
            return 0;
        if (++nodes > NODE_BUDGET) {
            aborted = true;
            return 0;
        }

        const LocalMixedSolverKey key = make_key(local_occ, white, white_count, black, black_count, side);
        auto it = memo.find(key);
        if (it != memo.end())
            return it->second;

        const bool white_to_move = side == WHITE;
        Square* us = white_to_move ? white : black;
        const int us_count = white_to_move ? white_count : black_count;
        const Bitboard128 occ = outside_occ | local_occ;

        bool found = false;
        int best = white_to_move ? -127 : 127;

        for (int i = 0; i < us_count; ++i) {
            const Square from = us[i];
            Bitboard128 occ_lifted = occ;
            occ_lifted.clear(from);

            Bitboard128 tos = get_queen_attacks(from, occ_lifted) & ~occ_lifted & component;
            while (tos) {
                const Square to = tos.pop_lsb();
                Bitboard128 occ_after_to = occ_lifted;
                occ_after_to.set(to);

                Bitboard128 arrows = get_queen_attacks(to, occ_after_to) & ~occ_after_to & component;
                while (arrows) {
                    const Square arrow = arrows.pop_lsb();
                    found = true;

                    Bitboard128 next_local_occ = local_occ;
                    next_local_occ.clear(from);
                    next_local_occ.set(to);
                    next_local_occ.set(arrow);

                    Square next_white[NUM_AMAZONS];
                    Square next_black[NUM_AMAZONS];
                    for (int j = 0; j < white_count; ++j)
                        next_white[j] = white[j];
                    for (int j = 0; j < black_count; ++j)
                        next_black[j] = black[j];
                    if (white_to_move)
                        next_white[i] = to;
                    else
                        next_black[i] = to;

                    const int child = solve(next_local_occ,
                                            next_white,
                                            white_count,
                                            next_black,
                                            black_count,
                                            flip(side),
                                            empty_left - 1);
                    const int value = child + (white_to_move ? 1 : -1);
                    best = white_to_move ? std::max(best, value)
                                         : std::min(best, value);
                    if (aborted)
                        return found ? best : 0;
                }
            }
        }

        if (!found)
            best = 0;

        memo[key] = static_cast<int8_t>(std::clamp(best, -127, 127));
        return best;
    }
};

struct CgtSumKey {
    uint64_t occ_lo = 0;
    uint64_t occ_hi = 0;
    uint8_t side = WHITE;
    uint8_t white_count = 0;
    uint8_t black_count = 0;
    uint8_t white_reserve = 0;
    uint8_t black_reserve = 0;
    uint8_t white[NUM_AMAZONS] = {};
    uint8_t black[NUM_AMAZONS] = {};

    bool operator==(const CgtSumKey& other) const {
        if (occ_lo != other.occ_lo || occ_hi != other.occ_hi
            || side != other.side
            || white_count != other.white_count
            || black_count != other.black_count
            || white_reserve != other.white_reserve
            || black_reserve != other.black_reserve)
            return false;
        for (int i = 0; i < white_count; ++i)
            if (white[i] != other.white[i])
                return false;
        for (int i = 0; i < black_count; ++i)
            if (black[i] != other.black[i])
                return false;
        return true;
    }
};

struct CgtSumKeyHash {
    std::size_t operator()(const CgtSumKey& k) const {
        uint64_t h = k.occ_lo ^ (k.occ_hi + 0x9e3779b97f4a7c15ULL + (k.occ_lo << 6) + (k.occ_lo >> 2));
        h ^= uint64_t(k.side + 13 * k.white_reserve + 29 * k.black_reserve) * 0xbf58476d1ce4e5b9ULL;
        h ^= uint64_t(k.white_count + 11 * k.black_count) * 0x94d049bb133111ebULL;
        for (int i = 0; i < k.white_count; ++i) {
            h ^= uint64_t(k.white[i] + 1) * 0x9ddfea08eb382d69ULL;
            h ^= h >> 29;
        }
        for (int i = 0; i < k.black_count; ++i) {
            h ^= uint64_t(k.black[i] + 101) * 0xc2b2ae3d27d4eb4fULL;
            h ^= h >> 31;
        }
        return static_cast<std::size_t>(h);
    }
};

struct CgtSumSolver {
    static constexpr int MAX_MIXED_EMPTY = 20;
    static constexpr int MAX_MIXED_AMAZONS = 6;
    static constexpr int MAX_RESERVE = 48;
    static constexpr int NODE_BUDGET = 650000;

    Bitboard128 mixed_union;
    Bitboard128 outside_occ;
    std::unordered_map<CgtSumKey, int8_t, CgtSumKeyHash> memo;
    int nodes = 0;
    bool aborted = false;

    CgtSumKey make_key(Bitboard128 local_occ,
                       const Square* white,
                       int white_count,
                       const Square* black,
                       int black_count,
                       Color side,
                       int white_reserve,
                       int black_reserve) const {
        CgtSumKey key;
        key.occ_lo = local_occ.lo;
        key.occ_hi = local_occ.hi;
        key.side = static_cast<uint8_t>(side);
        const int wc = std::min(white_count, NUM_AMAZONS);
        const int bc = std::min(black_count, NUM_AMAZONS);
        key.white_count = static_cast<uint8_t>(wc);
        key.black_count = static_cast<uint8_t>(bc);
        key.white_reserve = static_cast<uint8_t>(std::clamp(white_reserve, 0, 255));
        key.black_reserve = static_cast<uint8_t>(std::clamp(black_reserve, 0, 255));
        for (int i = 0; i < wc; ++i)
            key.white[i] = static_cast<uint8_t>(white[i]);
        for (int i = 0; i < bc; ++i)
            key.black[i] = static_cast<uint8_t>(black[i]);
        std::sort(key.white, key.white + wc);
        std::sort(key.black, key.black + bc);
        return key;
    }

    int solve(Bitboard128 local_occ,
              Square* white,
              int white_count,
              Square* black,
              int black_count,
              Color side,
              int white_reserve,
              int black_reserve) {
        if (++nodes > NODE_BUDGET) {
            aborted = true;
            return 0;
        }

        const CgtSumKey key = make_key(local_occ,
                                       white,
                                       white_count,
                                       black,
                                       black_count,
                                       side,
                                       white_reserve,
                                       black_reserve);
        auto it = memo.find(key);
        if (it != memo.end())
            return it->second;

        const bool white_to_move = side == WHITE;
        Square* us = white_to_move ? white : black;
        const int us_count = white_to_move ? white_count : black_count;
        const Bitboard128 occ = outside_occ | local_occ;

        bool found = false;
        int best = white_to_move ? -127 : 127;

        if (white_to_move && white_reserve > 0) {
            found = true;
            best = std::max(best, 1 + solve(local_occ,
                                            white,
                                            white_count,
                                            black,
                                            black_count,
                                            BLACK,
                                            white_reserve - 1,
                                            black_reserve));
        } else if (!white_to_move && black_reserve > 0) {
            found = true;
            best = std::min(best, -1 + solve(local_occ,
                                             white,
                                             white_count,
                                             black,
                                             black_count,
                                             WHITE,
                                             white_reserve,
                                             black_reserve - 1));
        }
        if (aborted)
            return 0;

        for (int i = 0; i < us_count; ++i) {
            const Square from = us[i];
            Bitboard128 occ_lifted = occ;
            occ_lifted.clear(from);

            Bitboard128 tos = get_queen_attacks(from, occ_lifted) & ~occ_lifted & mixed_union;
            while (tos) {
                const Square to = tos.pop_lsb();
                Bitboard128 occ_after_to = occ_lifted;
                occ_after_to.set(to);

                Bitboard128 arrows = get_queen_attacks(to, occ_after_to) & ~occ_after_to & mixed_union;
                while (arrows) {
                    const Square arrow = arrows.pop_lsb();
                    found = true;

                    Bitboard128 next_local_occ = local_occ;
                    next_local_occ.clear(from);
                    next_local_occ.set(to);
                    next_local_occ.set(arrow);

                    Square next_white[NUM_AMAZONS];
                    Square next_black[NUM_AMAZONS];
                    for (int j = 0; j < white_count; ++j)
                        next_white[j] = white[j];
                    for (int j = 0; j < black_count; ++j)
                        next_black[j] = black[j];
                    if (white_to_move)
                        next_white[i] = to;
                    else
                        next_black[i] = to;

                    const int child = solve(next_local_occ,
                                            next_white,
                                            white_count,
                                            next_black,
                                            black_count,
                                            flip(side),
                                            white_reserve,
                                            black_reserve);
                    const int value = child + (white_to_move ? 1 : -1);
                    best = white_to_move ? std::max(best, value)
                                         : std::min(best, value);
                    if (aborted)
                        return 0;
                }
            }
        }

        if (!found)
            best = 0;
        memo[key] = static_cast<int8_t>(std::clamp(best, -127, 127));
        return best;
    }
};

static int exact_single_side_component_value(const Position& pos,
                                             Bitboard128 component,
                                             Color side,
                                             int amazon_count,
                                             int empty_count) {
    if (amazon_count <= 0 || empty_count < 0 || empty_count > LocalFillSolver::MAX_EMPTY)
        return -1;
    if (amazon_count > NUM_AMAZONS)
        return -1;

    Square amazons[NUM_AMAZONS];
    int count = 0;
    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square sq = pos.amazons[side][i];
        if (component.test(sq))
            amazons[count++] = sq;
    }
    if (count != amazon_count)
        return -1;

    LocalFillSolver solver;
    solver.component = component;
    solver.outside_occ = pos.bb_occupied & ~component;
    solver.memo.reserve(8192);

    const int exact = solver.solve(pos.bb_occupied & component, amazons, count, empty_count);
    return solver.aborted ? -1 : exact;
}

static int exact_mixed_component_value(const Position& pos,
                                       Bitboard128 component,
                                       int white_count,
                                       int black_count,
                                       int empty_count) {
    if (white_count <= 0 || black_count <= 0)
        return MIXED_SOLVER_UNKNOWN;
    if (empty_count < 0 || empty_count > LocalMixedSolver::MAX_EMPTY)
        return MIXED_SOLVER_UNKNOWN;
    if (white_count + black_count > LocalMixedSolver::MAX_AMAZONS_TOTAL)
        return MIXED_SOLVER_UNKNOWN;

    Square white[NUM_AMAZONS];
    Square black[NUM_AMAZONS];
    int wc = 0;
    int bc = 0;

    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square wsq = pos.amazons[WHITE][i];
        if (component.test(wsq))
            white[wc++] = wsq;
        const Square bsq = pos.amazons[BLACK][i];
        if (component.test(bsq))
            black[bc++] = bsq;
    }

    if (wc != white_count || bc != black_count)
        return MIXED_SOLVER_UNKNOWN;

    LocalMixedSolver solver;
    solver.component = component;
    solver.outside_occ = pos.bb_occupied & ~component;
    solver.memo.reserve(32768);

    const int exact = solver.solve(pos.bb_occupied & component,
                                   white,
                                   white_count,
                                   black,
                                   black_count,
                                   pos.side_to_move,
                                   empty_count);
    return solver.aborted ? MIXED_SOLVER_UNKNOWN : exact;
}

static bool partition_cgt_sum_value(const Position& pos, int& diff_out) {
    const Bitboard128 passable = ~pos.bb_arrow & BOARD_MASK_BB;
    const Bitboard128 empty = ~pos.bb_occupied & BOARD_MASK_BB;
    Bitboard128 visited = Bitboard128::zero();
    Bitboard128 remaining = passable;

    Bitboard128 mixed_union = Bitboard128::zero();
    int mixed_empty = 0;
    int mixed_amazons = 0;
    int white_reserve = 0;
    int black_reserve = 0;

    while (remaining) {
        const Square start = remaining.lsb();
        const Bitboard128 component = bb_flood_fill(SQUARE_BB[start], passable & ~visited);
        visited |= component;
        remaining &= ~component;

        const int white_count = (component & pos.bb_white).popcount();
        const int black_count = (component & pos.bb_black).popcount();
        const int empty_count = (component & empty).popcount();

        if (white_count > 0 && black_count > 0) {
            if (empty_count > LocalMixedSolver::MAX_EMPTY)
                return false;
            mixed_empty += empty_count;
            mixed_amazons += white_count + black_count;
            if (mixed_empty > CgtSumSolver::MAX_MIXED_EMPTY
                || mixed_amazons > CgtSumSolver::MAX_MIXED_AMAZONS)
                return false;
            mixed_union |= component;
        } else if (white_count > 0) {
            const int exact = exact_single_side_component_value(pos, component, WHITE, white_count, empty_count);
            if (exact < 0)
                return false;
            white_reserve += exact;
        } else if (black_count > 0) {
            const int exact = exact_single_side_component_value(pos, component, BLACK, black_count, empty_count);
            if (exact < 0)
                return false;
            black_reserve += exact;
        }
    }

    if (white_reserve > CgtSumSolver::MAX_RESERVE
        || black_reserve > CgtSumSolver::MAX_RESERVE)
        return false;

    Square white[NUM_AMAZONS];
    Square black[NUM_AMAZONS];
    int wc = 0;
    int bc = 0;
    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square wsq = pos.amazons[WHITE][i];
        if (mixed_union.test(wsq))
            white[wc++] = wsq;
        const Square bsq = pos.amazons[BLACK][i];
        if (mixed_union.test(bsq))
            black[bc++] = bsq;
    }

    CgtSumSolver solver;
    solver.mixed_union = mixed_union;
    solver.outside_occ = pos.bb_occupied & ~mixed_union;
    solver.memo.reserve(65536);

    const int exact = solver.solve(pos.bb_occupied & mixed_union,
                                   white,
                                   wc,
                                   black,
                                   bc,
                                   pos.side_to_move,
                                   white_reserve,
                                   black_reserve);
    if (solver.aborted)
        return false;

    diff_out = exact;
    return true;
}

static int component_fill_value(const Position& pos,
                                Bitboard128 component,
                                Color side,
                                int amazon_count,
                                int empty_count) {
    const int exact = exact_single_side_component_value(pos, component, side, amazon_count, empty_count);
    if (exact >= 0)
        return exact;
    return simple_component_value(empty_count, amazon_count);
}

static Score eval_partition2(const Position& pos, int* active_areas_out = nullptr) {
    int cgt_diff = 0;
    if (partition_cgt_sum_value(pos, cgt_diff)) {
        if (active_areas_out)
            *active_areas_out = 0;
        return cgt_diff;
    }

    const Bitboard128 passable = ~pos.bb_arrow & BOARD_MASK_BB;
    const Bitboard128 empty = ~pos.bb_occupied & BOARD_MASK_BB;
    Bitboard128 visited = Bitboard128::zero();
    Bitboard128 remaining = passable;

    int simple_diff = 0;
    int active_areas = 0;
    int active_empty = 0;

    while (remaining) {
        const Square start = remaining.lsb();
        const Bitboard128 component = bb_flood_fill(SQUARE_BB[start], passable & ~visited);
        visited |= component;
        remaining &= ~component;

        const int white_count = (component & pos.bb_white).popcount();
        const int black_count = (component & pos.bb_black).popcount();
        const int empty_count = (component & empty).popcount();

        if (white_count > 0 && black_count > 0) {
            const int exact = exact_mixed_component_value(pos, component, white_count, black_count, empty_count);
            if (exact != MIXED_SOLVER_UNKNOWN) {
                simple_diff += exact;
            } else {
                ++active_areas;
                active_empty += empty_count;
            }
        } else if (white_count > 0) {
            simple_diff += component_fill_value(pos, component, WHITE, white_count, empty_count);
        } else if (black_count > 0) {
            simple_diff -= component_fill_value(pos, component, BLACK, black_count, empty_count);
        }
    }

    if (active_areas_out)
        *active_areas_out = active_areas;

    Score score = simple_diff;
    if (active_areas == 0 && simple_diff != 0)
        score += simple_diff > 0 ? 30 : -30;
    else if (active_areas > 1)
        score += (simple_diff > 0 ? 1 : simple_diff < 0 ? -1 : 0) * std::min(12, active_empty / 4);

    return score;
}

static bool partition_fast_track_impl(const Position& pos, Score& out) {
    int cgt_diff = 0;
    if (partition_cgt_sum_value(pos, cgt_diff)) {
        const Score white_score = clamp_static_eval(cgt_diff * 40);
        out = pos.side_to_move == WHITE ? white_score : -white_score;
        return true;
    }

    const Bitboard128 passable = ~pos.bb_arrow & BOARD_MASK_BB;
    const Bitboard128 empty = ~pos.bb_occupied & BOARD_MASK_BB;
    Bitboard128 visited = Bitboard128::zero();
    Bitboard128 remaining = passable;

    int diff = 0;
    while (remaining) {
        const Square start = remaining.lsb();
        const Bitboard128 component = bb_flood_fill(SQUARE_BB[start], passable & ~visited);
        visited |= component;
        remaining &= ~component;

        const int white_count = (component & pos.bb_white).popcount();
        const int black_count = (component & pos.bb_black).popcount();

        const int empty_count = (component & empty).popcount();
        if (white_count > 0 && black_count > 0) {
            const int exact = exact_mixed_component_value(pos, component, white_count, black_count, empty_count);
            if (exact == MIXED_SOLVER_UNKNOWN)
                return false;
            diff += exact;
        } else if (white_count > 0) {
            diff += component_fill_value(pos, component, WHITE, white_count, empty_count);
        } else if (black_count > 0) {
            diff -= component_fill_value(pos, component, BLACK, black_count, empty_count);
        }
    }

    Score white_score = clamp_static_eval(diff * 32);
    out = pos.side_to_move == WHITE ? white_score : -white_score;
    return true;
}

static EvalBreakdown compute_classical2_breakdown(const Position& pos) {
    const DistanceInfo info = compute_distances(pos);
    EvalBreakdown out{};
    out.phase = pos.bb_arrow.popcount();
    out.phase_bucket = fusion_phase_bucket(out.phase);
    out.t1 = eval_t1_queen_territory(pos, info);
    out.t2 = eval_t2_king_territory(pos, info);
    out.p1 = eval_p1_queen_position(pos, info);
    out.p2 = eval_p2_king_position(pos, info);
    out.m = eval_enclosure_m(pos);
    out.partition = eval_partition2(pos, &out.active_areas);
    out.mobility = eval_mobility_fast(pos);

    const EvalWeights w = weights_by_phase(out.phase);
    out.raw_white = clamp_static_eval(w.t1 * out.t1
                                    + w.t2 * out.t2
                                    + w.p1 * out.p1
                                    + w.p2 * out.p2
                                    + w.m * out.m
                                    + w.part * out.partition);
    out.classical = pos.side_to_move == WHITE ? out.raw_white : -out.raw_white;
    return out;
}

static GlobalFeatures make_global_features(const Position& pos, const EvalBreakdown& bd) {
    GlobalFeatures gf{};
    const int side_sign = pos.side_to_move == WHITE ? 1 : -1;
    const int empty_count = (~pos.bb_occupied & BOARD_MASK_BB).popcount();

    gf.v[0]  = static_cast<int8_t>(bd.phase_bucket);
    gf.v[1]  = scaled_i8(bd.phase);
    gf.v[2]  = scaled_i8(empty_count);
    gf.v[3]  = scaled_i8(side_sign * bd.t1);
    gf.v[4]  = scaled_i8(side_sign * bd.t2);
    gf.v[5]  = scaled_i8(side_sign * bd.p1, 4);
    gf.v[6]  = scaled_i8(side_sign * bd.p2, 4);
    gf.v[7]  = scaled_i8(side_sign * bd.m);
    gf.v[8]  = scaled_i8(side_sign * bd.partition);
    gf.v[9]  = scaled_i8(side_sign * bd.mobility);
    gf.v[10] = scaled_i8(bd.classical, 16);
    gf.v[11] = scaled_i8(bd.raw_white * side_sign, 16);
    gf.v[12] = static_cast<int8_t>(std::clamp(bd.active_areas * 16, 0, 127));
    gf.v[13] = bd.active_areas == 0 ? 127 : 0;
    gf.v[14] = static_cast<int8_t>(std::clamp((BOARD_SQ - empty_count) * 2, 0, 127));
    return gf;
}

static EvalInfo compute_eval_info(const Position& pos) {
    EvalInfo info{};
    info.breakdown = compute_classical2_breakdown(pos);
    info.phase_bucket = info.breakdown.phase_bucket;
    info.global = make_global_features(pos, info.breakdown);
    return info;
}

static const EvalInfo& cached_eval_info(const Position& pos) {
    const uint32_t epoch = g_eval_cache_epoch.load(std::memory_order_relaxed);
    EvalCacheEntry& entry = t_eval_cache[pos.key & EVAL_CACHE_MASK];
    if (entry.epoch != epoch || entry.key != pos.key) {
        entry.key = pos.key;
        entry.epoch = epoch;
        entry.info = compute_eval_info(pos);
    } else {
        entry.info.breakdown.classical = pos.side_to_move == WHITE
                                       ? entry.info.breakdown.raw_white
                                       : -entry.info.breakdown.raw_white;
    }
    return entry.info;
}

static const EvalBreakdown& cached_classical2_breakdown(const Position& pos) {
    return cached_eval_info(pos).breakdown;
}

static Score eval_partition_legacy(const Position& pos) {
    Bitboard128 passable = ~pos.bb_arrow & BOARD_MASK_BB;
    Bitboard128 visited = Bitboard128::zero();

    int white_liberties = 0;
    int black_liberties = 0;
    bool is_partitioned = true;

    Bitboard128 remaining = passable;
    while (remaining) {
        int start = remaining.lsb();
        Bitboard128 seed = SQUARE_BB[start];
        Bitboard128 component = bb_flood_fill(seed, passable & ~visited);
        visited |= component;
        remaining &= ~component;

        bool has_white = bool(component & pos.bb_white);
        bool has_black = bool(component & pos.bb_black);
        int empty_count = (component & ~pos.bb_white & ~pos.bb_black).popcount();

        if (has_white && has_black) {
            is_partitioned = false;
        } else if (has_white) {
            white_liberties += empty_count;
        } else if (has_black) {
            black_liberties += empty_count;
        }
    }

    if (!is_partitioned)
        return (white_liberties - black_liberties) * 3;

    int diff = white_liberties - black_liberties;
    if (diff > 0) return 200 + diff * 15;
    if (diff < 0) return -200 + diff * 15;
    return 0;
}

static Score evaluate_classical_legacy_impl(const Position& pos, bool allow_partition_fast_track) {
    if (allow_partition_fast_track && should_try_partition_fast_track(pos)) {
        Score partition_score = SCORE_ZERO;
        if (partition_fast_track_impl(pos, partition_score))
            return partition_score;
    }

    Score raw_eval = g_territory_weight * eval_territory(pos)
                   + g_mobility_weight * eval_mobility(pos);

    raw_eval += eval_partition_legacy(pos);
    raw_eval = clamp_static_eval(raw_eval);
    return pos.side_to_move == WHITE ? raw_eval : -raw_eval;
}

static Score evaluate_classical_legacy(const Position& pos) {
    return evaluate_classical_legacy_impl(pos, true);
}

static int champion_local_mobility(const Position& pos, Color c) {
    const Bitboard128 occ = pos.bb_occupied;
    int score = 0;

    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square from = pos.amazons[c][i];
        Bitboard128 reach = get_queen_attacks(from, occ) & ~occ;
        int ray_count = 0;
        int local = 0;

        while (reach) {
            const Square to = reach.pop_lsb();
            ++ray_count;

            const int d = std::max(std::abs(sq_row(to) - sq_row(from)),
                                   std::abs(sq_col(to) - sq_col(from)));
            if (d <= 3) {
                const int empty_neighbors = (NEIGHBOR_BB[to] & ~occ & BOARD_MASK_BB).popcount();
                local += empty_neighbors * (d == 1 ? 8 : d == 2 ? 4 : 2);
            }
        }

        score += local + ray_count * 2;
        if (ray_count == 0)
            score -= 220;
        else if (ray_count <= 2)
            score -= 90;
        else if (ray_count <= 4)
            score -= 35;
    }

    return score;
}

static Score champion_eval_white(const Position& pos) {
    const DistanceInfo info = compute_distances(pos);

    int t1 = 0;
    int t2 = 0;
    for (Square sq = 0; sq < BOARD_SQ; ++sq) {
        if (pos.board[sq] != EMPTY)
            continue;

        const int qw = info.qdist[WHITE][sq];
        const int qb = info.qdist[BLACK][sq];
        if (qw != qb)
            t1 += qw < qb ? 1 : -1;

        const int kw = info.kdist[WHITE][sq];
        const int kb = info.kdist[BLACK][sq];
        if (kw != kb)
            t2 += kw < kb ? 1 : -1;
    }

    const int mobility = champion_local_mobility(pos, WHITE)
                       - champion_local_mobility(pos, BLACK);
    const int arrows = pos.bb_arrow.popcount();

    // Queen territory dominates early. King-distance territory and local
    // fill mobility matter more as arrows make regions harder to change.
    const int t1_weight = arrows < 20 ? 16 : arrows < 48 ? 13 : 8;
    const int t2_weight = arrows < 20 ? 3  : arrows < 48 ? 6  : 10;
    const int m_weight  = arrows < 20 ? 2  : arrows < 56 ? 3  : 5;

    return clamp_static_eval(t1 * t1_weight + t2 * t2_weight + mobility * m_weight / 8);
}

static int mobility_from(Square sq, Bitboard128 occ) {
    return (get_queen_attacks(sq, occ) & ~occ).popcount();
}

static int empty_neighbor_count(Square sq, Bitboard128 occ) {
    return (NEIGHBOR_BB[sq] & ~occ & BOARD_MASK_BB).popcount();
}

static int champion_move_prior_for_mover(const Position& pos, Move m) {
    const Color us = pos.side_to_move;
    const Color them = flip(us);
    const Square from = move_from(m);
    const Square to = move_to(m);
    const Square arrow = move_arrow(m);

    Bitboard128 occ_lifted = pos.bb_occupied;
    occ_lifted.clear(from);

    Bitboard128 occ_after_move = occ_lifted;
    occ_after_move.set(to);

    Bitboard128 occ_after_arrow = occ_after_move;
    occ_after_arrow.set(arrow);

    int score = 0;

    const int from_mob = mobility_from(from, pos.bb_occupied);
    const int to_mob_before = mobility_from(to, occ_after_move);
    const int to_mob_after = mobility_from(to, occ_after_arrow);

    score += (to_mob_after - from_mob) * 90;
    if (from_mob <= 3 && to_mob_before >= from_mob + 3)
        score += 900 + 80 * (to_mob_before - from_mob);

    if (to_mob_after == 0)
        score -= 6000;
    else if (to_mob_after <= 2)
        score -= 2200;
    else if (to_mob_after <= 4)
        score -= 800;

    const Bitboard128 enemy_bb = them == WHITE ? pos.bb_white : pos.bb_black;
    if (NEIGHBOR_BB[arrow] & enemy_bb)
        score += 1400;
    if (NEIGHBOR_BB[to] & enemy_bb)
        score += 650;

    int block_score = 0;
    for (int i = 0; i < NUM_AMAZONS; ++i) {
        const Square enemy = pos.amazons[them][i];
        Bitboard128 before = get_queen_attacks(enemy, occ_after_move) & ~occ_after_move;
        if (!before.test(arrow))
            continue;

        const int before_mob = before.popcount();
        const int after_mob = mobility_from(enemy, occ_after_arrow);
        const int loss = std::max(0, before_mob - after_mob);
        block_score += 700 + loss * 130;
        if (after_mob <= 2)
            block_score += 900;
    }
    score += block_score;

    const int arrow_neighbors = empty_neighbor_count(arrow, occ_after_move);
    if (arrow_neighbors <= 2)
        score += 420;
    else if (arrow_neighbors <= 4)
        score += 160;

    const int center_dr = std::abs(2 * sq_row(arrow) - 9);
    const int center_dc = std::abs(2 * sq_col(arrow) - 9);
    score += 80 - 5 * (center_dr + center_dc);

    const Bitboard128 own_after = ((us == WHITE ? pos.bb_white : pos.bb_black) & ~SQUARE_BB[from]) | SQUARE_BB[to];
    if ((NEIGHBOR_BB[arrow] & own_after) && !(NEIGHBOR_BB[arrow] & enemy_bb))
        score -= 220;

    return score;
}

} // namespace

void clear_eval_cache() {
    uint32_t next = g_eval_cache_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    if (next == 0) {
        g_eval_cache_epoch.store(1, std::memory_order_relaxed);
        for (auto& entry : t_eval_cache)
            entry.epoch = 0;
    }
}

void set_territory_weight(int value) {
    g_territory_weight = value;
}

void set_mobility_weight(int value) {
    g_mobility_weight = value;
}

Score eval_territory(const Position& pos) {
    Bitboard128 empty = ~pos.bb_occupied & BOARD_MASK_BB;

    Bitboard128 w_front = pos.bb_white;
    Bitboard128 b_front = pos.bb_black;

    int white_owned = 0;
    int black_owned = 0;

    while (bool(empty) && (bool(w_front) || bool(b_front))) {
        w_front = king_attacks_all(w_front) & empty;
        b_front = king_attacks_all(b_front) & empty;

        Bitboard128 ties = w_front & b_front;
        w_front &= ~ties;
        b_front &= ~ties;

        empty &= ~(w_front | b_front | ties);

        white_owned += w_front.popcount();
        black_owned += b_front.popcount();
    }

    return white_owned - black_owned;
}

Score eval_mobility(const Position& pos) {
    return eval_mobility_fast(pos);
}

Score champion_eval_fast(const Position& pos, Color perspective) {
    const Score white_score = champion_eval_white(pos);
    return perspective == WHITE ? white_score : -white_score;
}

int champion_move_prior(const Position& pos, Move m, Color perspective) {
    if (m == MOVE_NONE)
        return 0;

    const int mover_score = champion_move_prior_for_mover(pos, m);
    return pos.side_to_move == perspective ? mover_score : -mover_score;
}

Score evaluate_classical2(const Position& pos) {
    return cached_classical2_breakdown(pos).classical;
}

Score evaluate_classical(const Position& pos) {
    return evaluate_with_tier(pos, g_use_strong_classical ? g_eval_tier : EVAL_TIER_FAST);
}

bool get_eval_breakdown(const Position& pos, EvalBreakdown& out) {
    out = cached_classical2_breakdown(pos);
    return true;
}

bool get_eval_info(const Position& pos, EvalInfo& out) {
    out = cached_eval_info(pos);
    return true;
}

bool partition_fast_track(const Position& pos, Score& out) {
    return partition_fast_track_impl(pos, out);
}

Score evaluate_with_tier(const Position& pos, int tier) {
    const bool nnue_active = g_use_nnue && g_nnue_loaded;

    if (!g_use_strong_classical && !nnue_active)
        return evaluate_classical_legacy(pos);

    if (!nnue_active) {
        Score partition_score = SCORE_ZERO;
        if (should_try_partition_fast_track(pos) && partition_fast_track_impl(pos, partition_score))
            return partition_score;

        if (tier <= EVAL_TIER_FAST)
            return evaluate_classical_legacy_impl(pos, false);

        if (tier == EVAL_TIER_MIXED && pos.bb_arrow.popcount() < 56)
            return evaluate_classical_legacy_impl(pos, false);
    }

    const EvalInfo& info = cached_eval_info(pos);
    const Score classical = g_use_strong_classical
                          ? info.breakdown.classical
                          : evaluate_classical_legacy(pos);

    if (nnue_active) {
        Score nnue_eval = NNUE::evaluate_nnue(pos.side_to_move,
                                              pos.history[pos.ply].accumulator,
                                              info.global.v,
                                              info.phase_bucket);
        nnue_eval = clamp_static_eval(nnue_eval);

        if (g_use_pure_nnue)
            return nnue_eval;

        if (g_use_residual_nnue)
            return clamp_static_eval(classical + nnue_eval);
    }

    return classical;
}

Score evaluate_search(const Position& pos, bool is_pv, bool is_root, int depth, Score alpha, Score beta) {
    const bool nnue_active = g_use_nnue && g_nnue_loaded;
    if (!g_use_strong_classical && !nnue_active)
        return evaluate_classical_legacy(pos);

    if (g_eval_tier == EVAL_TIER_STRONG || nnue_active)
        return evaluate_with_tier(pos, EVAL_TIER_STRONG);

    if (g_eval_tier == EVAL_TIER_FAST)
        return evaluate_with_tier(pos, EVAL_TIER_FAST);

    if (is_root || is_pv || depth >= 3 || pos.bb_arrow.popcount() >= 56)
        return evaluate_with_tier(pos, EVAL_TIER_STRONG);

    const Score fast = evaluate_with_tier(pos, EVAL_TIER_FAST);
    if (std::abs(fast - alpha) <= 180 || std::abs(fast - beta) <= 180)
        return evaluate_with_tier(pos, EVAL_TIER_STRONG);

    return fast;
}

Score evaluate(const Position& pos) {
    return evaluate_with_tier(pos, g_use_strong_classical ? g_eval_tier : EVAL_TIER_FAST);
}

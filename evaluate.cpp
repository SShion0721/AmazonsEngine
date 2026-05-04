/*==================================================================
 * AMAZONS ENGINE - evaluate.cpp
 * Static position evaluation.
 *==================================================================*/
#include "evaluate.h"
#include "nnue/nnue.h"
#include "bitboard.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

int g_territory_weight = 10;
int g_mobility_weight = 2;

bool g_nnue_loaded = false;
bool g_use_nnue = false;
bool g_use_residual_nnue = true;
bool g_use_pure_nnue = false;
bool g_use_strong_classical = true;

namespace {

constexpr int DIST_INF = 127;
constexpr int EVAL_CACHE_SIZE = 1 << 20;
constexpr int EVAL_CACHE_MASK = EVAL_CACHE_SIZE - 1;

struct DistanceInfo {
    int qdist[2][BOARD_SQ];
    int kdist[2][BOARD_SQ];
};

struct EvalCacheEntry {
    uint64_t key = 0;
    bool valid = false;
    EvalInfo info{};
};

alignas(64) std::array<EvalCacheEntry, EVAL_CACHE_SIZE> g_eval_cache;

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

static Score eval_partition2(const Position& pos, int* active_areas_out = nullptr) {
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
            ++active_areas;
            active_empty += empty_count;
        } else if (white_count > 0) {
            simple_diff += simple_component_value(empty_count, white_count);
        } else if (black_count > 0) {
            simple_diff -= simple_component_value(empty_count, black_count);
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
    EvalCacheEntry& entry = g_eval_cache[pos.key & EVAL_CACHE_MASK];
    if (!entry.valid || entry.key != pos.key) {
        entry.key = pos.key;
        entry.valid = true;
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

static Score evaluate_classical_legacy(const Position& pos) {
    Score raw_eval = g_territory_weight * eval_territory(pos)
                   + g_mobility_weight * eval_mobility(pos);

    raw_eval += eval_partition_legacy(pos);
    raw_eval = clamp_static_eval(raw_eval);
    return pos.side_to_move == WHITE ? raw_eval : -raw_eval;
}

} // namespace

void clear_eval_cache() {
    for (auto& entry : g_eval_cache)
        entry.valid = false;
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

Score evaluate_classical2(const Position& pos) {
    return cached_classical2_breakdown(pos).classical;
}

Score evaluate_classical(const Position& pos) {
    return g_use_strong_classical ? evaluate_classical2(pos)
                                  : evaluate_classical_legacy(pos);
}

bool get_eval_breakdown(const Position& pos, EvalBreakdown& out) {
    out = cached_classical2_breakdown(pos);
    return true;
}

bool get_eval_info(const Position& pos, EvalInfo& out) {
    out = cached_eval_info(pos);
    return true;
}

Score evaluate(const Position& pos) {
    const EvalInfo& info = cached_eval_info(pos);
    const Score classical = g_use_strong_classical
                          ? info.breakdown.classical
                          : evaluate_classical_legacy(pos);

    if (g_use_nnue && g_nnue_loaded) {
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

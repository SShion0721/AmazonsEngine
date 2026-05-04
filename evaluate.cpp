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
constexpr int EVAL_CACHE_SIZE = 1 << 15;
constexpr int EVAL_CACHE_MASK = EVAL_CACHE_SIZE - 1;

struct DistanceInfo {
    int qdist[2][BOARD_SQ];
    int kdist[2][BOARD_SQ];
};

struct EvalCacheEntry {
    uint64_t key = 0;
    bool valid = false;
    EvalBreakdown breakdown{};
};

std::array<EvalCacheEntry, EVAL_CACHE_SIZE> g_eval_cache;

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
    out.t1 = eval_t1_queen_territory(pos, info);
    out.t2 = eval_t2_king_territory(pos, info);
    out.p1 = eval_p1_queen_position(pos, info);
    out.p2 = eval_p2_king_position(pos, info);
    out.m = eval_enclosure_m(pos);
    out.partition = eval_partition2(pos, &out.active_areas);

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

static const EvalBreakdown& cached_classical2_breakdown(const Position& pos) {
    EvalCacheEntry& entry = g_eval_cache[pos.key & EVAL_CACHE_MASK];
    if (!entry.valid || entry.key != pos.key) {
        entry.key = pos.key;
        entry.valid = true;
        entry.breakdown = compute_classical2_breakdown(pos);
    } else {
        entry.breakdown.classical = pos.side_to_move == WHITE
                                  ? entry.breakdown.raw_white
                                  : -entry.breakdown.raw_white;
    }
    return entry.breakdown;
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
    Bitboard128 occ = pos.bb_occupied;
    int white_mob = 0;
    int black_mob = 0;

    for (int i = 0; i < NUM_AMAZONS; i++) {
        Square wsq = pos.amazons[WHITE][i];
        Bitboard128 w_atk = get_queen_attacks(wsq, occ);
        white_mob += (w_atk & ~occ).popcount();

        Square bsq = pos.amazons[BLACK][i];
        Bitboard128 b_atk = get_queen_attacks(bsq, occ);
        black_mob += (b_atk & ~occ).popcount();
    }

    return white_mob - black_mob;
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

Score evaluate(const Position& pos) {
    const Score classical = evaluate_classical(pos);

    if (g_use_nnue && g_nnue_loaded) {
        Score nnue_eval = NNUE::evaluate_nnue(pos.side_to_move, pos.history[pos.ply].accumulator);
        nnue_eval = clamp_static_eval(nnue_eval);

        if (g_use_pure_nnue)
            return nnue_eval;

        if (g_use_residual_nnue)
            return clamp_static_eval(classical + nnue_eval);
    }

    return classical;
}

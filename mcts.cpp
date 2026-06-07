/*==================================================================
 * AMAZONS ENGINE - mcts.cpp
 * Champion-style root PUCT/MCTS with progressive widening, shared
 * atomic tree parallelism, virtual loss, tree reuse, and heuristic
 * rollouts. All moves come from the legal PEXT generator.
 *==================================================================*/
#include "mcts.h"

#include "evaluate.h"
#include "movegen.h"
#include "policy_prior.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int MAX_ROLLOUT_PLIES = 12;
constexpr int64_t VALUE_SCALE = 1000000;

struct XorShift64 {
    uint64_t s;

    explicit XorShift64(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

    uint64_t next() {
        uint64_t x = s;
        x ^= x << 7;
        x ^= x >> 9;
        s = x;
        return x;
    }

    int bounded(int n) {
        return n <= 1 ? 0 : int(next() % uint64_t(n));
    }
};

struct Candidate {
    Move move = MOVE_NONE;
    int prior_root = 0;
    double prior_prob = 0.0;
};

struct Node {
    explicit Node(Move m = MOVE_NONE,
                  int p = -1,
                  uint64_t k = 0,
                  Color stm = WHITE,
                  int prior = 0,
                  double prob = 0.0)
        : move(m),
          parent(p),
          key(k),
          side_to_move(stm),
          prior_root(prior),
          prior_prob(prob) {}

    Move move = MOVE_NONE;
    int parent = -1;
    uint64_t key = 0;
    Color side_to_move = WHITE;
    std::atomic<int> visits{0};
    std::atomic<int> virtual_loss{0};
    std::atomic<int64_t> value_sum{0}; // root POV, scaled by VALUE_SCALE
    int prior_root = 0;
    double prior_prob = 0.0;
    bool generated = false; // protected by mutex
    std::vector<Candidate> candidates;
    std::vector<int> children;
    mutable std::mutex mutex;
};

struct SharedTree {
    std::mutex pool_mutex;
    std::vector<std::unique_ptr<Node>> nodes;
};

struct ReuseTree {
    std::mutex mutex;
    uint64_t root_key = 0;
    int root_index = 0;
    std::shared_ptr<SharedTree> tree;
};

struct ScoredMoveSmall {
    Move move = MOVE_NONE;
    int score = 0;
};

ReuseTree g_reuse_tree;

Node& node_at(const std::shared_ptr<SharedTree>& tree, int idx) {
    return *tree->nodes[idx];
}

int node_count(const std::shared_ptr<SharedTree>& tree) {
    std::lock_guard<std::mutex> lock(tree->pool_mutex);
    return static_cast<int>(tree->nodes.size());
}

int add_node(const std::shared_ptr<SharedTree>& tree,
             Move move,
             int parent,
             uint64_t key,
             Color stm,
             int prior,
             double prior_prob) {
    std::lock_guard<std::mutex> lock(tree->pool_mutex);
    const int idx = static_cast<int>(tree->nodes.size());
    tree->nodes.push_back(std::make_unique<Node>(move, parent, key, stm, prior, prior_prob));
    return idx;
}

int progressive_limit(const Node& node, bool is_root, const MctsConfig& cfg) {
    const int hard = is_root ? cfg.root_max_candidates : cfg.node_max_candidates;
    const int initial = is_root ? cfg.root_initial_width : cfg.node_initial_width;
    const int visits = node.visits.load(std::memory_order_relaxed)
                     + node.virtual_loss.load(std::memory_order_relaxed);
    const int limit = initial + (visits / std::max(1, cfg.width_visit_step)) * cfg.root_width_step;
    return std::min(hard, std::min<int>(node.candidates.size(), std::max(initial, limit)));
}

double prior_bias(int prior_root, Color parent_side, Color root_side) {
    double p = std::clamp(prior_root / 12000.0, -0.24, 0.24);
    return parent_side == root_side ? p : -p;
}

int cheap_move_prior(const Position& p, Move m, Color perspective) {
    const Color us = p.side_to_move;
    const Color them = flip(us);
    const Square from = move_from(m);
    const Square to = move_to(m);
    const Square arrow = move_arrow(m);

    Bitboard128 occ = p.bb_occupied;
    const int from_mob = (get_queen_attacks(from, occ) & ~occ).popcount();
    occ.clear(from);
    occ.set(to);
    Bitboard128 occ_arrow = occ;
    occ_arrow.set(arrow);
    const int to_mob = (get_queen_attacks(to, occ_arrow) & ~occ_arrow).popcount();

    const Bitboard128 enemy = them == WHITE ? p.bb_white : p.bb_black;
    int score = (to_mob - from_mob) * 80;
    if (NEIGHBOR_BB[arrow] & enemy)
        score += 1300;
    if (to_mob <= 2)
        score -= 1600;
    const int center = std::abs(2 * sq_row(arrow) - 9) + std::abs(2 * sq_col(arrow) - 9);
    score += 70 - 4 * center;
    return us == perspective ? score : -score;
}

bool ensure_candidates_locked(Node& node,
                              Position& pos,
                              Color root_side,
                              bool is_root,
                              const MctsConfig& cfg,
                              const PolicyPrior* root_policy) {
    if (node.generated)
        return !node.candidates.empty();

    node.generated = true;
    std::array<Move, MAX_LEGAL_MOVES> raw{};
    const int raw_count = generate_moves(pos, raw.data(), MAX_LEGAL_MOVES);
    if (raw_count <= 0)
        return false;

    node.candidates.reserve(std::min(raw_count, is_root ? cfg.root_max_candidates : cfg.node_max_candidates));
    for (int i = 0; i < raw_count; ++i) {
        // Use cheap_move_prior for speed - champion_move_prior is too expensive at low time controls
        int prior = cheap_move_prior(pos, raw[i], root_side);
        if (is_root && root_policy && cfg.use_policy_prior && policy_prior_loaded()) {
            const int policy = policy_move_prior(*root_policy, raw[i]);
            const int blend = std::clamp(cfg.policy_blend, 0, 100);
            prior = (blend * policy + (100 - blend) * prior) / 100;
        }
        node.candidates.push_back({raw[i], prior, 0.0});
    }

    const bool root_to_move = pos.side_to_move == root_side;
    std::sort(node.candidates.begin(), node.candidates.end(),
              [root_to_move](const Candidate& a, const Candidate& b) {
                  return root_to_move ? a.prior_root > b.prior_root
                                      : a.prior_root < b.prior_root;
              });

    const int cap = is_root ? cfg.root_max_candidates : cfg.node_max_candidates;
    if (static_cast<int>(node.candidates.size()) > cap)
        node.candidates.resize(cap);

    if (!node.candidates.empty()) {
        const auto preference = [root_to_move](const Candidate& c) {
            return root_to_move ? c.prior_root : -c.prior_root;
        };
        int best_prior = preference(node.candidates.front());
        for (const Candidate& c : node.candidates)
            best_prior = std::max(best_prior, preference(c));

        double total = 0.0;
        for (Candidate& c : node.candidates) {
            c.prior_prob = std::exp(std::clamp((preference(c) - best_prior) / 900.0, -20.0, 0.0));
            total += c.prior_prob;
        }
        if (total > 0.0) {
            for (Candidate& c : node.candidates)
                c.prior_prob /= total;
        } else {
            const double uniform = 1.0 / double(node.candidates.size());
            for (Candidate& c : node.candidates)
                c.prior_prob = uniform;
        }
    }
    return !node.candidates.empty();
}

void insert_rollout_candidate(ScoredMoveSmall* top, int& count, int cap, ScoredMoveSmall value) {
    if (cap <= 0)
        return;
    if (count == cap && value.score <= top[count - 1].score)
        return;

    int pos = std::min(count, cap - 1);
    if (count < cap)
        ++count;
    while (pos > 0 && top[pos - 1].score < value.score) {
        top[pos] = top[pos - 1];
        --pos;
    }
    top[pos] = value;
}

double terminal_value(const Position& pos, Color root_side) {
    return pos.side_to_move == root_side ? -1.0 : 1.0;
}

double rollout(Position& pos, Color root_side, XorShift64& rng, const MctsConfig& cfg) {
    Move stack[MAX_ROLLOUT_PLIES];
    int stack_count = 0;
    const int rollout_plies = std::clamp(cfg.rollout_plies, 1, MAX_ROLLOUT_PLIES);

    for (int ply = 0; ply < rollout_plies; ++ply) {
        std::array<Move, MAX_LEGAL_MOVES> raw{};
        const int raw_count = generate_moves(pos, raw.data(), MAX_LEGAL_MOVES);
        if (raw_count <= 0) {
            const double value = terminal_value(pos, root_side);
            while (stack_count > 0)
                pos.undo_move(stack[--stack_count]);
            return value;
        }

        Move chosen = MOVE_NONE;
        if ((rng.next() & 0xFF) < 38) {
            chosen = raw[rng.bounded(raw_count)];
        } else {
            ScoredMoveSmall top[24];
            int top_count = 0;
            for (int i = 0; i < raw_count; ++i)
                insert_rollout_candidate(top, top_count, 24,
                                         {raw[i], cheap_move_prior(pos, raw[i], pos.side_to_move)});
            chosen = top[rng.bounded(std::min(top_count, 6))].move;
        }

        pos.do_move(chosen);
        stack[stack_count++] = chosen;
    }

    const Score cp = champion_eval_fast(pos, root_side);
    double value = std::tanh(double(cp) / 650.0);
    while (stack_count > 0)
        pos.undo_move(stack[--stack_count]);
    return std::clamp(value, -1.0, 1.0);
}

double node_mean_root_pov(const Node& node) {
    const int visits = node.visits.load(std::memory_order_acquire);
    if (visits <= 0)
        return 0.0;
    const double value = double(node.value_sum.load(std::memory_order_acquire)) / double(VALUE_SCALE);
    return value / double(visits);
}

int select_child(const std::vector<int>& children,
                 const std::shared_ptr<SharedTree>& tree,
                 Color parent_side,
                 Color root_side,
                 const MctsConfig& cfg) {
    const double sign = parent_side == root_side ? 1.0 : -1.0;
    double best_score = -std::numeric_limits<double>::infinity();
    int best = children.front();

    int parent_visits = 1;
    for (int child_idx : children)
        parent_visits += node_at(tree, child_idx).visits.load(std::memory_order_relaxed)
                       + node_at(tree, child_idx).virtual_loss.load(std::memory_order_relaxed);

    for (int child_idx : children) {
        const Node& child = node_at(tree, child_idx);
        const int visits = child.visits.load(std::memory_order_relaxed);
        const int vl = child.virtual_loss.load(std::memory_order_relaxed);
        const int effective_visits = visits + vl;
        const double q = node_mean_root_pov(child);
        const double puct = cfg.cpuct * child.prior_prob
                          * std::sqrt(double(parent_visits))
                          / double(effective_visits + 1);
        const double score = sign * q + puct
                           + prior_bias(child.prior_root, parent_side, root_side)
                           - 0.08 * vl;
        if (score > best_score) {
            best_score = score;
            best = child_idx;
        }
    }

    return best;
}

void add_virtual_loss(Node& node) {
    node.virtual_loss.fetch_add(1, std::memory_order_acq_rel);
}

void finish_node(Node& node, double value) {
    node.virtual_loss.fetch_sub(1, std::memory_order_acq_rel);
    node.value_sum.fetch_add(static_cast<int64_t>(std::llround(value * VALUE_SCALE)),
                             std::memory_order_acq_rel);
    node.visits.fetch_add(1, std::memory_order_acq_rel);
}

bool reused_root_is_safe(const std::shared_ptr<SharedTree>& tree,
                         int root_index,
                         const Position& root,
                         const MctsConfig& cfg) {
    if (!tree || root_index < 0)
        return false;
    {
        std::lock_guard<std::mutex> lock(tree->pool_mutex);
        if (root_index >= static_cast<int>(tree->nodes.size()))
            return false;
    }

    Node& root_node = node_at(tree, root_index);
    if (root_node.key != root.key)
        return false;

    std::lock_guard<std::mutex> lock(root_node.mutex);
    for (int child_idx : root_node.children) {
        if (child_idx < 0 || child_idx >= node_count(tree))
            return false;
        const Move m = node_at(tree, child_idx).move;
        if (m == MOVE_NONE || !is_pseudo_legal(root, m))
            return false;
    }
    if (root_node.generated) {
        std::array<Move, MAX_LEGAL_MOVES> raw{};
        const int raw_count = generate_moves(root, raw.data(), MAX_LEGAL_MOVES);
        const int needed = std::min(raw_count, cfg.root_max_candidates);
        if (static_cast<int>(root_node.candidates.size()) < needed)
            return false;
        for (const Candidate& cand : root_node.candidates) {
            if (cand.move == MOVE_NONE || !is_pseudo_legal(root, cand.move))
                return false;
        }
    }
    return true;
}

std::shared_ptr<SharedTree> make_fresh_tree(const Position& root, int& root_index) {
    auto tree = std::make_shared<SharedTree>();
    tree->nodes.reserve(32768);
    tree->nodes.push_back(std::make_unique<Node>(MOVE_NONE, -1, root.key, root.side_to_move, 0, 1.0));
    root_index = 0;
    return tree;
}

std::shared_ptr<SharedTree> prepare_tree(const Position& root,
                                         const MctsConfig& cfg,
                                         bool allow_reuse,
                                         int& root_index) {
    if (allow_reuse && cfg.use_tree_reuse) {
        std::lock_guard<std::mutex> reuse_lock(g_reuse_tree.mutex);
        if (g_reuse_tree.tree) {
            if (g_reuse_tree.root_key == root.key
                && reused_root_is_safe(g_reuse_tree.tree, g_reuse_tree.root_index, root, cfg)) {
                root_index = g_reuse_tree.root_index;
                return g_reuse_tree.tree;
            }

            const int old_root = g_reuse_tree.root_index;
            if (old_root >= 0 && old_root < node_count(g_reuse_tree.tree)) {
                Node& old = node_at(g_reuse_tree.tree, old_root);
                std::vector<int> children;
                {
                    std::lock_guard<std::mutex> node_lock(old.mutex);
                    children = old.children;
                }
                for (int child_idx : children) {
                    if (child_idx >= 0
                        && child_idx < node_count(g_reuse_tree.tree)
                        && node_at(g_reuse_tree.tree, child_idx).key == root.key
                        && reused_root_is_safe(g_reuse_tree.tree, child_idx, root, cfg)) {
                        root_index = child_idx;
                        return g_reuse_tree.tree;
                    }
                }
            }
        }
    }

    return make_fresh_tree(root, root_index);
}

void worker_loop(const std::shared_ptr<SharedTree>& tree,
                 int root_index,
                 const Position& root,
                 int max_ms,
                 const MctsConfig& cfg,
                 const std::atomic<bool>* stop,
                 const PolicyPrior& root_policy,
                 std::atomic<int>& playouts,
                 int worker_id) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto limit = std::chrono::milliseconds(std::max(1, max_ms));
    const Color root_side = root.side_to_move;
    auto scratch_storage = std::make_unique<Position>(root);
    Position& scratch = *scratch_storage;
    XorShift64 rng(root.key ^ (0x517cc1b727220a95ULL + uint64_t(worker_id) * 0x9e3779b97f4a7c15ULL));

    while (Clock::now() - start < limit
           && (!stop || !stop->load(std::memory_order_relaxed))) {
        int node_idx = root_index;
        int path[160];
        int path_count = 0;
        Move move_stack[160];
        int move_count = 0;
        double value = 0.0;

        while (path_count < 159 && move_count < 159) {
            Node& node = node_at(tree, node_idx);
            path[path_count++] = node_idx;
            add_virtual_loss(node);

            const bool is_root = node_idx == root_index;
            bool terminal = false;
            bool expanded = false;
            int child_idx = -1;
            Move child_move = MOVE_NONE;
            std::vector<int> children;

            {
                std::lock_guard<std::mutex> lock(node.mutex);
                if (!ensure_candidates_locked(node, scratch, root_side, is_root, cfg, is_root ? &root_policy : nullptr)) {
                    terminal = true;
                } else {
                    const int allowed = progressive_limit(node, is_root, cfg);
                    if (static_cast<int>(node.children.size()) < allowed) {
                        const Candidate cand = node.candidates[node.children.size()];
                        child_move = cand.move;
                        scratch.do_move(child_move);
                        move_stack[move_count++] = child_move;

                        child_idx = add_node(tree,
                                             child_move,
                                             node_idx,
                                             scratch.key,
                                             scratch.side_to_move,
                                             cand.prior_root,
                                             cand.prior_prob);
                        node.children.push_back(child_idx);
                        expanded = true;
                    } else {
                        children = node.children;
                    }
                }
            }

            if (terminal) {
                value = terminal_value(scratch, root_side);
                break;
            }

            if (expanded) {
                Node& child = node_at(tree, child_idx);
                path[path_count++] = child_idx;
                add_virtual_loss(child);
                value = rollout(scratch, root_side, rng, cfg);
                break;
            }

            if (children.empty()) {
                value = terminal_value(scratch, root_side);
                break;
            }

            child_idx = select_child(children, tree, node.side_to_move, root_side, cfg);
            child_move = node_at(tree, child_idx).move;
            scratch.do_move(child_move);
            move_stack[move_count++] = child_move;
            node_idx = child_idx;
        }

        for (int i = 0; i < path_count; ++i)
            finish_node(node_at(tree, path[i]), value);

        while (move_count > 0)
            scratch.undo_move(move_stack[--move_count]);

        playouts.fetch_add(1, std::memory_order_relaxed);
    }
}

MctsResult build_result(const std::shared_ptr<SharedTree>& tree,
                        int root_index,
                        const Position& root,
                        const MctsConfig& cfg,
                        const PolicyPrior& root_policy,
                        int playouts) {
    auto scratch_storage = std::make_unique<Position>(root);
    Position& scratch = *scratch_storage;
    Node& root_node = node_at(tree, root_index);
    {
        std::lock_guard<std::mutex> lock(root_node.mutex);
        ensure_candidates_locked(root_node, scratch, root.side_to_move, true, cfg, &root_policy);
    }

    std::vector<int> children;
    {
        std::lock_guard<std::mutex> lock(root_node.mutex);
        children = root_node.children;
    }

    MctsResult result;
    result.playouts = playouts;
    result.root_candidates = static_cast<int>(root_node.candidates.size());
    result.root_children = static_cast<int>(children.size());
    result.root_moves.reserve(children.size());

    for (int child_idx : children) {
        const Node& child = node_at(tree, child_idx);
        const int visits = child.visits.load(std::memory_order_acquire);
        const double mean = node_mean_root_pov(child);
        result.root_moves.push_back({child.move, visits, mean, child.prior_root, child.prior_prob});
    }

    std::sort(result.root_moves.begin(), result.root_moves.end(),
              [](const MctsRootMove& a, const MctsRootMove& b) {
                  if (a.visits != b.visits)
                      return a.visits > b.visits;
                  const double av = a.mean + a.prior / 20000.0 + a.prior_prob;
                  const double bv = b.mean + b.prior / 20000.0 + b.prior_prob;
                  return av > bv;
              });

    if (!result.root_moves.empty()) {
        const MctsRootMove& best = result.root_moves.front();
        result.best_move = best.move;
        result.score = std::clamp<int>(static_cast<int>(best.mean * 850.0 + best.prior / 24),
                                       -SCORE_INF + 1,
                                       SCORE_INF - 1);
        if (!is_pseudo_legal(root, result.best_move)) {
            result.best_move = MOVE_NONE;
            result.score = SCORE_ZERO;
        } else {
            assert(is_pseudo_legal(root, result.best_move));
        }
    } else {
        std::array<Move, MAX_LEGAL_MOVES> moves{};
        const int count = generate_moves(root, moves.data(), MAX_LEGAL_MOVES);
        if (count > 0) {
            result.best_move = moves[0];
            result.score = champion_move_prior(root, moves[0], root.side_to_move) / 24;
        }
    }

    return result;
}

} // namespace

MctsResult run_mcts_root(const Position& root,
                         int max_ms,
                         const MctsConfig& config,
                         const std::atomic<bool>* stop) {
    int root_index = 0;
    const bool allow_reuse = config.use_tree_reuse;
    std::shared_ptr<SharedTree> tree = prepare_tree(root, config, allow_reuse, root_index);
    PolicyPrior root_policy = evaluate_policy_prior(root);
    std::atomic<int> playouts{0};

    const int threads = std::clamp(config.threads, 1, 64);
    if (threads <= 1) {
        worker_loop(tree, root_index, root, max_ms, config, stop, root_policy, playouts, 0);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (int i = 0; i < threads; ++i) {
            workers.emplace_back([&, i] {
                worker_loop(tree, root_index, root, max_ms, config, stop, root_policy, playouts, i);
            });
        }
        for (auto& worker : workers)
            worker.join();
    }

    MctsResult result = build_result(tree, root_index, root, config, root_policy,
                                     playouts.load(std::memory_order_relaxed));

    if (allow_reuse && config.use_tree_reuse) {
        std::lock_guard<std::mutex> lock(g_reuse_tree.mutex);
        if (node_count(tree) <= 500000) {
            g_reuse_tree.root_key = root.key;
            g_reuse_tree.root_index = root_index;
            g_reuse_tree.tree = tree;
        }
    }

    return result;
}

void clear_mcts_tree() {
    std::lock_guard<std::mutex> lock(g_reuse_tree.mutex);
    g_reuse_tree.root_key = 0;
    g_reuse_tree.root_index = 0;
    g_reuse_tree.tree.reset();
}

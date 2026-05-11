#include "../bitboard.h"
#include "../evaluate.h"
#include "../line_pattern.h"
#include "../mcts.h"
#include "../movegen.h"
#include "../policy_prior.h"
#include "../position.h"
#include "../rays.h"
#include "../search.h"
#include "../tt.h"
#include "../nnue/nnue.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    int roots = 80;
    int search_roots = 20;
    int movetime = 20;
    int max_random_plies = 36;
    int paired_games = 0;
    std::vector<int> threads = {1, 2};
    uint32_t seed = 0xC0FFEE;
};

struct PositionSnapshot {
    std::string board;
    Square amazons[2][NUM_AMAZONS];
    Color side = WHITE;
    uint64_t key = 0;
    Bitboard128 white{};
    Bitboard128 black{};
    Bitboard128 arrows{};
    Bitboard128 occupied{};
    int ply = 0;
};

std::vector<int> parse_thread_list(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty())
            continue;
        int value = std::max(1, std::atoi(item.c_str()));
        if (std::find(out.begin(), out.end(), value) == out.end())
            out.push_back(value);
    }
    if (out.empty())
        out.push_back(1);
    return out;
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_int = [&](int fallback) {
            if (i + 1 >= argc)
                return fallback;
            return std::atoi(argv[++i]);
        };

        if (arg == "--roots")
            opt.roots = std::max(1, next_int(opt.roots));
        else if (arg == "--search-roots")
            opt.search_roots = std::max(0, next_int(opt.search_roots));
        else if (arg == "--movetime")
            opt.movetime = std::max(1, next_int(opt.movetime));
        else if (arg == "--max-random-plies")
            opt.max_random_plies = std::max(0, next_int(opt.max_random_plies));
        else if (arg == "--paired-games")
            opt.paired_games = std::max(0, next_int(opt.paired_games));
        else if (arg == "--seed")
            opt.seed = static_cast<uint32_t>(next_int(static_cast<int>(opt.seed)));
        else if (arg == "--threads" && i + 1 < argc)
            opt.threads = parse_thread_list(argv[++i]);
        else if (arg == "--long") {
            opt.roots = 10000;
            opt.search_roots = 1000;
            opt.movetime = 20;
            opt.threads = {1, 2, 4};
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "mcts_safety options:\n"
                      << "  --roots N              direct MCTS random roots (default 80)\n"
                      << "  --search-roots N       Searcher random roots (default 20)\n"
                      << "  --movetime MS          MCTS/Search time per root (default 20)\n"
                      << "  --threads 1,2,4        MCTS thread counts (default 1,2)\n"
                      << "  --paired-games N       optional Match-vs-AB legality smoke\n"
                      << "  --long                 shorthand: --roots 10000 --search-roots 1000 --threads 1,2,4\n";
            std::exit(0);
        }
    }
    return opt;
}

void init_engine_tables() {
    Zobrist::init();
    init_rays();
    init_bitboards();
    init_attack_tables();
    init_between_line_bb();
    init_line_patterns();
    NNUE::init_random_weights();
    NNUE::g_accumulator_enabled = false;
    g_use_nnue = false;
    g_nnue_loaded = false;
    g_use_policy_prior = false;
    clear_eval_cache();
    clear_mcts_tree();
}

PositionSnapshot snapshot(const Position& pos) {
    PositionSnapshot s;
    s.board = pos.get_board_string();
    std::memcpy(s.amazons, pos.amazons, sizeof(s.amazons));
    s.side = pos.side_to_move;
    s.key = pos.key;
    s.white = pos.bb_white;
    s.black = pos.bb_black;
    s.arrows = pos.bb_arrow;
    s.occupied = pos.bb_occupied;
    s.ply = pos.ply;
    return s;
}

bool same_position(const Position& pos, const PositionSnapshot& s, const char* context) {
    if (pos.get_board_string() == s.board
        && std::equal(&pos.amazons[0][0], &pos.amazons[0][0] + 2 * NUM_AMAZONS, &s.amazons[0][0])
        && pos.side_to_move == s.side
        && pos.key == s.key
        && pos.bb_white == s.white
        && pos.bb_black == s.black
        && pos.bb_arrow == s.arrows
        && pos.bb_occupied == s.occupied
        && pos.ply == s.ply)
        return true;

    std::cerr << "Position roundtrip mismatch at " << context << "\n";
    return false;
}

void make_random_position(Position& pos, std::mt19937& rng, int plies) {
    pos.set_startpos();
    std::array<Move, MAX_LEGAL_MOVES> moves{};
    for (int i = 0; i < plies; ++i) {
        const int count = generate_moves(pos, moves.data(), MAX_LEGAL_MOVES);
        if (count == 0)
            return;
        std::uniform_int_distribution<int> dist(0, count - 1);
        pos.do_move(moves[dist(rng)]);
    }
}

bool check_best_move(Position& pos, Move best, const char* context) {
    const bool has_move = has_legal_move(pos);
    if (!has_move) {
        if (best == MOVE_NONE)
            return true;
        std::cerr << "Returned a move in terminal position at " << context
                  << ": " << move_to_str(best) << "\n";
        return false;
    }
    if (best == MOVE_NONE || !is_pseudo_legal(pos, best)) {
        std::cerr << "Illegal or missing best move at " << context
                  << ": " << move_to_str(best) << "\n";
        return false;
    }

    const PositionSnapshot before = snapshot(pos);
    pos.do_move(best);
    pos.undo_move(best);
    return same_position(pos, before, context);
}

bool run_direct_mcts_roots(const Options& opt, std::mt19937& rng, Position& pos) {
    MctsConfig cfg;
    cfg.root_initial_width = 32;
    cfg.root_max_candidates = 96;
    cfg.node_initial_width = 4;
    cfg.node_max_candidates = 48;
    cfg.rollout_plies = 2;
    cfg.use_policy_prior = false;

    for (int root = 0; root < opt.roots; ++root) {
        if (root % std::max(1, opt.roots / 10) == 0)
            std::cout << "mcts_safety: direct root " << root << "/" << opt.roots << "\n" << std::flush;
        const int plies = opt.max_random_plies == 0 ? 0
            : static_cast<int>(rng() % static_cast<uint32_t>(opt.max_random_plies + 1));
        make_random_position(pos, rng, plies);

        for (bool reuse : {false, true}) {
            cfg.use_tree_reuse = reuse;
            if (!reuse)
                clear_mcts_tree();

            for (int threads : opt.threads) {
                cfg.threads = threads;
                MctsResult result = run_mcts_root(pos, opt.movetime, cfg);
                const std::string context = "mcts root=" + std::to_string(root)
                                          + " plies=" + std::to_string(plies)
                                          + " reuse=" + std::to_string(reuse)
                                          + " threads=" + std::to_string(threads);
                if (!check_best_move(pos, result.best_move, context.c_str()))
                    return false;
            }
        }
    }
    return true;
}

bool run_search_roots(const Options& opt, std::mt19937& rng, Position& pos) {
    auto tt = std::make_unique<TranspositionTable>(16);
    auto searcher = std::make_unique<Searcher>(*tt);
    searcher->silent = true;
    g_search_mode = SEARCH_MODE_MATCH;
    g_use_mcts_root = true;
    g_teacher_safe_search = false;
    g_mcts_min_time_ms = 1;
    g_mcts_threads = std::max(1, opt.threads.back());
    g_use_tree_reuse = true;

    for (int root = 0; root < opt.search_roots; ++root) {
        if (root % std::max(1, std::max(1, opt.search_roots) / 10) == 0)
            std::cout << "mcts_safety: search root " << root << "/" << opt.search_roots << "\n" << std::flush;
        const int plies = opt.max_random_plies == 0 ? 0
            : static_cast<int>(rng() % static_cast<uint32_t>(std::min(opt.max_random_plies, 24) + 1));
        make_random_position(pos, rng, plies);
        searcher->time_man.set_movetime(opt.movetime);
        Move best = searcher->search(pos, 8, nullptr, 0);
        const std::string context = "search root=" + std::to_string(root)
                                  + " plies=" + std::to_string(plies);
        if (!check_best_move(pos, best, context.c_str()))
            return false;
    }
    return true;
}

bool run_paired_smoke(const Options& opt) {
    if (opt.paired_games <= 0)
        return true;

    auto tt = std::make_unique<TranspositionTable>(16);
    Searcher searcher(*tt);
    searcher.silent = true;
    Position pos;

    for (int game = 0; game < opt.paired_games; ++game) {
        pos.set_startpos();
        const bool mcts_white = (game % 2) == 0;

        for (int ply = 0; ply < 120; ++ply) {
            if (!has_legal_move(pos))
                break;

            const bool use_mcts_this_move = (pos.side_to_move == WHITE) == mcts_white;
            g_search_mode = use_mcts_this_move ? SEARCH_MODE_MATCH : SEARCH_MODE_FAST_SELFPLAY;
            g_use_mcts_root = use_mcts_this_move;
            g_mcts_min_time_ms = 1;
            g_mcts_threads = std::max(1, opt.threads.back());
            searcher.time_man.set_movetime(opt.movetime);
            Move best = searcher.search(pos, use_mcts_this_move ? 8 : 4, nullptr, 0);

            const std::string context = "paired game=" + std::to_string(game)
                                      + " ply=" + std::to_string(ply)
                                      + (use_mcts_this_move ? " mcts" : " ab");
            if (!check_best_move(pos, best, context.c_str()))
                return false;
            pos.do_move(best);
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse_options(argc, argv);
    std::cout << "mcts_safety: init"
              << " roots=" << opt.roots
              << " search_roots=" << opt.search_roots
              << " movetime=" << opt.movetime
              << " threads=";
    for (std::size_t i = 0; i < opt.threads.size(); ++i)
        std::cout << (i ? "," : "") << opt.threads[i];
    std::cout << " paired_games=" << opt.paired_games << "\n" << std::flush;

    init_engine_tables();
    std::mt19937 rng(opt.seed);
    auto pos = std::make_unique<Position>();

    if (!run_direct_mcts_roots(opt, rng, *pos))
        return 1;
    if (!run_search_roots(opt, rng, *pos))
        return 1;
    if (!run_paired_smoke(opt))
        return 1;

    std::cout << "mcts_safety: all checked MCTS/Search moves were legal and do/undo restored state\n";
    return 0;
}

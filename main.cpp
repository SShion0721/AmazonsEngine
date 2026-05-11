/*==================================================================
 * AMAZONS ENGINE -- main.cpp
 * UCI-compatible command loop and engine entry point.
 * Mirrors Stockfish's uci.cpp / main.cpp
 *
 * Supported commands:
 *   uci              -- engine identification
 *   isready          -- sync (engine replies "readyok")
 *   ucinewgame       -- reset TT + position
 *   position startpos [moves <move...>]
 *   go movetime <ms>
 *   go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <n>]
 *   stop             -- stop search
 *   quit             -- exit
 *   d                -- print current board (debug)
 *   eval             -- print evaluation breakdown (debug)
 *==================================================================*/
#include "position.h"
#include "movegen.h"
#include "evaluate.h"
#include "search.h"
#include "search_threads.h"
#include "tt.h"
#include "rays.h"
#include "bitboard.h"
#include "line_pattern.h"
#include "mcts.h"
#include "policy_prior.h"
#include "nnue/nnue.h"
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <functional>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <iomanip>
#include <limits>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

// ==== Global objects ====
static TranspositionTable g_tt(64); // 64 MB TT
static Searcher           g_searcher(g_tt);
static Position           g_pos;
static std::thread        g_search_thread;

// ==== UCI Configuration (mirrors Stockfish's Options map) ====
static int    g_threads      = 1;     // Lazy SMP thread count
static int    g_move_overhead = 10;   // ms safety buffer for communication delay
static std::string g_eval_file = "nnue.bin"; // NNUE weight file path
static std::string g_policy_file = "policy.bin";
static bool   g_save_policy_visits = false;
static int    g_policy_visit_topk = 128;

namespace {

struct CaseInsensitiveLess {
    bool operator()(const std::string& lhs, const std::string& rhs) const {
        return std::lexicographical_compare(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            [](unsigned char a, unsigned char b) { return std::tolower(a) < std::tolower(b); });
    }
};

struct UciOption {
    enum class Type { Spin, String, Button, Check };

    Type type;
    std::string default_value;
    std::string current_value;
    int min_value = 0;
    int max_value = 0;
    std::size_t order = 0;
    std::function<std::optional<std::string>(const std::string&)> on_change;
};

using UciOptionMap = std::map<std::string, UciOption, CaseInsensitiveLess>;

UciOptionMap g_uci_options;
std::size_t g_option_insert_order = 0;
bool g_options_initialized = false;
SearchThreadPool g_go_helpers(g_tt);
std::atomic<bool> g_selfplay_interrupt{ false };
std::atomic<bool> g_selfplay_active{ false };

bool try_parse_int(const std::string& text, int& out) {
    try {
        std::size_t idx = 0;
        long long parsed = std::stoll(text, &idx, 10);
        if (idx != text.size()) return false;
        if (parsed < std::numeric_limits<int>::min()
            || parsed > std::numeric_limits<int>::max()) return false;
        out = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool try_parse_bool(const std::string& text, bool& out) {
    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (lowered == "true" || lowered == "1" || lowered == "yes" || lowered == "on") {
        out = true;
        return true;
    }
    if (lowered == "false" || lowered == "0" || lowered == "no" || lowered == "off") {
        out = false;
        return true;
    }
    return false;
}

struct SelfplayThreadPlan {
    int game_threads = 1;
    int search_threads = 1;
    unsigned logical_threads = 1;
    bool auto_selected = false;
};

unsigned detect_logical_threads() {
#ifdef _WIN32
    const DWORD active = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (active > 0)
        return static_cast<unsigned>(active);

    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (info.dwNumberOfProcessors > 0)
        return static_cast<unsigned>(info.dwNumberOfProcessors);
#endif

    unsigned logical = std::thread::hardware_concurrency();
    return logical == 0 ? 1u : logical;
}

bool should_interrupt_selfplay() {
    return g_selfplay_interrupt.load(std::memory_order_relaxed);
}

void request_selfplay_interrupt() {
    g_selfplay_interrupt.store(true, std::memory_order_relaxed);
}

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
        if (g_selfplay_active.load(std::memory_order_relaxed)) {
            request_selfplay_interrupt();
            return TRUE;
        }
        return FALSE;
    default:
        return FALSE;
    }
}
#endif

std::vector<std::filesystem::path> find_selfplay_shards(const std::filesystem::path& output_path) {
    namespace fs = std::filesystem;

    fs::path directory = output_path.has_parent_path() ? output_path.parent_path() : fs::current_path();
    std::string prefix = output_path.filename().string() + ".part";
    std::vector<fs::path> shards;

    if (!fs::exists(directory))
        return shards;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file())
            continue;

        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && entry.path().extension() == ".tmp")
            shards.push_back(entry.path());
    }

    std::sort(shards.begin(), shards.end());
    return shards;
}

bool merge_selfplay_shards(const std::vector<std::filesystem::path>& shards,
                           const std::filesystem::path& output_path) {
    if (shards.empty())
        return true;

    std::ofstream out(output_path, std::ios::app | std::ios::binary);
    if (!out)
        return false;

    std::vector<char> merge_buffer(1 << 20);
    for (const auto& shard : shards) {
        std::ifstream part_in(shard, std::ios::binary);
        if (!part_in)
            continue;

        while (part_in) {
            part_in.read(merge_buffer.data(), static_cast<std::streamsize>(merge_buffer.size()));
            std::streamsize count = part_in.gcount();
            if (count > 0)
                out.write(merge_buffer.data(), count);
        }
        part_in.close();
        std::error_code ec;
        std::filesystem::remove(shard, ec);
    }

    out.flush();
    return bool(out);
}

#pragma pack(push, 1)
struct PolicyVisitHeader {
    char magic[8];
    uint32_t version;
    uint32_t board_squares;
    uint32_t max_topk;
};

struct PolicyVisitFixed {
    int8_t side;
    char board[100];
    int8_t outcome;
    int16_t score;
    uint32_t best_move;
    uint16_t visit_count;
};

struct PolicyVisitEntry {
    uint32_t move;
    uint16_t visits;
};
#pragma pack(pop)

struct PolicyVisitRecordMem {
    int8_t side = 0;
    std::array<char, BOARD_SQ> board{};
    int8_t outcome = 0;
    int16_t score = 0;
    Move best_move = MOVE_NONE;
    std::vector<PolicyVisitEntry> visits;
};

bool write_policy_visit_header(std::ostream& out, int max_topk) {
    PolicyVisitHeader header{};
    const char magic[8] = {'A', 'M', 'Z', 'S', 'V', '3', '\0', '\0'};
    std::memcpy(header.magic, magic, sizeof(header.magic));
    header.version = 1;
    header.board_squares = BOARD_SQ;
    header.max_topk = static_cast<uint32_t>(std::max(1, max_topk));
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    return bool(out);
}

bool write_policy_visit_record(std::ostream& out, const PolicyVisitRecordMem& rec) {
    PolicyVisitFixed fixed{};
    fixed.side = rec.side;
    std::memcpy(fixed.board, rec.board.data(), rec.board.size());
    fixed.outcome = rec.outcome;
    fixed.score = rec.score;
    fixed.best_move = static_cast<uint32_t>(rec.best_move);
    fixed.visit_count = static_cast<uint16_t>(std::min<std::size_t>(rec.visits.size(), 65535));

    out.write(reinterpret_cast<const char*>(&fixed), sizeof(fixed));
    if (fixed.visit_count > 0) {
        out.write(reinterpret_cast<const char*>(rec.visits.data()),
                  static_cast<std::streamsize>(fixed.visit_count * sizeof(PolicyVisitEntry)));
    }
    return bool(out);
}

bool merge_policy_visit_shards(const std::vector<std::filesystem::path>& shards,
                               const std::filesystem::path& output_path,
                               int max_topk) {
    if (shards.empty())
        return true;

    const bool need_header = !std::filesystem::exists(output_path)
                          || std::filesystem::file_size(output_path) == 0;

    std::ofstream out(output_path, std::ios::app | std::ios::binary);
    if (!out)
        return false;
    if (need_header && !write_policy_visit_header(out, max_topk))
        return false;

    std::vector<char> merge_buffer(1 << 20);
    for (const auto& shard : shards) {
        std::ifstream part_in(shard, std::ios::binary);
        if (!part_in)
            continue;

        while (part_in) {
            part_in.read(merge_buffer.data(), static_cast<std::streamsize>(merge_buffer.size()));
            std::streamsize count = part_in.gcount();
            if (count > 0)
                out.write(merge_buffer.data(), count);
        }
        part_in.close();
        std::error_code ec;
        std::filesystem::remove(shard, ec);
    }

    out.flush();
    return bool(out);
}

std::optional<int> parse_positive_int_or_auto(const std::string& text) {
    if (text.empty())
        return std::nullopt;

    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (lowered == "auto" || lowered == "0")
        return std::nullopt;

    int value = 0;
    if (!try_parse_int(text, value) || value <= 0)
        return std::nullopt;
    return value;
}

int preferred_search_threads_for_selfplay(unsigned logical_threads) {
    if (logical_threads <= 8)
        return 1;
    if (logical_threads <= 32)
        return 2;
    return 3;
}

SelfplayThreadPlan resolve_selfplay_threads(
    int games,
    const std::optional<int>& requested_game_threads,
    const std::optional<int>& requested_search_threads) {

    const unsigned logical = detect_logical_threads();
    const int capped_games = std::max(1, games);
    SelfplayThreadPlan plan;
    plan.logical_threads = logical;

    if (requested_game_threads && requested_search_threads) {
        plan.game_threads = std::max(1, std::min(*requested_game_threads, capped_games));
        plan.search_threads = std::max(1, *requested_search_threads);
        return plan;
    }

    plan.auto_selected = true;

    if (!requested_game_threads && !requested_search_threads) {
        const int preferred_search = preferred_search_threads_for_selfplay(logical);
        plan.game_threads = std::max(1, std::min(capped_games, int(logical) / preferred_search));
        plan.search_threads = std::max(1, int(logical) / plan.game_threads);
        return plan;
    }

    if (requested_game_threads) {
        plan.game_threads = std::max(1, std::min(*requested_game_threads, capped_games));
        plan.search_threads = std::max(1, int(logical) / plan.game_threads);
        return plan;
    }

    plan.search_threads = std::max(1, *requested_search_threads);
    plan.game_threads = std::max(1, std::min(capped_games, int(logical) / plan.search_threads));
    return plan;
}

void add_spin_option(const std::string& name,
                     int default_value,
                     int min_value,
                     int max_value,
                     std::function<std::optional<std::string>(const std::string&)> on_change) {
    UciOption option;
    option.type = UciOption::Type::Spin;
    option.default_value = std::to_string(default_value);
    option.current_value = option.default_value;
    option.min_value = min_value;
    option.max_value = max_value;
    option.order = g_option_insert_order++;
    option.on_change = std::move(on_change);
    g_uci_options[name] = std::move(option);
}

void add_string_option(const std::string& name,
                       const std::string& default_value,
                       std::function<std::optional<std::string>(const std::string&)> on_change) {
    UciOption option;
    option.type = UciOption::Type::String;
    option.default_value = default_value;
    option.current_value = default_value;
    option.order = g_option_insert_order++;
    option.on_change = std::move(on_change);
    g_uci_options[name] = std::move(option);
}

void add_button_option(const std::string& name,
                       std::function<std::optional<std::string>(const std::string&)> on_change) {
    UciOption option;
    option.type = UciOption::Type::Button;
    option.order = g_option_insert_order++;
    option.on_change = std::move(on_change);
    g_uci_options[name] = std::move(option);
}

void add_check_option(const std::string& name,
                      bool default_value,
                      std::function<std::optional<std::string>(const std::string&)> on_change) {
    UciOption option;
    option.type = UciOption::Type::Check;
    option.default_value = default_value ? "true" : "false";
    option.current_value = option.default_value;
    option.order = g_option_insert_order++;
    option.on_change = std::move(on_change);
    g_uci_options[name] = std::move(option);
}

void init_uci_options() {
    if (g_options_initialized) return;
    g_options_initialized = true;

    add_spin_option("Hash", 64, 1, 4096, [](const std::string& value) -> std::optional<std::string> {
        int mb = 0;
        if (!try_parse_int(value, mb)) return "Invalid Hash value: " + value;
        g_tt.resize(std::max(1, mb));
        return "Hash resized to " + std::to_string(std::max(1, mb)) + " MB";
    });

    add_spin_option("Threads", 1, 1, 128, [](const std::string& value) -> std::optional<std::string> {
        int threads = 0;
        if (!try_parse_int(value, threads)) return "Invalid Threads value: " + value;
        g_threads = std::max(1, threads);
        g_go_helpers.resize(g_threads - 1);
        return "Threads set to " + std::to_string(g_threads);
    });

    add_spin_option("Move Overhead", 10, 0, 5000,
                    [](const std::string& value) -> std::optional<std::string> {
        int overhead = 0;
        if (!try_parse_int(value, overhead)) return "Invalid Move Overhead value: " + value;
        g_move_overhead = std::max(0, overhead);
        return "Move Overhead set to " + std::to_string(g_move_overhead) + " ms";
    });

    add_string_option("EvalFile", g_eval_file,
                      [](const std::string& value) -> std::optional<std::string> {
        g_eval_file = value;
        bool loaded = NNUE::load_weights(g_eval_file);
        g_nnue_loaded = loaded;
        NNUE::g_accumulator_enabled = g_use_nnue && g_nnue_loaded;
        if (NNUE::accumulator_enabled())
            g_pos.compute_accumulator();
        g_tt.clear();
        clear_eval_cache();
        if (loaded) {
            if (g_use_nnue)
                return "NNUE weights loaded from " + g_eval_file
                     + (g_use_pure_nnue ? "; pure NNUE eval active; hash cleared"
                                        : "; Fusion NNUE v2 eval active; hash cleared");
            return "NNUE weights loaded from " + g_eval_file
                 + "; Use NNUE=false so classical eval remains active; hash cleared";
        }
        return "Failed to load " + g_eval_file + "; using classical eval only";
    });

    add_string_option("PolicyFile", g_policy_file,
                      [](const std::string& value) -> std::optional<std::string> {
        g_policy_file = value;
        const bool loaded = load_policy_prior(g_policy_file);
        clear_mcts_tree();
        g_tt.clear();
        if (loaded)
            return "AMZPOL1 policy prior loaded from " + g_policy_file;
        return "Failed to load " + g_policy_file + "; using heuristic prior fallback";
    });

    add_check_option("Use NNUE", false,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use NNUE value: " + value;

        g_use_nnue = enabled;
        NNUE::g_accumulator_enabled = g_use_nnue && g_nnue_loaded;
        if (NNUE::accumulator_enabled())
            g_pos.compute_accumulator();
        g_tt.clear();
        clear_eval_cache();

        if (!g_use_nnue)
            return "Use NNUE disabled; using classical eval";
        if (g_nnue_loaded)
            return g_use_pure_nnue
                 ? "Use NNUE enabled; pure NNUE eval is active"
                 : "Use NNUE enabled; Fusion NNUE v2 is active";
        return "Use NNUE enabled, but no valid EvalFile is loaded; using classical eval";
    });

    add_check_option("Use Residual NNUE", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Residual NNUE value: " + value;

        g_use_residual_nnue = enabled;
        g_tt.clear();
        clear_eval_cache();
        return enabled ? "Residual NNUE enabled; NNUE score is added to classical eval"
                       : "Residual NNUE disabled";
    });

    add_check_option("Use Pure NNUE", false,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Pure NNUE value: " + value;

        g_use_pure_nnue = enabled;
        g_tt.clear();
        clear_eval_cache();
        return enabled ? "Pure NNUE enabled; classical eval will be bypassed when Use NNUE=true"
                       : "Pure NNUE disabled; residual/classical eval path restored";
    });

    add_check_option("Use Strong Classical", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Strong Classical value: " + value;

        g_use_strong_classical = enabled;
        g_tt.clear();
        clear_eval_cache();
        return enabled ? "Strong classical eval enabled"
                       : "Strong classical eval disabled; legacy territory/mobility eval active";
    });

    add_string_option("Eval Tier", "Mixed",
                      [](const std::string& value) -> std::optional<std::string> {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (lowered == "fast")
            g_eval_tier = EVAL_TIER_FAST;
        else if (lowered == "mixed")
            g_eval_tier = EVAL_TIER_MIXED;
        else if (lowered == "strong")
            g_eval_tier = EVAL_TIER_STRONG;
        else
            return "Invalid Eval Tier value: " + value + " (expected Fast, Mixed, or Strong)";

        g_tt.clear();
        clear_eval_cache();
        return "Eval Tier set to " + value;
    });

    add_string_option("Search Mode", "Match",
                      [](const std::string& value) -> std::optional<std::string> {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        if (lowered == "fastselfplay" || lowered == "fast") {
            g_search_mode = SEARCH_MODE_FAST_SELFPLAY;
            g_teacher_safe_search = false;
            g_use_bounded_movegen = true;
            g_use_lmp_pruning = true;
            g_use_null_move_pruning = false;
            g_use_move_categories = false;
            g_use_mcts_root = false;
            g_use_strong_classical = false;
            g_eval_tier = EVAL_TIER_FAST;
        } else if (lowered == "teachersafe" || lowered == "teacher") {
            g_search_mode = SEARCH_MODE_TEACHER_SAFE;
            g_teacher_safe_search = true;
            g_use_bounded_movegen = false;
            g_use_lmp_pruning = false;
            g_use_null_move_pruning = false;
            g_use_move_categories = true;
            g_use_mcts_root = false;
            g_use_strong_classical = true;
            g_eval_tier = EVAL_TIER_STRONG;
        } else if (lowered == "match") {
            g_search_mode = SEARCH_MODE_MATCH;
            g_teacher_safe_search = false;
            g_use_bounded_movegen = false;
            g_use_lmp_pruning = true;
            g_use_null_move_pruning = false;
            g_use_move_categories = true;
            g_use_mcts_root = true;
            g_use_strong_classical = true;
            g_eval_tier = EVAL_TIER_MIXED;
        } else {
            return "Invalid Search Mode value: " + value
                 + " (expected FastSelfplay, TeacherSafe, or Match)";
        }

        g_tt.clear();
        clear_eval_cache();
        clear_mcts_tree();
        return "Search Mode set to " + value + "; recommended pruning/eval defaults applied";
    });

    add_check_option("Use Bounded MoveGen", false,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Bounded MoveGen value: " + value;
        g_use_bounded_movegen = enabled;
        return enabled ? "Bounded MoveGen enabled" : "Bounded MoveGen disabled";
    });

    add_spin_option("Bounded Dest Cap", 16, 1, 64,
                    [](const std::string& value) -> std::optional<std::string> {
        int cap = 0;
        if (!try_parse_int(value, cap)) return "Invalid Bounded Dest Cap value: " + value;
        g_bounded_dest_cap = std::clamp(cap, 1, 64);
        return "Bounded Dest Cap set to " + std::to_string(g_bounded_dest_cap);
    });

    add_spin_option("Bounded Arrow Cap", 12, 1, 64,
                    [](const std::string& value) -> std::optional<std::string> {
        int cap = 0;
        if (!try_parse_int(value, cap)) return "Invalid Bounded Arrow Cap value: " + value;
        g_bounded_arrow_cap = std::clamp(cap, 1, 64);
        return "Bounded Arrow Cap set to " + std::to_string(g_bounded_arrow_cap);
    });

    add_check_option("Use Null Move", false,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Null Move value: " + value;

        g_use_null_move_pruning = enabled;
        g_tt.clear();
        return enabled ? "Null move pruning enabled"
                       : "Null move pruning disabled";
    });

    add_check_option("Use LMP", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use LMP value: " + value;

        g_use_lmp_pruning = enabled;
        g_tt.clear();
        return enabled ? "LMP enabled with Amazons phase caps"
                       : "LMP disabled";
    });

    add_check_option("Use Move Categories", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Move Categories value: " + value;

        g_use_move_categories = enabled;
        g_tt.clear();
        return enabled ? "Amazons move categories enabled"
                       : "Amazons move categories disabled";
    });

    add_check_option("Use Policy Prior", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Policy Prior value: " + value;

        g_use_policy_prior = enabled;
        clear_mcts_tree();
        return enabled ? "Policy prior enabled for PUCT when AMZPOL1 is loaded"
                       : "Policy prior disabled; PUCT uses heuristic prior";
    });

    add_spin_option("Policy Prior Blend", 70, 0, 100,
                    [](const std::string& value) -> std::optional<std::string> {
        int blend = 0;
        if (!try_parse_int(value, blend)) return "Invalid Policy Prior Blend value: " + value;
        g_policy_prior_blend = std::clamp(blend, 0, 100);
        clear_mcts_tree();
        return "Policy Prior Blend set to " + std::to_string(g_policy_prior_blend) + "%";
    });

    add_check_option("Save Policy Visits", false,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Save Policy Visits value: " + value;
        g_save_policy_visits = enabled;
        return enabled ? "Match selfplay AMZSV3 policy-visit recording enabled"
                       : "Match selfplay AMZSV3 policy-visit recording disabled";
    });

    add_spin_option("Policy Visit TopK", 128, 1, 512,
                    [](const std::string& value) -> std::optional<std::string> {
        int topk = 0;
        if (!try_parse_int(value, topk)) return "Invalid Policy Visit TopK value: " + value;
        g_policy_visit_topk = std::clamp(topk, 1, 512);
        return "Policy Visit TopK set to " + std::to_string(g_policy_visit_topk);
    });

    add_check_option("Use Soft Tail Search", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Soft Tail Search value: " + value;

        g_use_soft_tail_search = enabled;
        g_tt.clear();
        return enabled ? "Soft tail search enabled"
                       : "Soft tail search disabled; top-K behaves as hard truncation";
    });

    add_check_option("Use MCTS Root", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use MCTS Root value: " + value;

        g_use_mcts_root = enabled;
        clear_mcts_tree();
        return enabled ? "MCTS root enabled for Match mode"
                       : "MCTS root disabled";
    });

    add_spin_option("MCTS Time Share", 75, 1, 100,
                    [](const std::string& value) -> std::optional<std::string> {
        int share = 0;
        if (!try_parse_int(value, share)) return "Invalid MCTS Time Share value: " + value;
        g_mcts_time_share = std::clamp(share, 1, 100);
        return "MCTS Time Share set to " + std::to_string(g_mcts_time_share) + "%";
    });

    add_spin_option("MCTS Min Time", 80, 1, 5000,
                    [](const std::string& value) -> std::optional<std::string> {
        int ms = 0;
        if (!try_parse_int(value, ms)) return "Invalid MCTS Min Time value: " + value;
        g_mcts_min_time_ms = std::clamp(ms, 1, 5000);
        return "MCTS Min Time set to " + std::to_string(g_mcts_min_time_ms) + " ms";
    });

    add_spin_option("MCTS Cpuct", 120, 1, 1000,
                    [](const std::string& value) -> std::optional<std::string> {
        int cpuct = 0;
        if (!try_parse_int(value, cpuct)) return "Invalid MCTS Cpuct value: " + value;
        g_mcts_cpuct = std::clamp(cpuct, 1, 1000);
        clear_mcts_tree();
        return "MCTS Cpuct set to " + std::to_string(g_mcts_cpuct) + " (divided by 100)";
    });

    add_spin_option("MCTS Threads", 1, 1, 128,
                    [](const std::string& value) -> std::optional<std::string> {
        int threads = 0;
        if (!try_parse_int(value, threads)) return "Invalid MCTS Threads value: " + value;
        g_mcts_threads = std::clamp(threads, 1, 128);
        return "MCTS Threads set to " + std::to_string(g_mcts_threads);
    });

    add_check_option("Use Tree Reuse", true,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Use Tree Reuse value: " + value;
        g_use_tree_reuse = enabled;
        if (!enabled)
            clear_mcts_tree();
        return enabled ? "MCTS root tree reuse enabled"
                       : "MCTS root tree reuse disabled and cleared";
    });

    add_spin_option("AB Verify TopN", 6, 0, 64,
                    [](const std::string& value) -> std::optional<std::string> {
        int topn = 0;
        if (!try_parse_int(value, topn)) return "Invalid AB Verify TopN value: " + value;
        g_ab_verify_topn = std::clamp(topn, 0, 64);
        return "AB Verify TopN set to " + std::to_string(g_ab_verify_topn);
    });

    add_check_option("Teacher Safe Search", false,
                     [](const std::string& value) -> std::optional<std::string> {
        bool enabled = false;
        if (!try_parse_bool(value, enabled))
            return "Invalid Teacher Safe Search value: " + value;

        g_teacher_safe_search = enabled;
        if (enabled) {
            g_search_mode = SEARCH_MODE_TEACHER_SAFE;
            g_use_bounded_movegen = false;
            g_use_lmp_pruning = false;
            g_use_null_move_pruning = false;
            g_use_mcts_root = false;
            g_use_strong_classical = true;
            g_eval_tier = EVAL_TIER_STRONG;
        } else if (g_search_mode == SEARCH_MODE_TEACHER_SAFE) {
            g_search_mode = SEARCH_MODE_MATCH;
            g_use_bounded_movegen = false;
            g_use_lmp_pruning = true;
            g_use_null_move_pruning = false;
            g_use_mcts_root = true;
            g_use_strong_classical = true;
            g_eval_tier = EVAL_TIER_MIXED;
        }
        g_tt.clear();
        return enabled ? "Teacher safe search enabled; selective pruning reduced"
                       : "Teacher safe search disabled";
    });

    add_spin_option("Territory Weight", 10, 0, 100,
                    [](const std::string& value) -> std::optional<std::string> {
        int weight = 0;
        if (!try_parse_int(value, weight)) return "Invalid Territory Weight value: " + value;
        set_territory_weight(weight);
        g_tt.clear();
        clear_eval_cache();
        return "Territory Weight set to " + std::to_string(weight) + "; hash cleared";
    });

    add_spin_option("Mobility Weight", 2, 0, 50,
                    [](const std::string& value) -> std::optional<std::string> {
        int weight = 0;
        if (!try_parse_int(value, weight)) return "Invalid Mobility Weight value: " + value;
        set_mobility_weight(weight);
        g_tt.clear();
        clear_eval_cache();
        return "Mobility Weight set to " + std::to_string(weight) + "; hash cleared";
    });

    add_button_option("Clear Hash", [](const std::string&) -> std::optional<std::string> {
        g_tt.clear();
        clear_eval_cache();
        return "Hash and eval cache cleared";
    });
}

void print_uci_options() {
    for (std::size_t order = 0; order < g_uci_options.size(); ++order) {
        for (const auto& [name, option] : g_uci_options) {
            if (option.order != order) continue;

            std::cout << "option name " << name << " type ";
            if (option.type == UciOption::Type::Spin) {
                std::cout << "spin default " << option.default_value
                          << " min " << option.min_value
                          << " max " << option.max_value << "\n";
            } else if (option.type == UciOption::Type::Check) {
                std::cout << "check default " << option.default_value << "\n";
            } else if (option.type == UciOption::Type::String) {
                std::cout << "string default " << option.default_value << "\n";
            } else {
                std::cout << "button\n";
            }
            break;
        }
    }
}

void setoption_from_stream(std::istringstream& is) {
    std::string token;
    std::string option_name;
    std::string option_value;

    is >> token; // consume "name"
    while (is >> token && token != "value")
        option_name += (option_name.empty() ? "" : " ") + token;

    if (token == "value")
        std::getline(is >> std::ws, option_value);

    auto it = g_uci_options.find(option_name);
    if (it == g_uci_options.end()) {
        std::cout << "info string Unknown option: " << option_name << "\n";
        return;
    }

    UciOption& option = it->second;
    if (option.type == UciOption::Type::Button) {
        option_value.clear();
    } else if (option_value.empty()) {
        std::cout << "info string Missing value for option: " << option_name << "\n";
        return;
    }

    if (option.type == UciOption::Type::Spin) {
        int parsed = 0;
        if (!try_parse_int(option_value, parsed)) {
            std::cout << "info string Invalid integer for option " << option_name
                      << ": " << option_value << "\n";
            return;
        }
        if (parsed < option.min_value || parsed > option.max_value) {
            std::cout << "info string Value out of range for option " << option_name
                      << ": " << option_value << "\n";
            return;
        }
    } else if (option.type == UciOption::Type::Check) {
        bool parsed = false;
        if (!try_parse_bool(option_value, parsed)) {
            std::cout << "info string Invalid boolean for option " << option_name
                      << ": " << option_value << "\n";
            return;
        }
    }

    option.current_value = option_value;
    if (option.on_change) {
        if (auto message = option.on_change(option_value))
            std::cout << "info string " << *message << "\n";
    }
}

} // namespace

// ==== Parse "position" command ====
// Format: position startpos [moves e1-e3/e4 ...]
static void stop_search_and_join(bool request_stop = true) {
    if (!g_search_thread.joinable()) {
        return;
    }

    if (request_stop) {
        g_go_helpers.request_stop();
        g_searcher.request_stop();
    }

    g_search_thread.join();
    g_go_helpers.wait_for_search_finished();
}
static void cmd_position(std::istringstream& is) {
    std::string token;
    is >> token;

    if (token == "startpos") {
        g_pos.set_startpos();
    }
    // (FEN parsing could be added here)

    if (is >> token && token == "moves") {
        std::string mv_str;
        while (is >> mv_str) {
            // Parse "a4-d7/d3" format
            // from: up to '-', to: up to '/', arrow: rest
            auto dash  = mv_str.find('-');
            auto slash = mv_str.find('/');
            if (dash == std::string::npos || slash == std::string::npos) break;

            Square from  = str_to_sq(mv_str.substr(0, dash));
            Square to    = str_to_sq(mv_str.substr(dash + 1, slash - dash - 1));
            Square arrow = str_to_sq(mv_str.substr(slash + 1));

            if (from < 0 || to < 0 || arrow < 0) break;
            Move m = encode_move(from, to, arrow);
            if (!is_pseudo_legal(g_pos, m)) {
                std::cout << "info string Ignoring illegal move in position command: "
                          << mv_str << "\n";
                break;
            }
            g_pos.do_move(m);
        }
    }
}

// ==== Parse "go" command ====
// Search runs in a background thread so the UCI loop stays responsive.
static void cmd_go(std::istringstream& is) {
    std::string token;
    int wtime = 0, btime = 0, winc = 0, binc = 0;
    int movestogo = 30, movetime = 0;
    int depth_limit = 64;
    uint64_t node_limit = 0;

    while (is >> token) {
        if      (token == "wtime"     && is >> wtime)    {}
        else if (token == "btime"     && is >> btime)    {}
        else if (token == "winc"      && is >> winc)     {}
        else if (token == "binc"      && is >> binc)     {}
        else if (token == "movestogo" && is >> movestogo) {}
        else if (token == "movetime"  && is >> movetime)  {}
        else if (token == "depth"     && is >> depth_limit) {}
        else if (token == "nodes") {
            int parsed_nodes = 0;
            if (is >> parsed_nodes && parsed_nodes > 0)
                node_limit = static_cast<uint64_t>(parsed_nodes);
        }
    }
    stop_search_and_join();

    auto root_pos = std::make_shared<Position>(g_pos);

    g_search_thread = std::thread([wtime, btime, winc, binc, movestogo, movetime,
                                   depth_limit, node_limit, root_pos]() mutable {
        if (movetime > 0) {
            g_searcher.time_man.set_movetime(movetime, g_move_overhead);
        } else if (node_limit > 0 || (wtime <= 0 && btime <= 0)) {
            g_searcher.time_man.set_movetime(60 * 60 * 1000, 0);
        } else {
            int my_time = (root_pos->side_to_move == WHITE) ? wtime : btime;
            int my_inc  = (root_pos->side_to_move == WHITE) ? winc  : binc;
            g_searcher.time_man.init(my_time, my_inc, movestogo, g_move_overhead);
        }
        const uint64_t per_thread_node_limit = node_limit > 0
            ? std::max<uint64_t>(1, node_limit / static_cast<uint64_t>(std::max(1, g_threads)))
            : 0;
        g_searcher.node_limit = per_thread_node_limit;

        const bool root_mcts_likely = g_search_mode == SEARCH_MODE_MATCH
                                   && g_use_mcts_root
                                   && !g_teacher_safe_search
                                   && node_limit == 0
                                   && depth_limit >= 8
                                   && root_pos->bb_arrow.popcount() < 56
                                   && g_searcher.time_man.soft_limit_ms() >= g_mcts_min_time_ms
                                   && g_searcher.time_man.soft_limit_ms() <= 30000;
        if (!root_mcts_likely)
            g_go_helpers.start_searching(*root_pos, g_searcher.time_man, depth_limit, per_thread_node_limit);

        g_searcher.silent = false;
        g_searcher.search(*root_pos, depth_limit, nullptr, 0);

        if (!root_mcts_likely) {
            g_go_helpers.request_stop();
            g_go_helpers.wait_for_search_finished();
        }
    });
}

// ==== Main UCI loop ====
int main() {
    // Must-do initializations (like SF's main() calling init functions)
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#endif

    init_uci_options();
    g_go_helpers.resize(g_threads - 1);
    Zobrist::init();
    init_rays();
    init_bitboards();
    init_attack_tables();     // PEXT O(1) queen attack lookup tables (~3 MB)
    init_between_line_bb();   // BETWEEN_BB + LINE_BB path masks (~320 KB)
    init_line_patterns();     // AmazonsNNUE-v2 line pattern features
    if (NNUE::load_weights(g_eval_file)) {
        g_nnue_loaded = true;
        std::cout << "info string AMZNUE2 weights loaded successfully from " << g_eval_file << ".\n";
    } else {
        g_nnue_loaded = false;
        std::cout << "info string Failed to load " << g_eval_file
                  << ", using classical eval only.\n";
    }
    if (load_policy_prior(g_policy_file))
        std::cout << "info string AMZPOL1 policy prior loaded successfully from " << g_policy_file << ".\n";
    else
        std::cout << "info string Failed to load " << g_policy_file
                  << ", using heuristic policy prior fallback.\n";
    NNUE::g_accumulator_enabled = g_use_nnue && g_nnue_loaded;
    g_pos.set_startpos();

    std::cout << "Amazons Engine v2.1 - Stockfish-inspired (PEXT+ClusterTT+NNUE/ClassicalEval)\n";

    std::string line, token;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        is >> token;

        if (token == "quit") {
            stop_search_and_join();
            g_go_helpers.shutdown();
            break;

        } else if (token == "stop") {
            stop_search_and_join();

        } else if (token == "uci") {
            std::cout << "id name AmazonsEngine v2.1\n"
                      << "id author Sion\n"
                      << "\n";
            print_uci_options();
            std::cout << "uciok\n";

        } else if (token == "setoption") {
            stop_search_and_join();
            setoption_from_stream(is);

        } else if (token == "isready") {
            std::cout << "readyok\n";

        } else if (token == "ucinewgame") {
            stop_search_and_join();
            g_tt.clear();
            clear_eval_cache();
            clear_mcts_tree();
            g_searcher.new_game();
            g_go_helpers.new_game();
            g_pos.set_startpos();

        } else if (token == "position") {
            stop_search_and_join();
            cmd_position(is);

        } else if (token == "go") {
            cmd_go(is);

        } else if (token == "d") {
            // Debug: print board
            g_pos.print();

        } else if (token == "eval") {
            // Debug: print evaluation breakdown
            Score t = eval_territory(g_pos);
            Score m = eval_mobility(g_pos);
            EvalBreakdown bd;
            get_eval_breakdown(g_pos, bd);
            EvalInfo ei;
            get_eval_info(g_pos, ei);
            Score classical = evaluate_classical(g_pos);
            const bool nnue_active = g_use_nnue && g_nnue_loaded;
            Score n = nnue_active
                    ? NNUE::evaluate_nnue(g_pos.side_to_move,
                                          g_pos.history[g_pos.ply].accumulator,
                                          ei.global.v,
                                          ei.phase_bucket)
                    : 0;
            Score total = evaluate(g_pos);
            const char* mode_name = !nnue_active ? "Classical"
                                  : g_use_pure_nnue ? "Pure NNUE"
                                  : g_use_residual_nnue ? "Fusion NNUE v2"
                                  : "Classical";
            std::string nnue_status;
            if (!g_nnue_loaded)
                nnue_status = " (inactive: no weights)";
            else if (!g_use_nnue)
                nnue_status = " (inactive: Use NNUE=false)";
            std::cout << "Territory : " << t << " (x" << g_territory_weight
                      << " = " << t * g_territory_weight << ")\n"
                      << "Mobility  : " << m << " (x" << g_mobility_weight
                      << " = " << m * g_mobility_weight << ")\n"
                      << "StrongEval: phase " << bd.phase
                      << " bucket " << bd.phase_bucket
                      << " active " << bd.active_areas
                      << " t1 " << bd.t1
                      << " t2 " << bd.t2
                      << " p1 " << bd.p1
                      << " p2 " << bd.p2
                      << " m " << bd.m
                      << " part " << bd.partition
                      << " mob " << bd.mobility
                      << " rawW " << bd.raw_white << "\n"
                      << "Classical : " << classical << "\n"
                      << "NNUE      : " << n
                      << nnue_status << "\n"
                      << "NNUE fmt  : " << NNUE::weight_format() << "\n"
                      << "Policy fmt: " << policy_prior_format()
                      << (g_use_policy_prior ? "" : " (disabled)") << "\n"
                      << "Global    :";
            for (int i = 0; i < NNUE::GLOBAL_FEATURE_SIZE; ++i)
                std::cout << ' ' << int(ei.global.v[i]);
            std::cout << "\n"
                      << "Mode      : " << mode_name << "\n"
                      << "Total     : " << total
                      << " (from " << (g_pos.side_to_move == WHITE ? "White" : "Black")
                      << "'s perspective)\n";

        } else if (token == "perft") {
            // Perft: count leaf nodes at depth N (for move generator testing)
            int depth = 1;
            is >> depth;
            std::function<uint64_t(Position&, int)> perft =
                [&](Position& p, int d) -> uint64_t {
                    if (d == 0) return 1;
                    std::vector<Move> moves;
                    generate_moves(p, moves);
                    if (d == 1) return moves.size();
                    uint64_t nodes = 0;
                    for (Move m : moves) {
                        p.do_move(m);
                        nodes += perft(p, d - 1);
                        p.undo_move(m);
                    }
                    return nodes;
                };
            auto t0 = std::chrono::steady_clock::now();
            uint64_t count = perft(g_pos, depth);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
            std::cout << "Perft(" << depth << ") = " << count
                      << "  (" << ms << " ms)\n";

        } else if (token == "selfplay") {
            stop_search_and_join();
            int games = 1;
            int max_depth = 2; // Default depth
            int move_time = 50; // Default 50ms per move
            std::optional<int> requested_game_threads;
            int hash_size = 16; // Default 16 MB per game
            std::optional<int> requested_search_threads;
            uint64_t nodes_per_move = 0;

            std::vector<std::string> selfplay_args;
            while (is >> token) {
                selfplay_args.push_back(token);
            }

            if (!selfplay_args.empty()) {
                int parsed = 0;
                if (try_parse_int(selfplay_args[0], parsed) && parsed > 0)
                    games = parsed;
            }
            if (selfplay_args.size() >= 2) {
                int parsed = 0;
                if (try_parse_int(selfplay_args[1], parsed) && parsed > 0)
                    max_depth = parsed;
            }
            if (selfplay_args.size() >= 3) {
                int parsed = 0;
                if (try_parse_int(selfplay_args[2], parsed) && parsed > 0)
                    move_time = parsed;
            }
            if (selfplay_args.size() >= 4) {
                requested_game_threads = parse_positive_int_or_auto(selfplay_args[3]);
            }
            if (selfplay_args.size() >= 5) {
                int parsed = 0;
                if (try_parse_int(selfplay_args[4], parsed) && parsed > 0)
                    hash_size = parsed;
            }
            if (selfplay_args.size() >= 6) {
                requested_search_threads = parse_positive_int_or_auto(selfplay_args[5]);
            }
            if (selfplay_args.size() >= 7) {
                int parsed = 0;
                if (try_parse_int(selfplay_args[6], parsed) && parsed > 0)
                    nodes_per_move = static_cast<uint64_t>(parsed);
            }

            hash_size = std::max(1, hash_size);
            SelfplayThreadPlan thread_plan = resolve_selfplay_threads(
                games, requested_game_threads, requested_search_threads);
            const int game_threads = thread_plan.game_threads;
            const int search_threads = thread_plan.search_threads;

            std::cout << "Starting selfplay for " << games << " games...\n"
                      << "  Search Depth: " << max_depth << "\n"
                      << "  Move Time: " << move_time << "ms\n"
                      << "  Nodes/Move: " << (nodes_per_move ? std::to_string(nodes_per_move) : std::string("time-limited")) << "\n"
                      << "  Game Threads: " << game_threads << "\n"
                      << "  Search Threads/Game: " << search_threads << "\n"
                      << "  Hash Size/Game: " << hash_size << "MB\n";
            if (thread_plan.auto_selected) {
                std::cout << "  Auto Plan: " << game_threads << " x " << search_threads
                          << " (detected " << thread_plan.logical_threads
                          << " logical threads)\n";
            }

            const std::filesystem::path output_path("selfplay_data.bin");
            const std::filesystem::path policy_output_path("selfplay_policy_visits.amzsv3");
            const auto stale_shards = find_selfplay_shards(output_path);
            if (!stale_shards.empty()) {
                std::cout << "info string Found " << stale_shards.size()
                          << " unfinished selfplay shard(s); recovering before new run...\n";
                if (!merge_selfplay_shards(stale_shards, output_path)) {
                    std::cout << "info string Failed to recover unfinished selfplay shards; "
                                 "selfplay aborted.\n";
                    std::cout.flush();
                    continue;
                }
                std::cout << "info string Recovered unfinished selfplay data into "
                          << output_path.string() << "\n";
            }
            if (g_save_policy_visits) {
                const auto stale_policy_shards = find_selfplay_shards(policy_output_path);
                if (!stale_policy_shards.empty()) {
                    std::cout << "info string Found " << stale_policy_shards.size()
                              << " unfinished AMZSV3 shard(s); recovering before new run...\n";
                    if (!merge_policy_visit_shards(stale_policy_shards,
                                                   policy_output_path,
                                                   g_policy_visit_topk)) {
                        std::cout << "info string Failed to recover unfinished AMZSV3 shards; "
                                     "selfplay aborted.\n";
                        std::cout.flush();
                        continue;
                    }
                }
            }

            g_selfplay_interrupt.store(false, std::memory_order_relaxed);
            g_selfplay_active.store(true, std::memory_order_relaxed);

            // Define a compact binary structure matching numpy dtype
            #pragma pack(push, 1)
            struct GameRecord {
                int8_t side;       // 0 for WHITE, 1 for BLACK
                char board[100];   // '0', '1', '2', '3'
                int8_t outcome;    // 1 or -1
                int16_t score;     // search score (side-to-move perspective)
            };
            #pragma pack(pop)

            std::vector<std::filesystem::path> part_paths(game_threads);
            std::vector<std::filesystem::path> policy_part_paths(game_threads);
            for (int i = 0; i < game_threads; ++i) {
                part_paths[i] = output_path.string() + ".part" + std::to_string(i) + ".tmp";
                policy_part_paths[i] = policy_output_path.string() + ".part" + std::to_string(i) + ".tmp";
            }

            std::atomic<int> next_game{0};
            std::atomic<int> completed_games{0};
            std::mutex print_mutex;
            const int report_interval = std::max(1, games / 100);
            const auto progress_start = std::chrono::steady_clock::now();
            std::atomic<int> next_count_report{ std::min(games, report_interval) };
            std::atomic<long long> next_time_report_ms{ 1000 };
            constexpr std::size_t kBufferedRecords = 4096;

            auto report_progress = [&](int done, bool force = false) {
                const long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - progress_start).count();

                bool should_print = force;
                while (!should_print) {
                    int expected_games = next_count_report.load(std::memory_order_relaxed);
                    if (done >= expected_games) {
                        const int desired_games = std::min(games, expected_games + report_interval);
                        if (next_count_report.compare_exchange_weak(
                                expected_games, desired_games, std::memory_order_relaxed)) {
                            should_print = true;
                            break;
                        }
                        continue;
                    }

                    long long expected_ms = next_time_report_ms.load(std::memory_order_relaxed);
                    if (elapsed_ms >= expected_ms) {
                        if (next_time_report_ms.compare_exchange_weak(
                                expected_ms, expected_ms + 1000, std::memory_order_relaxed)) {
                            should_print = true;
                            break;
                        }
                        continue;
                    }
                    break;
                }

                if (!should_print)
                    return;

                const double pct = games > 0 ? (100.0 * done / games) : 100.0;
                const int filled = std::max(0, std::min(50, static_cast<int>(pct / 2.0)));
                const double elapsed_sec = std::max(0.001, elapsed_ms / 1000.0);
                const double games_per_sec = done / elapsed_sec;

                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "\r["
                          << std::string(filled, '#')
                          << std::string(50 - filled, '.')
                          << "] " << std::fixed << std::setprecision(2) << pct
                          << "% (" << done << "/" << games << ")"
                          << "  " << std::setprecision(1) << games_per_sec << " games/s"
                          << std::defaultfloat
                          << std::flush;
            };

            auto worker_func = [&](int thread_id) {
                std::ofstream thread_out(part_paths[thread_id], std::ios::binary | std::ios::trunc);
                if (!thread_out) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "\ninfo string Failed to open selfplay shard: "
                              << part_paths[thread_id].string() << "\n";
                    return;
                }
                std::unique_ptr<std::ofstream> policy_out;
                if (g_save_policy_visits) {
                    policy_out = std::make_unique<std::ofstream>(policy_part_paths[thread_id],
                                                                 std::ios::binary | std::ios::trunc);
                    if (!*policy_out) {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::cout << "\ninfo string Failed to open AMZSV3 shard: "
                                  << policy_part_paths[thread_id].string() << "\n";
                        return;
                    }
                }
                TranspositionTable local_tt(hash_size);
                Searcher local_searcher(local_tt);
                local_searcher.silent = true;
                SearchThreadPool local_helpers(local_tt);
                local_helpers.resize(search_threads - 1);
                auto local_pos = std::make_unique<Position>();

                std::mt19937 rng(static_cast<unsigned>(
                    std::chrono::steady_clock::now().time_since_epoch().count()
                    ^ (static_cast<uint64_t>(thread_id + 1) * 2654435761ULL)));

                std::vector<Move> moves;
                moves.reserve(2048);
                std::vector<GameRecord> write_buffer;
                write_buffer.reserve(kBufferedRecords);

                auto flush_buffer = [&]() {
                    if (write_buffer.empty()) {
                        return;
                    }
                    thread_out.write(reinterpret_cast<const char*>(write_buffer.data()),
                                     static_cast<std::streamsize>(write_buffer.size() * sizeof(GameRecord)));
                    write_buffer.clear();
                };

                while (true) {
                    if (should_interrupt_selfplay())
                        break;

                    int g = next_game.fetch_add(1);
                    if (g >= games)
                        break;

                    local_searcher.new_game();
                    local_helpers.new_game();
                    local_pos->set_startpos();

                    {
                        int random_plies = 4 + static_cast<int>(rng() % 5);
                        bool opening_ok = true;
                        for (int rp = 0; rp < random_plies; ++rp) {
                            if (should_interrupt_selfplay()) {
                                opening_ok = false;
                                break;
                            }
                            moves.clear();
                            generate_moves(*local_pos, moves);
                            if (moves.empty()) { opening_ok = false; break; }
                            std::uniform_int_distribution<int> dist(
                                0, static_cast<int>(moves.size()) - 1);
                            local_pos->do_move(moves[dist(rng)]);
                        }
                        if (!opening_ok) {
                            if (should_interrupt_selfplay())
                                break;
                            continue;
                        }
                    }

                    std::vector<GameRecord> records;
                    records.reserve(128);
                    std::vector<PolicyVisitRecordMem> policy_records;
                    if (g_save_policy_visits)
                        policy_records.reserve(128);
                    int white_result = 0;
                    bool interrupted_game = false;
                    bool discard_game = false;

                    while (true) {
                        if (should_interrupt_selfplay()) {
                            interrupted_game = true;
                            break;
                        }

                        if (!has_legal_move(*local_pos)) {
                            white_result = (local_pos->side_to_move == WHITE) ? -1 : 1;
                            break;
                        }

                        Score search_score = SCORE_ZERO;
                        if (nodes_per_move > 0)
                            local_searcher.time_man.set_movetime(60 * 60 * 1000, 0);
                        else
                            local_searcher.time_man.set_movetime(move_time);
                        const uint64_t per_thread_nodes = nodes_per_move > 0
                            ? std::max<uint64_t>(1, nodes_per_move / static_cast<uint64_t>(std::max(1, search_threads)))
                            : 0;
                        local_searcher.node_limit = per_thread_nodes;
                        const bool root_mcts_likely = g_search_mode == SEARCH_MODE_MATCH
                                                   && g_use_mcts_root
                                                   && !g_teacher_safe_search
                                                   && per_thread_nodes == 0
                                                   && max_depth >= 8
                                                   && move_time >= g_mcts_min_time_ms
                                                   && move_time <= 30000
                                                   && local_pos->bb_arrow.popcount() < 56;
                        if (!root_mcts_likely)
                            local_helpers.start_searching(*local_pos, local_searcher.time_man, max_depth, per_thread_nodes);
                        Move best = local_searcher.search(*local_pos, max_depth, &search_score);
                        if (!root_mcts_likely) {
                            local_helpers.request_stop();
                            local_helpers.wait_for_search_finished();
                        }

                        if (should_interrupt_selfplay()) {
                            interrupted_game = true;
                            break;
                        }

                        if (best == MOVE_NONE) {
                            white_result = (local_pos->side_to_move == WHITE) ? -1 : 1;
                            break;
                        }
                        if (!is_pseudo_legal(*local_pos, best)) {
                            std::lock_guard<std::mutex> lock(print_mutex);
                            std::cout << "\ninfo string Illegal selfplay bestmove "
                                      << move_to_str(best)
                                      << " at game " << g
                                      << " ply " << local_pos->ply
                                      << "; discarding this game.\n";
                            discard_game = true;
                            break;
                        }

                        GameRecord rec;
                        rec.side = (local_pos->side_to_move == WHITE) ? 0 : 1;
                        rec.outcome = 0;
                        int sc = static_cast<int>(search_score);
                        rec.score = static_cast<int16_t>(
                            sc > 32000 ? 32000 : (sc < -32000 ? -32000 : sc));
                        for (int sq = 0; sq < 100; ++sq)
                            rec.board[sq] = static_cast<char>('0' + local_pos->board[sq]);
                        records.push_back(rec);

                        if (g_save_policy_visits
                            && local_searcher.last_mcts_valid()
                            && !local_searcher.last_mcts_result().root_moves.empty()) {
                            PolicyVisitRecordMem prec;
                            prec.side = rec.side;
                            prec.outcome = 0;
                            prec.score = rec.score;
                            prec.best_move = best;
                            prec.board = {};
                            for (int sq = 0; sq < BOARD_SQ; ++sq)
                                prec.board[sq] = rec.board[sq];

                            std::vector<MctsRootMove> root_moves = local_searcher.last_mcts_result().root_moves;
                            root_moves.erase(std::remove_if(root_moves.begin(), root_moves.end(),
                                            [&](const MctsRootMove& rm) {
                                                return rm.move == MOVE_NONE
                                                    || rm.visits <= 0
                                                    || !is_pseudo_legal(*local_pos, rm.move);
                                            }),
                                            root_moves.end());
                            std::sort(root_moves.begin(), root_moves.end(),
                                      [](const MctsRootMove& a, const MctsRootMove& b) {
                                          if (a.visits != b.visits)
                                              return a.visits > b.visits;
                                          return a.mean > b.mean;
                                      });

                            const int keep = std::min<int>(g_policy_visit_topk, root_moves.size());
                            prec.visits.reserve(keep);
                            for (int i = 0; i < keep; ++i) {
                                PolicyVisitEntry entry{};
                                entry.move = static_cast<uint32_t>(root_moves[i].move);
                                entry.visits = static_cast<uint16_t>(
                                    std::clamp(root_moves[i].visits, 0, 65535));
                                prec.visits.push_back(entry);
                            }
                            if (!prec.visits.empty())
                                policy_records.push_back(std::move(prec));
                        }

                        local_pos->do_move(best);
                    }

                    if (interrupted_game)
                        break;
                    if (discard_game)
                        continue;

                    for (auto& rec : records) {
                        rec.outcome = (rec.side == 0)
                            ? static_cast<int8_t>(white_result)
                            : static_cast<int8_t>(-white_result);
                    }
                    for (auto& rec : policy_records) {
                        rec.outcome = (rec.side == 0)
                            ? static_cast<int8_t>(white_result)
                            : static_cast<int8_t>(-white_result);
                        if (policy_out && !write_policy_visit_record(*policy_out, rec)) {
                            std::lock_guard<std::mutex> lock(print_mutex);
                            std::cout << "\ninfo string Failed while writing AMZSV3 shard: "
                                      << policy_part_paths[thread_id].string() << "\n";
                            request_selfplay_interrupt();
                            interrupted_game = true;
                            break;
                        }
                    }
                    if (interrupted_game)
                        break;

                    write_buffer.insert(write_buffer.end(), records.begin(), records.end());
                    if (write_buffer.size() >= kBufferedRecords) {
                        flush_buffer();
                    }

                    int done = completed_games.fetch_add(1) + 1;
                    report_progress(done, done == games);
                }

                flush_buffer();
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < game_threads; ++i) {
                threads.emplace_back(worker_func, i);
            }

            for (auto& t : threads) {
                t.join();
            }

            const bool interrupted = should_interrupt_selfplay();
            const bool merge_ok = merge_selfplay_shards(part_paths, output_path);
            const bool policy_merge_ok = !g_save_policy_visits
                || merge_policy_visit_shards(policy_part_paths, policy_output_path, g_policy_visit_topk);
            g_selfplay_active.store(false, std::memory_order_relaxed);

            const int done = completed_games.load(std::memory_order_relaxed);
            if (interrupted || done == 0)
                report_progress(done, true);
            std::cout << "\n";
            if (!merge_ok) {
                std::cout << "info string Failed to merge one or more selfplay shard files.\n";
            }
            if (!policy_merge_ok) {
                std::cout << "info string Failed to merge one or more AMZSV3 policy visit shard files.\n";
            }
            if (interrupted) {
                std::cout << "Selfplay interrupted. " << done
                          << " completed games merged into " << output_path.string() << "\n";
            } else {
                std::cout << "Selfplay complete! " << done
                          << " games saved to " << output_path.string() << "\n";
                if (g_save_policy_visits)
                    std::cout << "AMZSV3 policy visits saved to "
                              << policy_output_path.string() << "\n";
            }
        }

        std::cout.flush();
    }
    return 0;
}











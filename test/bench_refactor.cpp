#include "../bitboard.h"
#include "../evaluate.h"
#include "../line_pattern.h"
#include "../movegen.h"
#include "../movepicker.h"
#include "../position.h"
#include "../search.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>

namespace {

using Clock = std::chrono::steady_clock;

void init_engine_tables() {
    Zobrist::init();
    init_rays();
    init_bitboards();
    init_attack_tables();
    init_between_line_bb();
    init_line_patterns();
    NNUE::init_random_weights();
    clear_eval_cache();
}

template <typename Fn>
double time_ms(Fn&& fn) {
    const auto start = Clock::now();
    fn();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int main() {
    init_engine_tables();
    auto pos = std::make_unique<Position>();
    pos->set_startpos();

    std::array<Move, MAX_LEGAL_MOVES> move_buf{};
    const int root_moves = generate_moves(*pos, move_buf.data(), MAX_LEGAL_MOVES);

    volatile std::uint64_t sink = 0;

    constexpr int MOVEGEN_ITERS = 20000;
    const double movegen_ms = time_ms([&] {
        for (int i = 0; i < MOVEGEN_ITERS; ++i)
            sink += static_cast<std::uint64_t>(generate_moves(*pos, move_buf.data(), MAX_LEGAL_MOVES));
    });

    int history[2][BOARD_SQ][BOARD_SQ]{};
    int arrow_history[2][BOARD_SQ][BOARD_SQ]{};
    int from_arrow_history[2][BOARD_SQ][BOARD_SQ]{};
    auto picker_buffer = std::make_unique<MovePickerBuffer>();
    Stack stack[2]{};
    Stack* ss = stack + 1;
    ss->ply = 0;

    constexpr int PICKER_ITERS = 1000;
    const double picker_ms = time_ms([&] {
        for (int i = 0; i < PICKER_ITERS; ++i) {
            MovePicker picker(*pos,
                              MOVE_NONE,
                              ss,
                              history,
                              arrow_history,
                              from_arrow_history,
                              MOVE_NONE,
                              *picker_buffer,
                              256);
            Move m;
            while ((m = picker.next_move()) != MOVE_NONE)
                sink += static_cast<std::uint64_t>(m);
        }
    });

    constexpr int EVAL_HIT_ITERS = 5000;
    g_use_nnue = false;
    const double eval_hit_ms = time_ms([&] {
        for (int i = 0; i < EVAL_HIT_ITERS; ++i)
            sink += static_cast<std::uint64_t>(evaluate(*pos) + 32768);
    });

    constexpr int EVAL_MISS_ITERS = 500;
    const uint64_t base_key = pos->key;
    const double eval_miss_ms = time_ms([&] {
        for (int i = 0; i < EVAL_MISS_ITERS; ++i) {
            pos->key = base_key ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i + 1));
            sink += static_cast<std::uint64_t>(evaluate(*pos) + 32768);
        }
    });
    pos->key = base_key;

    EvalInfo info;
    get_eval_info(*pos, info);
    constexpr int NNUE_ITERS = 5000;
    const double nnue_ms = time_ms([&] {
        for (int i = 0; i < NNUE_ITERS; ++i)
            sink += static_cast<std::uint64_t>(
                NNUE::evaluate_nnue(pos->side_to_move,
                                    pos->history[pos->ply].accumulator,
                                    info.global.v,
                                    info.phase_bucket) + 32768);
    });

    std::cout << "bench_refactor\n";
    std::cout << "root_moves " << root_moves << "\n";
    std::cout << "movegen_ms " << movegen_ms
              << " per_call_us " << (movegen_ms * 1000.0 / MOVEGEN_ITERS) << "\n";
    std::cout << "movepicker_ms " << picker_ms
              << " per_call_us " << (picker_ms * 1000.0 / PICKER_ITERS) << "\n";
    std::cout << "evaluate_cache_hit_ms " << eval_hit_ms
              << " per_call_us " << (eval_hit_ms * 1000.0 / EVAL_HIT_ITERS) << "\n";
    std::cout << "evaluate_cache_miss_ms " << eval_miss_ms
              << " per_call_us " << (eval_miss_ms * 1000.0 / EVAL_MISS_ITERS) << "\n";
    std::cout << "nnue_forward_ms " << nnue_ms
              << " per_call_us " << (nnue_ms * 1000.0 / NNUE_ITERS) << "\n";
    std::cout << "sink " << sink << "\n";
    return 0;
}

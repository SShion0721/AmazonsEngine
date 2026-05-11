cmake_minimum_required(VERSION 3.14)
project(AmazonsEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags (mirrors Stockfish's build flags)
if (MSVC)
    add_compile_options(/O2 /DNDEBUG /GL /arch:AVX2)
    add_link_options(/LTCG)
else()
    add_compile_options(-O3 -DNDEBUG -march=native -flto -funroll-loops)
    add_link_options(-flto -static-libgcc -static-libstdc++ -static)
endif()

add_executable(amazons
    main.cpp
    rays.cpp
    bitboard.cpp
    line_pattern.cpp
    position.cpp
    movegen.cpp
    evaluate.cpp
    policy_prior.cpp
    movepicker.cpp
    mcts.cpp
    search.cpp
    search_threads.cpp
    nnue/nnue.cpp
)

# Link threads (needed for std::thread)
find_package(Threads REQUIRED)
target_link_libraries(amazons PRIVATE Threads::Threads)

add_executable(convert_selfplay
    train/convert_selfplay.cpp
    rays.cpp
    bitboard.cpp
    line_pattern.cpp
    position.cpp
    evaluate.cpp
    policy_prior.cpp
    nnue/nnue.cpp
)

add_executable(verify_refactor
    test/verify_refactor.cpp
    rays.cpp
    bitboard.cpp
    line_pattern.cpp
    position.cpp
    movegen.cpp
    evaluate.cpp
    policy_prior.cpp
    nnue/nnue.cpp
)

add_executable(bench_refactor
    test/bench_refactor.cpp
    rays.cpp
    bitboard.cpp
    line_pattern.cpp
    position.cpp
    movegen.cpp
    movepicker.cpp
    mcts.cpp
    search.cpp
    evaluate.cpp
    policy_prior.cpp
    nnue/nnue.cpp
)
target_link_libraries(bench_refactor PRIVATE Threads::Threads)

add_executable(mcts_safety
    test/mcts_safety.cpp
    rays.cpp
    bitboard.cpp
    line_pattern.cpp
    position.cpp
    movegen.cpp
    movepicker.cpp
    mcts.cpp
    search.cpp
    evaluate.cpp
    policy_prior.cpp
    nnue/nnue.cpp
)
target_link_libraries(mcts_safety PRIVATE Threads::Threads)

cmake_minimum_required(VERSION 3.14)
project(AmazonsEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags (mirrors Stockfish's build flags)
if (MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -funroll-loops)
endif()

add_executable(amazons
    main.cpp
    rays.cpp
    bitboard.cpp
    position.cpp
    movegen.cpp
    evaluate.cpp
    movepicker.cpp
    search.cpp
    search_threads.cpp
    nnue/nnue.cpp
)

# Link threads (needed for std::thread)
find_package(Threads REQUIRED)
target_link_libraries(amazons PRIVATE Threads::Threads)

add_executable(convert_selfplay
    train/convert_selfplay.cpp
)

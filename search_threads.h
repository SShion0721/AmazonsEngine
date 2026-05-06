#pragma once

#include "position.h"
#include "search.h"
#include "tt.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class SearchThread {
public:
    explicit SearchThread(TranspositionTable& tt);
    ~SearchThread();

    void start_searching(std::shared_ptr<Position> root,
                         const TimeManager& time_manager,
                         int max_depth,
                         int thread_id,
                         uint64_t node_limit);
    void request_stop();
    void wait_for_search_finished();
    void new_game();
    void shutdown();

private:
    void idle_loop();

    Searcher searcher_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool exit_ = false;
    bool job_ready_ = false;
    bool searching_ = false;
    std::shared_ptr<Position> root_pos_;
    TimeManager time_manager_;
    int max_depth_ = 64;
    int thread_id_ = 0;
    uint64_t node_limit_ = 0;
};

class SearchThreadPool {
public:
    explicit SearchThreadPool(TranspositionTable& tt);
    ~SearchThreadPool();

    void resize(int helper_count);
    void start_searching(const Position& root,
                         const TimeManager& time_manager,
                         int max_depth,
                         uint64_t node_limit = 0);
    void request_stop();
    void wait_for_search_finished();
    void new_game();
    void shutdown();
    int size() const;

private:
    TranspositionTable& tt_;
    std::vector<std::unique_ptr<SearchThread>> threads_;
};

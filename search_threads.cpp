#include "search_threads.h"

#include <algorithm>

SearchThread::SearchThread(TranspositionTable& tt)
    : searcher_(tt), thread_(&SearchThread::idle_loop, this) {
    searcher_.silent = true;
}

SearchThread::~SearchThread() {
    shutdown();
}

void SearchThread::start_searching(std::shared_ptr<Position> root,
                                   const TimeManager& time_manager,
                                   int max_depth,
                                   int thread_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !searching_ && !job_ready_; });
    root_pos_ = std::move(root);
    time_manager_ = time_manager;
    max_depth_ = max_depth;
    thread_id_ = thread_id;
    job_ready_ = true;
    lock.unlock();
    cv_.notify_one();
}

void SearchThread::request_stop() {
    searcher_.request_stop();
}

void SearchThread::wait_for_search_finished() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !searching_ && !job_ready_; });
}

void SearchThread::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (exit_)
            return;
        exit_ = true;
    }

    searcher_.request_stop();
    cv_.notify_one();
    if (thread_.joinable())
        thread_.join();
}

void SearchThread::idle_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        cv_.wait(lock, [&] { return job_ready_ || exit_; });
        if (exit_)
            break;

        std::shared_ptr<Position> root = std::move(root_pos_);
        TimeManager time_manager = time_manager_;
        int max_depth = max_depth_;
        int thread_id = thread_id_;
        job_ready_ = false;
        searching_ = true;
        lock.unlock();

        searcher_.silent = true;
        searcher_.time_man = time_manager;
        searcher_.search(*root, max_depth, nullptr, thread_id);

        lock.lock();
        searching_ = false;
        root.reset();
        cv_.notify_all();
    }

    searching_ = false;
    job_ready_ = false;
    cv_.notify_all();
}

SearchThreadPool::SearchThreadPool(TranspositionTable& tt) : tt_(tt) {}

SearchThreadPool::~SearchThreadPool() {
    shutdown();
}

void SearchThreadPool::resize(int helper_count) {
    helper_count = std::max(0, helper_count);
    if (static_cast<int>(threads_.size()) == helper_count)
        return;

    shutdown();
    threads_.reserve(helper_count);
    for (int i = 0; i < helper_count; ++i)
        threads_.push_back(std::make_unique<SearchThread>(tt_));
}

void SearchThreadPool::start_searching(const Position& root,
                                       const TimeManager& time_manager,
                                       int max_depth) {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
        threads_[i]->start_searching(
            std::make_shared<Position>(root), time_manager, max_depth, static_cast<int>(i) + 1);
    }
}

void SearchThreadPool::request_stop() {
    for (auto& thread : threads_)
        thread->request_stop();
}

void SearchThreadPool::wait_for_search_finished() {
    for (auto& thread : threads_)
        thread->wait_for_search_finished();
}

void SearchThreadPool::shutdown() {
    for (auto& thread : threads_)
        thread->shutdown();
    threads_.clear();
}

int SearchThreadPool::size() const {
    return static_cast<int>(threads_.size());
}

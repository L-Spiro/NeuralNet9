/**
 * Copyright L. Spiro 2022
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A thread pool, providing worker threads associated with cores and results obtained with future promises.
 */

#include "NN9ThreadPool.h"
#include "../OS/NN9Os.h"

namespace nn9 {

    ThreadPool::ThreadPool( size_t _tThreads ) :
        m_bStop( false ) {
        for ( size_t i = 0; i < _tThreads; ++i ) {
            m_vWorkers.emplace_back(
                [this] {
                    for (;;) {
                        std::function<void()> fTask;

                        // Acquire lock and wait for tasks.
                        {
                            std::unique_lock<std::mutex> ulLock( this->m_mQueueMutex );
                            this->m_cvCondition.wait(
                                ulLock, [this] { return this->m_bStop || !this->m_qTasks.empty(); });

                            // Exit condition.
                            if ( this->m_bStop && this->m_qTasks.empty() ) { return; }

                            // Get the next task.
                            fTask = std::move( this->m_qTasks.front() );
                            this->m_qTasks.pop();
                        }

                        // Execute the task.
                        fTask();
                    }
                });

            // Set thread affinity to associate each thread with a different core.
            std::thread & tWorker = m_vWorkers.back();

            // Platform-specific thread affinity setting.
            SetThreadAffinity( tWorker.native_handle(), i );
        }
    }
    ThreadPool::~ThreadPool() {
        // Indicate that the pool is stopping.
        {
            std::unique_lock<std::mutex> ulLock( m_mQueueMutex );
            m_bStop = true;
        }

        // Wake up all threads to finish execution.
        m_cvCondition.notify_all();

        // Join all threads.
        for ( std::thread & tWorker : m_vWorkers ) {
            tWorker.join();
        }
    }

}	// namespace nn9

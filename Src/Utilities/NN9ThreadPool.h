/**
 * Copyright L. Spiro 2022
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A thread pool, providing worker threads associated with cores and results obtained with future promises.
 */

#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>

// Include platform-specific headers
#if defined( __APPLE__ )
#include <pthread.h>
#include <sched.h>
#endif	// #if defined( __APPLE__ )

namespace nn9 {

    /**
	 * Class ThreadPool
	 * \brief A simple thread pool implementation that supports task submission and thread affinity.
	 *
	 * Description: A thread pool, providing worker threads associated with cores and results obtained with future promises.
	 */
	class ThreadPool {
	public :
		ThreadPool( size_t _tThreads );         // Constructor: Ccreates a pool with the specified number of threads.
		~ThreadPool();                          // Destructor: Joins all threads.


		// == Functions.
		/**
		 * \brief Submits a task to the thread pool.
		 *
		 * The task is a callable object, which can be a function, lambda expression, or any other
		 * callable. The method returns a `std::future` that holds the result of the task.
		 *
		 * \tparam F The type of the callable.
		 * \tparam Args The types of the arguments to pass to the callable.
		 * \param _fF The callable to execute.
		 * \param _aArgs The arguments to pass to the callable.
		 * \return A `std::future` that will hold the result of the task.
		 * \throws std::runtime_error if the thread pool is stopped.
		 */
		template <class F, class ... Args>
		inline auto								Submit( F && _fF, Args && ... _aArgs )
			-> std::future<typename std::invoke_result_t<F, Args ...>>;


	private :
		std::vector<std::thread>                m_vWorkers;                             /**< Vector holding all worker threads. */
		std::queue<std::function<void()>>       m_qTasks;                               /**< Task queue where incoming tasks are stored. */
		std::mutex                              m_mQueueMutex;                          /**< Mutex for synchronizing access to the task queue. */
		std::condition_variable                 m_cvCondition;                          /**< Condition variable for notifying worker threads of new tasks. */
		bool                                    m_bStop;                                /**< Flag indicating whether the thread pool is stopping. */
	};


	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// DEFINITIONS
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// == Functions.
	/**
	 * \brief Submits a task to the thread pool.
	 *
	 * The task is a callable object, which can be a function, lambda expression, or any other
	 * callable. The method returns a `std::future` that holds the result of the task.
	 *
	 * \tparam F The type of the callable.
	 * \tparam Args The types of the arguments to pass to the callable.
	 * \param _fF The callable to execute.
	 * \param _aArgs The arguments to pass to the callable.
	 * \return A `std::future` that will hold the result of the task.
	 * \throws std::runtime_error if the thread pool is stopped.
	 */
	template <class F, class ... Args>
	inline auto ThreadPool::Submit( F && _fF, Args && ... _aArgs )
		-> std::future<typename std::invoke_result_t<F, Args ...>> {
		using return_type = typename std::invoke_result_t<F, Args ...>;

		// Create a packaged task.
		auto aTask = std::make_shared<std::packaged_task<return_type()>>(
			std::bind( std::forward<F>( _fF ), std::forward<Args>( _aArgs ) ... ));

		// Get the future.
		std::future<return_type> fRes = aTask->get_future();

		// Add the task to the queue.
		{
			std::unique_lock<std::mutex> ulLock( m_mQueueMutex );

			// Don't allow adding tasks after stopping the pool.
			if ( m_bStop ) {
				throw std::runtime_error( "Cannot submit to a stopped ThreadPool." );
			}

			m_qTasks.emplace( [aTask]() { (*aTask)(); } );
		}

		// Notify one waiting thread.
		m_cvCondition.notify_one();
		return fRes;
	}

}	// namespace nn9

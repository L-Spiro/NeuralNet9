/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Manages buffers.  Buffers need to be flushed to disk sometimes depending on memory constraints etc.  This class operates behind-the-scenes.
 */

#pragma once

#include "NN9Buffer.h"

#include <mutex>

// Include platform-specific headers
#if defined( __APPLE__ )
#include <pthread.h>
#include <sched.h>
#endif	// #if defined( __APPLE__ )


namespace nn9 {

	/**
	 * Class BufferManager
	 * \brief Manages buffers.
	 *
	 * Description: Manages buffers.  Buffers need to be flushed to disk sometimes depending on memory constraints etc.  This class operates behind-the-scenes.
	 */
	class BufferManager {
	public :
		static BufferManager					GblBufferManager;						/**< Behind-the-scenes buffer manager. */


	protected :
		std::mutex                              m_mQueueMutex;                          /**< Mutex for synchronizing access to the task queue. */
	};

	

}	// namespace nn9

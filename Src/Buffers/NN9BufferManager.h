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
#include <vector>

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

		BufferManager();
		~BufferManager();


		// == Functions.
		/**
		 * Creates a buffer and returns its pointer.  The buffer starts with a reference counter of 1.
		 * 
		 * \param _sSize The size of the buffer to create, in bytes.
		 * \param _prcOwner A pointer to the owning object, which the buffer will also reference.
		 * \return Returns a pointer to the created buffer.
		 **/
		Buffer *								CreateBuffer( size_t _sSize, RefCnt * _prcOwner = nullptr );

		/**
		 * Dereferences a buffer.  If the reference count reaches 0, the buffer is deleted from memory.
		 * 
		 * \param _pbBuffer The buffer to dereference and possibly delete.
		 **/
		void									DeleteBuffer( Buffer * _pbBuffer );


	protected :
		std::mutex								m_mMutex;								/**< Mutex for synchronizing access to the task queue. */
		std::vector<std::unique_ptr<Buffer>>	m_vBuffers;								/**< The buffers we manage. */
	};

}	// namespace nn9

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
		 * \param _tType The data type of the buffer to create.
		 * \param _sSize The size of the buffer to create, in bytes.
		 * \param _prcOwner A pointer to the owning object, which the buffer will also reference.
		 * \return Returns a pointer to the created buffer.
		 **/
		Buffer *								CreateBuffer( NN9_TYPE _tType, size_t _sSize, RefCnt * _prcOwner = nullptr );

		/**
		 * Dereferences a buffer.  If the reference count reaches 0, the buffer is deleted from memory.
		 * 
		 * \param _pbBuffer The buffer to dereference and possibly delete.
		 * \return Returns true if the reference count reached 0 and the buffer was deleted from memory, or if the passed-in buffer is nullptr.
		 **/
		bool									DeleteBuffer( Buffer * _pbBuffer );

		/**
		 * Adds to the allocated-memory counter.
		 * 
		 * \param _ui64Allocated The amount of memory allocated.
		 **/
		void									AddMem( uint64_t _ui64Allocated );

		/**
		 * Removes the amount from the allocated-memory counter.
		 * 
		 * \param _ui64DeAllocated The amount of memory deallocated.
		 **/
		void									DelMem( uint64_t _ui64DeAllocated );


	protected :
		std::atomic<uint64_t>					m_ui64TotalMemory = 0;					/**< The total memory consumed by the buffers we manage. */
		std::mutex								m_mMutex;								/**< Mutex for synchronizing access to the task queue. */
		std::vector<std::unique_ptr<Buffer>>	m_vBuffers;								/**< The buffers we manage. */
	};

}	// namespace nn9

/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Manages buffers.  Buffers need to be flushed to disk sometimes depending on memory constraints etc.  This class operates behind-the-scenes.
 */
 
#include "NN9BufferManager.h"

#include <iomanip>
#include <iostream>


namespace nn9 {
	
	// == Members.
	BufferManager BufferManager::GblBufferManager;								/**< Behind-the-scenes buffer manager. */

	// == Functions.
	BufferManager::BufferManager() {
	}
	BufferManager::~BufferManager() {
		{
			std::unique_lock<std::mutex> ulLock( m_mMutex );
			if ( m_vBuffers.size() ) {
				std::wcout << L"BufferManager Warning: " << m_vBuffers.size() << L" unreleased buffers." << std::endl;
			}
			m_vBuffers = std::vector<std::unique_ptr<Buffer>>();
		}
	}

	// == Functions.
	/**
	 * Creates a buffer and returns its pointer.  The buffer starts with a reference counter of 1.
	 * 
	 * \param _tType The data type of the buffer to create.
	 * \param _sSize The size of the buffer to create, in bytes.
	 * \param _prcOwner A pointer to the owning object, which the buffer will also reference.
	 * \return Returns a pointer to the created buffer.
	 **/
	Buffer * BufferManager::CreateBuffer( NN9_TYPE _tType, size_t _sSize, RefCnt * _prcOwner ) {
		Buffer * pbBuffer = nullptr;
		{
			std::unique_lock<std::mutex> ulLock( m_mMutex );
			try {
				m_vBuffers.emplace_back( std::make_unique<Buffer>( _tType, _sSize, _prcOwner ) );
				pbBuffer = m_vBuffers.back().get();
				if ( pbBuffer ) {
					pbBuffer->IncRef();
				}
				else {
					throw std::runtime_error( "BufferManager::CreateBuffer: Failed to create Buffer." );
				}
			}
			catch ( ... ) {
				throw std::runtime_error( "BufferManager::CreateBuffer: Failed to create Buffer." );
			}
		}
		return pbBuffer;
	}

	/**
	 * Dereferences a buffer.  If the reference count reaches 0, the buffer is deleted from memory.
	 * 
	 * \param _pbBuffer The buffer to dereference and possibly delete.
	 * \return Returns true if the reference count reached 0 and the buffer was deleted from memory, or if the passed-in buffer is nullptr.
	 **/
	bool BufferManager::DeleteBuffer( Buffer * _pbBuffer ) {
		if ( !_pbBuffer ) { return true; }
		{
			std::unique_lock<std::mutex> ulLock( m_mMutex );
			for ( auto I = m_vBuffers.size(); I--; ) {
				if ( m_vBuffers[I].get() == _pbBuffer ) {
					if ( _pbBuffer->DecRef() == 0 ) {
						m_vBuffers.erase( m_vBuffers.begin() + I );
						return true;
					}
					return false;
				}
			}
			std::wcout << L"BufferManager Warning: Buffer 0x" << std::uppercase << std::hex << std::setfill( L'0' ) << std::setw( 8 ) << _pbBuffer << L" not found." << std::endl;
			return false;
		}
	}

	/**
	 * Adds to the allocated-memory counter.
	 * 
	 * \param _ui64Allocated The amount of memory allocated.
	 **/
	void BufferManager::AddMem( uint64_t _ui64Allocated ) {
		m_ui64TotalMemory += _ui64Allocated;
	}

	/**
	 * Removes the amount from the allocated-memory counter.
	 * 
	 * \param _ui64DeAllocated The amount of memory deallocated.
	 **/
	void BufferManager::DelMem( uint64_t _ui64DeAllocated ) {
		m_ui64TotalMemory -= _ui64DeAllocated;
	}

}	// namespace nn9

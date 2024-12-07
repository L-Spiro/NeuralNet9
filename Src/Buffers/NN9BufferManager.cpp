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
			for ( auto I = m_vBuffers.size(); I--; ) {
				m_vBuffers[I].reset();
			}
		}
	}

	// == Functions.
	/**
	 * Creates a buffer and returns its pointer.  The buffer starts with a reference counter of 1.
	 * 
	 * \param _sSize The size of the buffer to create, in bytes.
	 * \param _prcOwner A pointer to the owning object, which the buffer will also reference.
	 * \return Returns a pointer to the created buffer.
	 **/
	Buffer * BufferManager::CreateBuffer( size_t _sSize, RefCnt * _prcOwner ) {
		Buffer * pbBuffer = nullptr;
		{
			std::unique_lock<std::mutex> ulLock( m_mMutex );
			try {
				m_vBuffers.emplace_back( std::make_unique<Buffer>( _sSize, _prcOwner ) );
				pbBuffer = m_vBuffers.back().get();
				if ( pbBuffer ) {
					pbBuffer->IncRef();
				}
			}
			catch ( ... ) {
				throw std::runtime_error( "BufferManager failed to create Buffer." );
			}
		}
		return pbBuffer;
	}

	/**
	 * Dereferences a buffer.  If the reference count reaches 0, the buffer is deleted from memory.
	 * 
	 * \param _pbBuffer The buffer to dereference and possibly delete.
	 **/
	void BufferManager::DeleteBuffer( Buffer * _pbBuffer ) {
		if ( !_pbBuffer ) { return; }
		{
			std::unique_lock<std::mutex> ulLock( m_mMutex );
			for ( auto I = m_vBuffers.size(); I--; ) {
				if ( m_vBuffers[I].get() == _pbBuffer ) {
					if ( _pbBuffer->DecRef() == 0 ) {
						m_vBuffers.erase( m_vBuffers.begin() + I );
					}
					return;
				}
			}
			std::wcout << L"BufferManager Warning: Buffer 0x" << std::uppercase << std::hex << std::setfill( L'0' ) << std::setw( 8 ) << _pbBuffer << L" not found." << std::endl;
		}
	}

 }	// namespace nn9

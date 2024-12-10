/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A buffer can be interpreted as any kind of data and be flushed to disk.  They maintain reference counts for sharing and can be fully
 *	or partially mapped to memory.  Buffers are always aligned in memory to 64 bytes.
 */
 
 #include "NN9Buffer.h"
 #include "NN9BufferManager.h"

 namespace nn9 {

	Buffer::Buffer( NN9_TYPE _tType, size_t _sSize, RefCnt * _rcOwner ) :
		m_tType( _tType ),
		m_vBuffer( Types::SizeOf( _tType ) * _sSize ),
		m_prcOwner( _rcOwner ) {
		BufferManager::GblBufferManager.AddMem( MemUsed() );
		//if ( m_prcOwner ) { m_prcOwner->IncRef(); }
	}
	Buffer::~Buffer() {
		BufferManager::GblBufferManager.DelMem( MemUsed() );
		//if ( m_prcOwner ) { m_prcOwner->DecRef(); }
	}

 }	// namespace nn9

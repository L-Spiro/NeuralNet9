/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A buffer can be interpreted as any kind of data and be flushed to disk.  They maintain reference counts for sharing and can be fully
 *	or partially mapped to memory.  Buffers are always aligned in memory to 64 bytes.
 */

#pragma once

#include "../Foundation/NN9AlignmentAllocator.h"
#include "../Foundation/NN9RefCnt.h"
#include "../Types/NN9Types.h"

#include <vector>


namespace nn9 {

	class BufferManager;

	/**
	 * Class Buffer
	 * \brief A buffer can be interpreted as any kind of data and be flushed to disk.
	 *
	 * Description: A buffer can be interpreted as any kind of data and be flushed to disk.  They maintain reference counts for sharing and can be fully
	 *	or partially mapped to memory.  Buffers are always aligned in memory to 64 bytes.
	 */
	class Buffer : public RefCnt {
	public :
		Buffer( size_t _sSize ) :
			m_vBuffer( _sSize ) {
		}
		Buffer( size_t _sSize, RefCnt * _rcOwner ) :
			m_vBuffer( _sSize ),
			m_prcOwner( _rcOwner ) {
		}


		// == Functions.
		/**
		 * Increases the reference count.
		 **/
		virtual void																IncRef() {
			Parent::IncRef();
			if ( m_prcOwner ) { m_prcOwner->IncRef(); }
		}

		/**
		 * Decreases the reference count.
		 * 
		 * \return Returns the reference count.
		 **/
		virtual int32_t																DecRef() {
			auto aRet = Parent::DecRef();
			if ( m_prcOwner ) { m_prcOwner->DecRef(); }
			
			return aRet;
		}

		/**
		 * S
		 * 
		 * \param PARM DESC
		 * \param PARM DESC
		 * \return DESC
		 **/


	protected :
		// == Members.
		std::vector<uint8_t, AlignmentAllocator<uint8_t, 64>>						m_vBuffer;					/**< The actual data buffer. */
		RefCnt *																	m_prcOwner = nullptr;		/**< An optional pointer to an owning object which also needs to be reference-counted when this one is. */


	private :
		typedef RefCnt																Parent;
	};

}	// namespace nn9

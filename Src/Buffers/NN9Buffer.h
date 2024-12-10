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
#include "../Tensor/NN9View.h"
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
		Buffer( NN9_TYPE _tType, size_t _sSize, RefCnt * _rcOwner = nullptr );
		~Buffer();


		// == Types.
		typedef std::vector<uint8_t, AlignmentAllocator<uint8_t, 64>>				BufferType;


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
		 * Gets the total amount of memory allocated by the buffer.
		 * 
		 * \return Returns the total amount of memory allocated by the buffer.
		 **/
		size_t																		MemUsed() const { return m_vBuffer.capacity() * sizeof( BufferType::value_type ); }

		/**
		 * Gets the total number of bytes referenced by the buffer.
		 * 
		 * \return Returns the total number of bytes referenced by the buffer.
		 **/
		size_t																		Size() const { return m_vBuffer.size() * sizeof( BufferType::value_type ); }

		/**
		 * Gets the total number of elements when interpreted as the given templated type.
		 * 
		 * \return Returns the total number of elements when interpreted as the given templated type.
		 **/
		template <typename _tType>
		size_t																		Size() const { return m_vBuffer.size() * sizeof( BufferType::value_type ) / sizeof( _tType ); }

		/**
		 * Gets the total number of elements when interpreted as the given type.
		 * 
		 * \param _tType The type as which to reinterpret the buffer.
		 * \return Returns the total number of elements when interpreted as the given type.
		 **/
		size_t																		Size( NN9_TYPE _tType ) const { return m_vBuffer.size() * sizeof( BufferType::value_type ) / Types::SizeOf( _tType ); }

		/**
		 * Gets a view to the whole buffer.
		 * 
		 * \tparam _tType The type to which to interpret the buffer.
		 * \return Returns a view to the entire buffer interpreted as the given type.
		 **/
		template <typename _tType>
		View<_tType>																FullView() {
			return View<_tType>( reinterpret_cast<_tType *>(m_vBuffer.data()), Size<_tType>(), this );
		}

		/**
		 * Gets a view of a part of the buffer.  If _DEBUG, throws if any part of the requested range is out of range of the buffer.
		 * 
		 * \param _sStart the starting index for the view.
		 * \param _sTotal The total number of elements to map into the view.
		 * \throw If _DEBUG, it will throw if the requested range extends beyond the valid buffer range.
		 * \return Returns a view of the given range within the buffer.
		 **/
		template <typename _tType>
		View<_tType>																RangeView( size_t _sStart, size_t _sTotal ) {
#ifdef _DEBUG
			if ( _sStart + _sTotal > Size<_tType>() ) {
				throw std::out_of_range( "Buffer::RangeView: Range is out of bounds." );
			}
#endif	// #ifdef _DEBUG
			return View<_tType>( reinterpret_cast<_tType *>(m_vBuffer.data()) + _sStart, _sTotal, this );
		}


	protected :
		// == Members.
		BufferType																	m_vBuffer;					/**< The actual data buffer. */
		RefCnt *																	m_prcOwner = nullptr;		/**< An optional pointer to an owning object which also needs to be reference-counted when this one is. */
		NN9_TYPE																	m_tType = NN9_T_FLOAT;		/**< the buffer data type. */


	private :
		typedef RefCnt																Parent;
	};

}	// namespace nn9

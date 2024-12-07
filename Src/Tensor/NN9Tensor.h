/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A tensor.
 */

#pragma once

#include "../Buffers/NN9Buffer.h"
#include "../Buffers/NN9BufferManager.h"
#include "../Foundation/NN9RefCnt.h"

#include <cstdarg>
#include <initializer_list>
#include <vector>


namespace nn9 {

	/**
	 * Class Tensor
	 * \brief A tensor.
	 *
	 * Description: A tensor.
	 */
	class Tensor : public RefCnt {
	public :
		Tensor( std::initializer_list<size_t> _ilShape, NN9_TYPE _tType ) :
			m_vShape( _ilShape ) {
			if ( m_vShape.size() == 0 ) {
				throw std::invalid_argument( "There must be at least 1 dimension." );
			}
			size_t sTotalSize = 1;
			for ( size_t sDim : m_vShape ) {
				sTotalSize *= sDim;
			}

			m_pfIndexFunc = m_pfAtTable[std::min(m_vShape.size(),size_t(5))-1];
			CalculateStrides();
			m_pbBuffer = BufferManager::GblBufferManager.CreateBuffer( Types::SizeOf( _tType ), this );
		}
		template <typename _tInitType>
		Tensor( std::initializer_list<size_t> _ilShape, NN9_TYPE _tType, _tInitType _tInitValue ) :
			m_vShape( _ilShape ) {
			if ( m_vShape.size() == 0 ) {
				throw std::invalid_argument( "There must be at least 1 dimension." );
			}
			size_t sTotalSize = 1;
			for ( size_t sDim : m_vShape ) {
				sTotalSize *= sDim;
			}

			m_pfIndexFunc = m_pfAtTable[std::min(m_vShape.size(),size_t(5))-1];
			CalculateStrides();
			/*data_.resize(sTotalSize, _tInitValue);
			calculateStrides();*/
		}
		~Tensor();


		// == Functions.
		/**
		 * Converts multidimensional array indices into a flat indix.
		 * 
		 * \param _aArgs The indices, one for each dimension of the tensor.
		 * \return Returns the flat index into the buffer where the given item can be found.
		 **/
		template <typename ... Arg>
		size_t											Flat( Arg ... _aArgs ) const {
			constexpr size_t sNumArgs = sizeof...( _aArgs );

			if ( sNumArgs != m_vShape.size() ) {
				throw std::invalid_argument( "Number of arguments does not match tensor dimensions." );
			}

			return m_pfIndexFunc( this, _aArgs ... );
		}



	protected :
		// == Types.
		using											PfGetIndexFunc = size_t(*)( const Tensor *, size_t, size_t, size_t, size_t, ... );
		

		// == Members.
		std::vector<size_t>								m_vShape;						/**< Shape of the tensor  */
		std::vector<size_t>								m_vStride;						/**< Strides for each dimension. */
		PfGetIndexFunc									m_pfIndexFunc = GetIndex1D;		/**< Calculates indices by breaking them down into special cases and providing specialized functions for those cases. */
		Buffer *										m_pbBuffer = nullptr;			/**< Pointer to the reference-counted buffer. */

		static PfGetIndexFunc							m_pfAtTable[5];					/**< The table of GetIndex*() functions. */


		// == Functions.
		/**
		 * Calculates the stride table.
		 **/
		void											CalculateStrides();

		/**
		 * The function for handling 1-D indexing.
		 * 
		 * \param _ptTensor The tensor object.
		 * \param _I The first index.
		 * \return Returns the offset into the buffer via 1-D indexing.
		 **/
		static size_t									GetIndex1D( const Tensor * _ptTensor, size_t _I, size_t, size_t, size_t, ... ) {
			return _I * _ptTensor->m_vStride[0];
		}

		/**
		 * The function for handling 2-D indexing.
		 * 
		 * \param _ptTensor The tensor object.
		 * \param _I The first index.
		 * \param _J The second index.
		 * \return Returns the offset into the buffer via 2-D indexing.
		 **/
		static size_t									GetIndex2D( const Tensor * _ptTensor, size_t _I, size_t _J, size_t, size_t, ... ) {
			return _I * _ptTensor->m_vStride[0] + _J * _ptTensor->m_vStride[1];
		}

		/**
		 * The function for handling 3-D indexing.
		 * 
		 * \param _ptTensor The tensor object.
		 * \param _I The first index.
		 * \param _J The second index.
		 * \param _K The third index.
		 * \return Returns the offset into the buffer via 3-D indexing.
		 **/
		static size_t									GetIndex3D( const Tensor * _ptTensor, size_t _I, size_t _J, size_t _K, size_t, ... ) {
			return _I * _ptTensor->m_vStride[0] + _J * _ptTensor->m_vStride[1] + _K * _ptTensor->m_vStride[2];
		}

		/**
		 * The function for handling 4-D indexing.
		 * 
		 * \param _ptTensor The tensor object.
		 * \param _I The first index.
		 * \param _J The second index.
		 * \param _K The third index.
		 * \param _L The fourth index.
		 * \return Returns the offset into the buffer via 4-D indexing.
		 **/
		static size_t									GetIndex4D( const Tensor * _ptTensor, size_t _I, size_t _J, size_t _K, size_t _L, ... ) {
			return _I * _ptTensor->m_vStride[0] + _J * _ptTensor->m_vStride[1] + _K * _ptTensor->m_vStride[2] + _L * _ptTensor->m_vStride[3];
		}

		/**
		 * The function for handling X-D indexing.
		 * 
		 * \param _ptTensor The tensor object.
		 * \param _I The first index.
		 * \param _J The second index.
		 * \param _K The third index.
		 * \param _L The fourth index.
		 * \return Returns the offset into the buffer via X-D indexing.
		 **/
		static size_t									GetIndexXD( const Tensor * _ptTensor, size_t _I, size_t _J, size_t _K, size_t _L, ... ) {
			size_t sIdx = _I * _ptTensor->m_vStride[0] + _J * _ptTensor->m_vStride[1] + _K * _ptTensor->m_vStride[2] + _L * _ptTensor->m_vStride[3];

			va_list vaArgs;
            va_start( vaArgs, _L );
            for ( size_t D = 4; D < _ptTensor->m_vStride.size(); ++D ) {
                size_t sThis = va_arg( vaArgs, size_t );
                sIdx += sThis * _ptTensor->m_vStride[D];
            }
            va_end( vaArgs );
			return sIdx;
		}
	};

}	// namespace nn9

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
				throw std::invalid_argument( "Tensor: There must be at least 1 dimension." );
			}
			m_sSize = 1;
			for ( size_t sDim : m_vShape ) {
				m_sSize *= sDim;
			}

			CalculateStrides();
			m_pbBuffer = BufferManager::GblBufferManager.CreateBuffer( _tType, m_sSize, this );
		}
		template <typename _tInitType>
		Tensor( std::initializer_list<size_t> _ilShape, NN9_TYPE _tType, _tInitType _tInitValue ) :
			m_vShape( _ilShape ) {
			if ( m_vShape.size() == 0 ) {
				throw std::invalid_argument( "Tensor: There must be at least 1 dimension." );
			}
			m_sSize = 1;
			for ( size_t sDim : m_vShape ) {
				m_sSize *= sDim;
			}

			CalculateStrides();
			m_pbBuffer = BufferManager::GblBufferManager.CreateBuffer( _tType, m_sSize, this );

#define NN9_SET( CASE, TYPE )												\
	case CASE : {															\
		auto aView = m_pbBuffer->FullView<TYPE>();							\
		auto aSize = aView.size();											\
		for ( size_t I = 0; I < aSize; ++I ) {								\
			aView[I] = static_cast<TYPE>(_tInitValue);						\
		}																	\
		break;																\
	}
			switch ( _tType ) {
				NN9_SET( NN9_T_BFLOAT16, bfloat16_t )
				NN9_SET( NN9_T_FLOAT16, nn9::float16 )
				NN9_SET( NN9_T_FLOAT, float )
				NN9_SET( NN9_T_DOUBLE, double )

				NN9_SET( NN9_T_UINT8, uint8_t )
				NN9_SET( NN9_T_UINT16, uint16_t )
				NN9_SET( NN9_T_UINT32, uint32_t )
				NN9_SET( NN9_T_UINT64, uint64_t )

				NN9_SET( NN9_T_INT8, int8_t )
				NN9_SET( NN9_T_INT16, int16_t )
				NN9_SET( NN9_T_INT32, int32_t )
				NN9_SET( NN9_T_INT64, int64_t )

				NN9_SET( NN9_T_BOOL, bool )

				NN9_SET( NN9_T_COMPLEX64, std::complex<float> )
				NN9_SET( NN9_T_COMPLEX128, std::complex<double> )
			}
#undef NN9_SET
		}
		~Tensor();


		// == Functions.
		/**
		 * Converts multidimensional array indices into a flat indix.
		 * 
		 * \param _aArgs The indices, one for each dimension of the tensor.
		 * \throw If _DEBUG, it will throw if the number of arguments does not match the tensor shape, or if the flattened index is out of range.
		 * \return Returns the flat index into the buffer where the given item can be found.
		 **/
		template <typename ... Arg>
		size_t											Flat( Arg ... _aArgs ) const {
			constexpr size_t sNumArgs = sizeof...( _aArgs );
#ifdef _DEBUG
			if ( sNumArgs != m_vShape.size() ) {
				throw std::invalid_argument( "Tensor::Flat: Number of arguments does not match tensor dimensions." );
			}
#endif	// #ifdef _DEBUG

			size_t sRet = 0;
			size_t strideIdx = 0;
			// Fold expression to accumulate the stride-based calculation.
			((sRet += m_vStride[strideIdx++] * _aArgs), ...);
#ifdef _DEBUG
			if ( sRet >= m_sSize ) {
				throw std::out_of_range( "Tensor::Flat: Index out of range." );
			}
#endif	// #ifdef _DEBUG
			return sRet;
		}

		/**
		 * Converts multidimensional array indices into a flat indix.
		 * 
		 * \param _aArgs The indices, one for each dimension of the tensor.
		 * \throw If _DEBUG, it will throw if the number of arguments does not match the tensor shape, or if the flattened index is out of range.
		 * \return Returns the flat index into the buffer where the given item can be found.
		 **/
		template <typename _tArgType>
		size_t											Flat( const _tArgType &_aArgs ) const {
			size_t sNumArgs = _aArgs.size();
#ifdef _DEBUG
			if ( sNumArgs != m_vShape.size() ) {
				throw std::invalid_argument( "Tensor::Flat: Number of arguments does not match tensor dimensions." );
			}
#endif	// #ifdef _DEBUG

			size_t sRet = 0;
			size_t strideIdx = 0;
			// Fold expression to accumulate the stride-based calculation.
			for ( size_t I = 0; I < sNumArgs; ++I ) {
				sRet += m_vStride[I] * _aArgs[I];
			}
#ifdef _DEBUG
			if ( sRet >= m_sSize ) {
				throw std::out_of_range( "Tensor::Flat: Index out of range." );
			}
#endif	// #ifdef _DEBUG
			return sRet;
		}

		/**
		 * Gets the total number of bytes referenced by the buffer.
		 * 
		 * \return Returns the total number of bytes referenced by the buffer.
		 **/
		size_t											Size() const { return m_pbBuffer->Size(); }

		/**
		 * Gets the total number of elements when interpreted as the given templated type.
		 * 
		 * \return Returns the total number of elements when interpreted as the given templated type.
		 **/
		template <typename _tType>
		size_t											Size() const { return m_pbBuffer->Size<_tType>(); }

		/**
		 * Gets the total number of elements when interpreted as the given type.
		 * 
		 * \param _tType The type as which to reinterpret the buffer.
		 * \return Returns the total number of elements when interpreted as the given type.
		 **/
		size_t											Size( NN9_TYPE _tType ) const { return m_pbBuffer->Size( _tType ); }

		/**
		 * Gets a view to the whole buffer.
		 * 
		 * \tparam _tType The type to which to interpret the buffer.
		 * \return Returns a view to the entire buffer interpreted as the given type.
		 **/
		template <typename _tType>
		View<_tType>									FullView() {
			return m_pbBuffer->FullView<_tType>();
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
		View<_tType>									RangeView( size_t _sStart, size_t _sTotal ) {
			return m_pbBuffer->RangeView<_tType>( _sStart, _sTotal );
		}

		/**
		 * For a 1-D tensor, puts the full buffer view into a single vector entry.  For 2-D tensors, such that X is the flat first dimension
		 *	and Y is the 2nd dimension, puts the full buffer into views spread across vector entries.  The number of views will be the Y
		 *	dimension and each view will have X items.
		 * 
		 * \throw If _DEBUG it will throw if the tensor shape is not 1-D or 2-D.  Also throws if _tType is a larger type than the tensor holds,
		 *	as this will always cause a buffer overrun.
		 * \return Returns an array of views, one view per 2nd dimension, each view with as many items as the 1st dimension.
		 **/
		template <typename _tType>
		std::vector<View<_tType>>						Full2dView() {
#ifdef _DEBUG
			if ( m_vShape.size() > 2 || m_vShape.size() == 0 ) {
				throw std::invalid_argument( "Tensor::Full2dView: Tensor must be either 1-D or 2-D." );
			}
#endif	// #ifdef _DEBUG
			std::vector<View<_tType>> vRet;
			if ( m_vShape.size() == 1 ) {
				vRet.reserve( 1 );
				// A 1-D buffer can be treated as a 2-D buffer with a single outer dimension.
				vRet.emplace_back( FullView<_tType>() );
				return vRet;
			}
			size_t sSize = m_vShape[m_vShape.size()-2];
			
			vRet.reserve( sSize );
			size_t s1dSize = m_vShape[m_vShape.size()-1];
			for ( size_t I = 0; I < sSize; ++I ) {
				vRet.emplace_back( RangeView<_tType>( Flat( I, 0 ), s1dSize ) );
			}

			return vRet;
		}

		/**
		 * For a 1-D tensor, puts the full buffer view into a single vector entry.  For 3-D tensors, such that X is the flat first dimension
		 *	and Y is the 2nd dimension, and Z is the 3rd dimension, puts the full buffer into views spread across vector entries.  The number
		 *	of views will be the Z dimension with each of the Y dimensions and each view will have X items.
		 * 
		 * \throw If _DEBUG it will throw if the tensor shape is not 1-D or 3-D.  Also throws if _tType is a larger type than the tensor holds,
		 *	as this will always cause a buffer overrun.
		 * \return Returns an array of views, one view per 2nd dimension, each view with as many items as the 1st dimension.
		 **/
		template <typename _tType>
		std::vector<std::vector<View<_tType>>>			Full3dView() {
#ifdef _DEBUG
			if ( m_vShape.size() != 3 && m_vShape.size() != 1 ) {
				throw std::invalid_argument( "Tensor::Full2dView: Tensor must be either 1-D or 3-D." );
			}
#endif	// #ifdef _DEBUG
			std::vector<std::vector<View<_tType>>> vRet;
			if ( m_vShape.size() == 1 ) {
				std::vector<View<_tType>> v2d;
				v2d.reserve( 1 );
				v2d.emplace_back( FullView<_tType>() );
				vRet.reserve( 1 );
				// A 1-D buffer can be treated as a 3-D buffer with a single outer dimension.
				vRet.emplace_back( std::move( v2d ) );
				return vRet;
			}
			size_t s3dSize = m_vShape[m_vShape.size()-3];
			size_t s2dSize = m_vShape[m_vShape.size()-2];
			size_t s1dSize = m_vShape[m_vShape.size()-1];

			vRet.reserve( s3dSize );
			
			for ( size_t I = 0; I < s3dSize; ++I ) {
				std::vector<View<_tType>> v2d;
				v2d.reserve( s2dSize );
				for ( size_t J = 0; J < s2dSize; ++J ) {
					v2d.emplace_back( RangeView<_tType>( Flat( I, J, 0 ), s1dSize ) );
				}
				vRet.emplace_back( std::move( v2d ) );
			}

			return vRet;
		}



	protected :
		// == Members.
		std::vector<size_t>								m_vShape;						/**< Shape of the tensor. */
		std::vector<size_t>								m_vStride;						/**< Strides for each dimension. */
		Buffer *										m_pbBuffer = nullptr;			/**< Pointer to the reference-counted buffer. */
		size_t											m_sSize = 0;					/**< The total size of the buffer, in elements. */


		// == Functions.
		/**
		 * Calculates the stride table.
		 **/
		void											CalculateStrides();
	};

}	// namespace nn9

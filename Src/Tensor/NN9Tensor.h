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
		Tensor( Tensor && _tOther ) noexcept {
			if ( this != &_tOther ) {
				m_vShape = std::move( _tOther.m_vShape );
				m_vStride = std::move( _tOther.m_vStride );
				m_pbBuffer = _tOther.m_pbBuffer;
				m_sSize = _tOther.m_sSize;
				m_dQuantizeScale = _tOther.m_dQuantizeScale;
				m_dQuantizeZero = _tOther.m_dQuantizeZero;
				int32_t i32Cnt = _tOther.m_aCnt;
				m_aCnt = i32Cnt;

				_tOther.m_pbBuffer = nullptr;
				_tOther.m_sSize = 0;
				_tOther.m_dQuantizeScale = 1.0;
				_tOther.m_dQuantizeZero = 0.0;
				_tOther.m_aCnt = 0;
			}
		}
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
		 * Converts multidimensional array indices into a flat index.
		 * 
		 * \param _aArgs The indices, one for each dimension of the tensor.
		 * \throw If NN9_SAFETY_CHECK, it will throw if the number of arguments does not match the tensor shape, or if the flattened index is out of range.
		 * \return Returns the flat index into the buffer where the given item can be found.
		 **/
		template <typename ... Arg>
		size_t											Flat( Arg ... _aArgs ) const {
			constexpr size_t sNumArgs = sizeof...( _aArgs );
#ifdef NN9_SAFETY_CHECK
			if ( sNumArgs != m_vShape.size() ) {
				throw std::invalid_argument( "Tensor::Flat: Number of arguments does not match tensor dimensions." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK

			size_t sRet = 0;
			size_t strideIdx = 0;
			// Fold expression to accumulate the stride-based calculation.
			((sRet += m_vStride[strideIdx++] * _aArgs), ...);
#ifdef NN9_SAFETY_CHECK
			if ( sRet >= m_sSize ) {
				throw std::out_of_range( "Tensor::Flat: Index out of range." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK
			return sRet;
		}

		/**
		 * Converts multidimensional array indices into a flat index.
		 * 
		 * \param _aArgs The indices, one for each dimension of the tensor.
		 * \throw If NN9_SAFETY_CHECK, it will throw if the number of arguments does not match the tensor shape, or if the flattened index is out of range.
		 * \return Returns the flat index into the buffer where the given item can be found.
		 **/
		template <typename _tArgType>
		size_t											Flat( const _tArgType &_aArgs ) const {
			size_t sNumArgs = _aArgs.size();
#ifdef NN9_SAFETY_CHECK
			if ( sNumArgs != m_vShape.size() ) {
				throw std::invalid_argument( "Tensor::Flat: Number of arguments does not match tensor dimensions." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK

			size_t sRet = 0;
			size_t strideIdx = 0;
			// Fold expression to accumulate the stride-based calculation.
			for ( size_t I = 0; I < sNumArgs; ++I ) {
				sRet += m_vStride[I] * _aArgs[I];
			}
#ifdef NN9_SAFETY_CHECK
			if ( sRet >= m_sSize ) {
				throw std::out_of_range( "Tensor::Flat: Index out of range." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK
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
		 * Gets the size of a dimension by index.  For a 1-D buffer, index must be 0 and the return will be the total number of elements in the
		 *	buffer as interpreted as the original buffer type.  Example using a 3-D buffer with shape { 60000, 28, 1 }: DimSize(0)=1,
		 *	DimSize(1)=28, and DimSize(2)=60000.
		 * 
		 * \param _sIdx The index of the dimension whose size is to be gotten.
		 * \return Returns the size of the dimension (shape) by index.
		 **/
		size_t											DimSize( size_t _sIdx ) const { return m_vShape[_sIdx]; }

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
		 * Gets a view of a part of the buffer.  If NN9_SAFETY_CHECK, throws if any part of the requested range is out of range of the buffer.
		 * 
		 * \param _sStart the starting index for the view.
		 * \param _sTotal The total number of elements to map into the view.
		 * \throw If NN9_SAFETY_CHECK, it will throw if the requested range extends beyond the valid buffer range.
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
		 * \throw If NN9_SAFETY_CHECK it will throw if the tensor shape is not 1-D or 2-D.  Also throws if _tType is a larger type than the tensor holds,
		 *	as this will always cause a buffer overrun.
		 * \return Returns an array of views, one view per 2nd dimension, each view with as many items as the 1st dimension.
		 **/
		template <typename _tType>
		std::vector<View<_tType>>						Full2dView() {
#ifdef NN9_SAFETY_CHECK
			if ( m_vShape.size() > 2 || m_vShape.size() == 0 ) {
				throw std::invalid_argument( "Tensor::Full2dView: Tensor must be either 1-D or 2-D." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK
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
		 * For a 1-D tensor, puts the full buffer view into nested vector entries.  For 3-D tensors, such that X is the flat first dimension
		 *	and Y is the 2nd dimension, and Z is the 3rd dimension, puts the full buffer into views spread across vector entries.  The number
		 *	of views will be the Z dimension with each of the Y dimensions and each view will have X items.
		 * 
		 * \throw If NN9_SAFETY_CHECK it will throw if the tensor shape is not 1-D or 3-D.  Also throws if _tType is a larger type than the tensor holds,
		 *	as this will always cause a buffer overrun.
		 * \return Returns an array of arrays of views, each final vector containing views with as many items as the 1st dimension.
		 **/
		template <typename _tType>
		std::vector<std::vector<View<_tType>>>			Full3dView() {
#ifdef NN9_SAFETY_CHECK
			if ( m_vShape.size() != 3 && m_vShape.size() != 1 ) {
				throw std::invalid_argument( "Tensor::Full2dView: Tensor must be either 1-D or 3-D." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK
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

		/**
		 * For a 1-D tensor, puts the full buffer view into nested vector entries.  For 4-D tensors, such that X is the flat first dimension
		 *	and Y is the 2nd dimension, Z is the 3rd dimension, and W is the 4th dimension, puts the full buffer into views spread across vector
		 *	entries.  The number of views will be the W dimension each with Z dimension with each of the Y dimensions and each view will have X items.
		 * 
		 * \throw If NN9_SAFETY_CHECK it will throw if the tensor shape is not 1-D or 4-D.  Also throws if _tType is a larger type than the tensor holds,
		 *	as this will always cause a buffer overrun.
		 * \return Returns an array of arrays of arrays of views, each final vector containing views with as many items as the 1st dimension.
		 **/
		template <typename _tType>
		std::vector<std::vector<std::vector<View<_tType>>>>
														Full4dView() {
#ifdef NN9_SAFETY_CHECK
			if ( m_vShape.size() != 4 && m_vShape.size() != 1 ) {
				throw std::invalid_argument( "Tensor::Full2dView: Tensor must be either 1-D or 4-D." );
			}
#endif	// #ifdef NN9_SAFETY_CHECK
			std::vector<std::vector<std::vector<View<_tType>>>> vRet;
			if ( m_vShape.size() == 1 ) {
				std::vector<View<_tType>> v2d;
				v2d.reserve( 1 );
				v2d.emplace_back( FullView<_tType>() );

				std::vector<std::vector<View<_tType>>> v3d;
				v3d.reserve( 1 );
				v3d.emplace_back( std::move( v2d ) );

				vRet.reserve( 1 );
				// A 1-D buffer can be treated as a 4-D buffer with a single outer dimension.
				vRet.emplace_back( std::move( v3d ) );
				return vRet;
			}
			size_t s4dSize = m_vShape[m_vShape.size()-4];
			size_t s3dSize = m_vShape[m_vShape.size()-3];
			size_t s2dSize = m_vShape[m_vShape.size()-2];
			size_t s1dSize = m_vShape[m_vShape.size()-1];

			vRet.reserve( s4dSize );
			
			for ( size_t K = 0; K < s4dSize; ++K ) {
				std::vector<std::vector<View<_tType>>> v3d;
				v3d.reserve( s3dSize );
				for ( size_t I = 0; I < s3dSize; ++I ) {
					std::vector<View<_tType>> v2d;
					v2d.reserve( s2dSize );
					for ( size_t J = 0; J < s2dSize; ++J ) {
						v2d.emplace_back( RangeView<_tType>( Flat( K, I, J, 0 ), s1dSize ) );
					}
					v3d.emplace_back( std::move( v2d ) );
				}
				vRet.emplace_back( std::move( v3d ) );
			}

			return vRet;
		}

		/**
		 * Copies the tensor, maintaining its shape, but optionally changing its underlying data type.
		 * 
		 * \param _tNewType The new type of the tensor to create.
		 * \return Returns a copy of this tensor with each value in the tensor being converted to the type specified by _tNewType.
		 **/
		Tensor											CopyAs( NN9_TYPE _tNewType ) const;


	protected :
		// == Members.
		double											m_dQuantizeScale = 1.0;			/**< Quantize scale. */
		double											m_dQuantizeZero = 0.0;			/**< Quantize 0 point. */
		std::vector<size_t>								m_vShape;						/**< Shape of the tensor. */
		std::vector<size_t>								m_vStride;						/**< Strides for each dimension. */
		Buffer *										m_pbBuffer = nullptr;			/**< Pointer to the reference-counted buffer. */
		size_t											m_sSize = 0;					/**< The total size of the buffer, in elements. */


		// == Constructors.
		Tensor( const std::vector<size_t> &_vShape, const std::vector<size_t> &_vStride, NN9_TYPE _tType,
			double _dQuantizeScale, double _dQuantizeZero );
			

		// == Functions.
		/**
		 * Calculates the stride table.
		 **/
		void											CalculateStrides();
	};

}	// namespace nn9

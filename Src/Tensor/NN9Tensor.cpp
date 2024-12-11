/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A tensor.
 */
 
#include "NN9Tensor.h"
#include "../Ops/NN9Init.h"


namespace nn9 {

	// == Constructors.
	Tensor::Tensor( const std::vector<size_t> &_vShape, const std::vector<size_t> &_vStride, NN9_TYPE _tType,
		double _dQuantizeScale, double _dQuantizeZero ) :
		m_vShape( _vShape ),
		m_vStride( _vStride ),
		m_dQuantizeScale( _dQuantizeScale ),
		m_dQuantizeZero( _dQuantizeZero ) {

		m_sSize = 1;
		for ( size_t sDim : m_vShape ) {
			m_sSize *= sDim;
		}

		m_pbBuffer = BufferManager::GblBufferManager.CreateBuffer( _tType, m_sSize, this );
	}
	Tensor::~Tensor() {
		BufferManager::GblBufferManager.DeleteBuffer( m_pbBuffer );
	}

	// == Functions.
	/**
	 * Copies the tensor, maintaining its shape, but optionally changing its underlying data type.
	 * 
	 * \param _tNewType The new type of the tensor to create.
	 * \return Returns a copy of this tensor with each value in the tensor being converted to the type specified by _tNewType.
	 **/
	Tensor Tensor::CopyAs( NN9_TYPE _tNewType ) const {
		Tensor tRet = Tensor( m_vShape, m_vStride, _tNewType,
			m_dQuantizeScale, m_dQuantizeZero );

#define NN9_INNER( SRC_TYPE, CASE_TYPE, DST_TYPE )									\
	case CASE_TYPE : {																\
		auto aSrc = const_cast<Tensor *>(this)->FullView<SRC_TYPE>();				\
		auto aDst = tRet.FullView<DST_TYPE>();										\
		nn9::Init::CopyView( aSrc, aDst );											\
		break;																		\
	}

#define NN9_OUTER( CASE, TYPE )														\
	case CASE : {																	\
		switch ( _tNewType ) {														\
			NN9_INNER( TYPE, NN9_T_BFLOAT16, bfloat16_t )							\
			NN9_INNER( TYPE, NN9_T_FLOAT16, nn9::float16 )							\
			NN9_INNER( TYPE, NN9_T_FLOAT, float )									\
			NN9_INNER( TYPE, NN9_T_DOUBLE, double )									\
																					\
			NN9_INNER( TYPE, NN9_T_UINT8, uint8_t )									\
			NN9_INNER( TYPE, NN9_T_UINT16, uint16_t )								\
			NN9_INNER( TYPE, NN9_T_UINT32, uint32_t )								\
			NN9_INNER( TYPE, NN9_T_UINT64, uint64_t )								\
																					\
			NN9_INNER( TYPE, NN9_T_INT8, int8_t )									\
			NN9_INNER( TYPE, NN9_T_INT16, int16_t )									\
			NN9_INNER( TYPE, NN9_T_INT32, int32_t )									\
			NN9_INNER( TYPE, NN9_T_INT64, int64_t )									\
																					\
			NN9_INNER( TYPE, NN9_T_BOOL, bool )										\
																					\
			/*NN9_INNER( TYPE, NN9_T_COMPLEX64, std::complex<float> )*/					\
			/*NN9_INNER( TYPE, NN9_T_COMPLEX128, std::complex<double> )*/				\
			default : {																\
				throw std::runtime_error( "Tensor::CopyAs: Unsupported type." );	\
			}																		\
		}																			\
		break;																		\
	}

		switch ( m_pbBuffer->Type() ) {
			NN9_OUTER( NN9_T_BFLOAT16, bfloat16_t )
			NN9_OUTER( NN9_T_FLOAT16, nn9::float16 )
			NN9_OUTER( NN9_T_FLOAT, float )
			NN9_OUTER( NN9_T_DOUBLE, double )

			NN9_OUTER( NN9_T_UINT8, uint8_t )
			NN9_OUTER( NN9_T_UINT16, uint16_t )
			NN9_OUTER( NN9_T_UINT32, uint32_t )
			NN9_OUTER( NN9_T_UINT64, uint64_t )

			NN9_OUTER( NN9_T_INT8, int8_t )
			NN9_OUTER( NN9_T_INT16, int16_t )
			NN9_OUTER( NN9_T_INT32, int32_t )
			NN9_OUTER( NN9_T_INT64, int64_t )

			NN9_OUTER( NN9_T_BOOL, bool )

			//NN9_OUTER( NN9_T_COMPLEX64, std::complex<float> )
			//NN9_OUTER( NN9_T_COMPLEX128, std::complex<double> )
			default : {
				throw std::runtime_error( "Tensor::CopyAs: Unsupported type." );
			}
		}
#undef NN9_OUTER
#undef NN9_INNER

		return tRet;
	}

	/**
	 * Calculates the stride table.
	 **/
	void Tensor::CalculateStrides() {
		m_vStride.resize( m_vShape.size() );
		size_t sStride = 1;
		for ( size_t I = m_vShape.size(); I > 0; --I ) {
			m_vStride[I-1] = sStride;
			sStride *= m_vShape[I-1];
		}
	}

}	// namespace nn9

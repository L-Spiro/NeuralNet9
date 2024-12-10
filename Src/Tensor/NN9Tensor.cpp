/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A tensor.
 */
 
#include "NN9Tensor.h"


namespace nn9 {

	Tensor::~Tensor() {
		BufferManager::GblBufferManager.DeleteBuffer( m_pbBuffer );
	}

	// == Functions.
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

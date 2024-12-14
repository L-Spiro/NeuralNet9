/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Common intrinsic operations.
 */

#pragma once

#include <cstdint>
#include <immintrin.h>


namespace nn9 {

	/**
	 * Class Intrin
	 * \brief Common intrinsic operations.
	 *
	 * Description: Common intrinsic operations.
	 */
	class Intrin {
	public :
		// == Functions.
#ifdef __AVX512F__
		/**
		 * Casts 64 int8_t's to 64 int16_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int8x64_to_int16x64( __m512i _mInt8, int16_t * _pi16Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mInt8, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mInt8, 1 );

			_mm512_storeu_si512( _pi16Dst, _mm512_cvtepi8_epi16( mLower ) );
			_mm512_storeu_si512( _pi16Dst + 32, _mm512_cvtepi8_epi16( mUpper ) );
		}
#endif	// #ifdef __AVX512F__
	};

}	// namespace nn9

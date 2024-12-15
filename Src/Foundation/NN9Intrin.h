/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Common intrinsic operations.
 */

#pragma once

#include "../Types/NN9BFloat16.h"
#include "../Types/NN9Float16.h"
#include "NN9Macros.h"

#include <algorithm>
#include <complex>
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
		// ===============================
		// int8_t/uint8_t
		// ===============================
#ifdef __AVX512F__
		/**
		 * Converts 64 int8_t values in a __m512i to 64 uint8_t with saturation.
		 *
		 * Negative values become 0, positive values remain as is (up to 127).
		 *
		 * \param _mInt8 Input vector containing 64 int8_t.
		 * \param _pu8Dst Output pointer to at least 64 uint8_t.
		 */
		static inline void										int8x64_to_uint8x64_saturated( __m512i _mInt8, uint8_t * _pu8Dst ) {
			__m512i mZero = _mm512_setzero_si512();
			__m512i mClamped = _mm512_max_epi8( _mInt8, mZero );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu8Dst), mClamped );
		}

		/**
		 * Converts 64 uint8_t values in a __m512i to 64 int8_t with saturation.
		 *
		 * \param _mUint8 Input vector containing 64 int8_t.
		 * \param _pi8Dst Output pointer to at least 64 int8_t.
		 */
		static inline void										uint8x64_to_int8x64_saturated( __m512i _mUint8, int8_t * _pi8Dst ) {
			__m512i m127 = _mm512_set1_epi8( 127 );
			__m512i mClamped = _mm512_min_epu8( _mUint8, m127 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi8Dst), mClamped );
		}

		/**
		 * Casts 64 int8_t's to 64 int16_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int8x64_to_xint16x64( __m512i _mInt8, int16_t * _pi16Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mInt8, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mInt8, 1 );

			_mm512_storeu_si512( _pi16Dst, _mm512_cvtepi8_epi16( mLower ) );
			_mm512_storeu_si512( _pi16Dst + 32, _mm512_cvtepi8_epi16( mUpper ) );
		}

		/**
		 * Casts 64 uint8_t's to 64 uint16_t's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _pu16Dst The destination buffer.
		 **/
		static inline void										uint8x64_to_xint16x64( __m512i _mUint8, uint16_t * _pu16Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mUint8, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mUint8, 1 );

			_mm512_storeu_si512( _pu16Dst, _mm512_cvtepu8_epi16( mLower ) );
			_mm512_storeu_si512( _pu16Dst + 32, _mm512_cvtepu8_epi16( mUpper ) );
		}

		/**
		 * Casts 64 int8_t's to 64 int32_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi32Dst The destination buffer.
		 **/
		static inline void										int8x64_to_xint32x64( __m512i _mInt8, int32_t * _pi32Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt8, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt8, 1 );

			__m512i mLower16 = _mm512_cvtepi8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepi8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );

			__m512i mUpper32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst), mLower32_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst + 16), mLower32_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst + 32), mUpper32_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst + 48), mUpper32_2 );
		}

		/**
		 * Casts 64 uint8_t's to 64 uint32_t's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										uint8x64_to_xint32x64( __m512i _mUint8, uint32_t * _pu32Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mUint8, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mUint8, 1 );

			__m512i mLower16 = _mm512_cvtepu8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepu8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );

			__m512i mUpper32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mLower32_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 16), mLower32_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 32), mUpper32_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 48), mUpper32_2 );
		}

		/**
		 * Casts 64 int8_t's to 64 int64_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi64Dst The destination buffer.
		 **/
		static inline void										int8x64_to_xint64x64( __m512i _mInt8, int64_t * _pi64Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt8, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt8, 1 );

			__m512i mLower16 = _mm512_cvtepi8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepi8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );
			__m512i mUpper32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			__m512i mLower64_1 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32_1, 0 ) );
			__m512i mLower64_2 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32_1, 1 ) );

			__m512i mLower64_3 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32_2, 0 ) );
			__m512i mLower64_4 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32_2, 1 ) );

			__m512i mUpper64_1 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32_1, 0 ) );
			__m512i mUpper64_2 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32_1, 1 ) );

			__m512i mUpper64_3 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32_2, 0 ) );
			__m512i mUpper64_4 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32_2, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst), mLower64_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 8), mLower64_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 16), mLower64_3 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 24), mLower64_4 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 32), mUpper64_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 40), mUpper64_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 48), mUpper64_3 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 56), mUpper64_4 );
		}

		/**
		 * Casts 64 uint8_t's to 64 uint64_t's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										uint8x64_to_xint64x64( __m512i _mUint8, uint64_t * _pu64Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mUint8, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mUint8, 1 );

			__m512i mLower16 = _mm512_cvtepu8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepu8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );
			__m512i mUpper32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			__m512i mLower64_1 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mLower32_1, 0 ) );
			__m512i mLower64_2 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mLower32_1, 1 ) );

			__m512i mLower64_3 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mLower32_2, 0 ) );
			__m512i mLower64_4 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mLower32_2, 1 ) );

			__m512i mUpper64_1 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mUpper32_1, 0 ) );
			__m512i mUpper64_2 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mUpper32_1, 1 ) );

			__m512i mUpper64_3 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mUpper32_2, 0 ) );
			__m512i mUpper64_4 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mUpper32_2, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst), mLower64_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 8), mLower64_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 16), mLower64_3 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 24), mLower64_4 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 32), mUpper64_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 40), mUpper64_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 48), mUpper64_3 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 56), mUpper64_4 );
		}

		/**
		 * Casts 64 int8_t's to 64 float's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _m0 The first 16 return values.
		 * \param _m1 The second 16 return values.
		 * \param _m2 The third 16 return values.
		 * \param _m3 The fourth 16 return values.
		 **/
		static inline void										int8x64_to_float32x64( __m512i _mInt8, __m512 &_m0, __m512 &_m1, __m512 &_m2, __m512 &_m3 ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt8, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt8, 1 );

			__m512i mLower16 = _mm512_cvtepi8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepi8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );
			__m512i mUpper32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			_m0 = _mm512_cvtepi32_ps( mLower32_1 );
			_m1 = _mm512_cvtepi32_ps( mLower32_2 );
			_m2 = _mm512_cvtepi32_ps( mUpper32_1 );
			_m3 = _mm512_cvtepi32_ps( mUpper32_2 );
		}

		/**
		 * Casts 64 int8_t's to 64 float's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _m0 The first 16 return values.
		 * \param _m1 The second 16 return values.
		 * \param _m2 The third 16 return values.
		 * \param _m3 The fourth 16 return values.
		 **/
		static inline void										uint8x64_to_float32x64( __m512i _mUint8, __m512 &_m0, __m512 &_m1, __m512 &_m2, __m512 &_m3 ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mUint8, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mUint8, 1 );

			__m512i mLower16 = _mm512_cvtepu8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepu8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );
			__m512i mUpper32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			_m0 = _mm512_cvtepi32_ps( mLower32_1 );
			_m1 = _mm512_cvtepi32_ps( mLower32_2 );
			_m2 = _mm512_cvtepi32_ps( mUpper32_1 );
			_m3 = _mm512_cvtepi32_ps( mUpper32_2 );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * Converts 32 int8_t in a __m256i to 32 uint8_t with saturation.
		 *	Negative values become 0, positive are unchanged.
		 *
		 * \param _mInt8 Input vector of 32 int8_t's.
		 * \param _pu8Dst Pointer to store 32 uint8_t's.
		 */
		static inline void										int8x32_to_uint8x32_saturated( __m256i _mInt8, uint8_t * _pu8Dst ) {
			__m256i mZero = _mm256_setzero_si256();
			__m256i mRes = _mm256_max_epi8( _mInt8, mZero );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu8Dst), mRes );
		}

		/**
		 * Converts 32 uint8_t in a __m256i to 32 int8_t with saturation.
		 *	Negative values become 0, positive are unchanged.
		 *
		 * \param _mUint8 Input vector of 32 uint8_t's.
		 * \param _pi8Dst Pointer to store 32 int8_t's.
		 */
		static inline void										uint8x32_to_int8x32_saturated( __m256i _mUint8, int8_t * _pi8Dst ) {
			__m256i m127 = _mm256_set1_epi8( 127 );
			__m256i mRes = _mm256_min_epu8( _mUint8, m127 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi8Dst), mRes );
		}

		/**
		 * Casts 32 int8_t's to 32 int16_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int8x32_to_xint16x32( __m256i _mInt8, int16_t * _pi16Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt8 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt8, 1 );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi16Dst), _mm256_cvtepi8_epi16( mLower ) );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi16Dst + 16), _mm256_cvtepi8_epi16( mUpper ) );
		}

		/**
		 * Casts 32 uint8_t's to 32 uint16_t's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _pu16Dst The destination buffer.
		 **/
		static inline void										uint8x32_to_xint16x32( __m256i _mUint8, uint16_t * _pu16Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mUint8 );
			__m128i mUpper = _mm256_extracti128_si256( _mUint8, 1 );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), _mm256_cvtepu8_epi16( mLower ) );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst + 16), _mm256_cvtepu8_epi16( mUpper ) );
		}

		/**
		 * Casts 32 int8_t's to 32 int32_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi32Dst The destination buffer.
		 **/
		static inline void										int8x32_to_xint32x32( __m256i _mInt8, int32_t * _pi32Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt8 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt8, 1 );

			__m256i mLower16 = _mm256_cvtepi8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepi8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );
        
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst), mLower32_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst + 8), mLower32_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst + 16), mUpper32_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst + 24), mUpper32_2 );
		}

		/**
		 * Casts 32 uint8_t's to 32 uint32_t's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										uint8x32_to_xint32x32( __m256i _mUint8, uint32_t * _pu32Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mUint8 );
			__m128i mUpper = _mm256_extracti128_si256( _mUint8, 1 );

			__m256i mLower16 = _mm256_cvtepu8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepu8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepu16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepu16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepu16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepu16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );
        
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), mLower32_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 8), mLower32_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 16), mUpper32_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 24), mUpper32_2 );
		}

		/**
		 * Casts 32 int8_t's to 32 int64_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi64Dst The destination buffer.
		 **/
		static inline void										int8x32_to_xint64x32( __m256i _mInt8, int64_t * _pi64Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt8 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt8, 1 );

			__m256i mLower16 = _mm256_cvtepi8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepi8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );

			__m256i mLower64_1 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mLower32_1 ) );
			__m256i mLower64_2 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mLower32_1, 1 ) );
			__m256i mLower64_3 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mLower32_2 ) );
			__m256i mLower64_4 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mLower32_2, 1 ) );
			__m256i mUpper64_1 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mUpper32_1 ) );
			__m256i mUpper64_2 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mUpper32_1, 1 ) );
			__m256i mUpper64_3 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mUpper32_2 ) );
			__m256i mUpper64_4 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mUpper32_2, 1 ) );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst), mLower64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 4), mLower64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 8), mLower64_3 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 12), mLower64_4 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 16), mUpper64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 20), mUpper64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 24), mUpper64_3 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 28), mUpper64_4 );
		}

		/**
		 * Casts 32 uint8_t's to 32 uint64_t's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										uint8x32_to_xint64x32( __m256i _mUint8, uint64_t * _pu64Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mUint8 );
			__m128i mUpper = _mm256_extracti128_si256( _mUint8, 1 );

			__m256i mLower16 = _mm256_cvtepu8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepu8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepu16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepu16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepu16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepu16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );

			__m256i mLower64_1 = _mm256_cvtepu32_epi64( _mm256_castsi256_si128( mLower32_1 ) );
			__m256i mLower64_2 = _mm256_cvtepu32_epi64( _mm256_extracti128_si256( mLower32_1, 1 ) );
			__m256i mLower64_3 = _mm256_cvtepu32_epi64( _mm256_castsi256_si128( mLower32_2 ) );
			__m256i mLower64_4 = _mm256_cvtepu32_epi64( _mm256_extracti128_si256( mLower32_2, 1 ) );
			__m256i mUpper64_1 = _mm256_cvtepu32_epi64( _mm256_castsi256_si128( mUpper32_1 ) );
			__m256i mUpper64_2 = _mm256_cvtepu32_epi64( _mm256_extracti128_si256( mUpper32_1, 1 ) );
			__m256i mUpper64_3 = _mm256_cvtepu32_epi64( _mm256_castsi256_si128( mUpper32_2 ) );
			__m256i mUpper64_4 = _mm256_cvtepu32_epi64( _mm256_extracti128_si256( mUpper32_2, 1 ) );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst), mLower64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 4), mLower64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 8), mLower64_3 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 12), mLower64_4 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 16), mUpper64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 20), mUpper64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 24), mUpper64_3 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 28), mUpper64_4 );
		}

		/**
		 * Casts 32 int8_t's to 32 float's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _m0 The first 8 return values.
		 * \param _m1 The second 8 return values.
		 * \param _m2 The third 8 return values.
		 * \param _m3 The fourth 8 return values.
		 **/
		static inline void										int8x32_to_float32x32( __m256i _mInt8, __m256 &_m0, __m256 &_m1, __m256 &_m2, __m256 &_m3 ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt8 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt8, 1 );

			__m256i mLower16 = _mm256_cvtepi8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepi8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );

			_m0 = _mm256_cvtepi32_ps( mLower32_1 );
			_m1 = _mm256_cvtepi32_ps( mLower32_2 );
			_m2 = _mm256_cvtepi32_ps( mUpper32_1 );
			_m3 = _mm256_cvtepi32_ps( mUpper32_2 );
		}

		/**
		 * Casts 32 int8_t's to 32 float's.
		 * 
		 * \param _mUint8 The values to cast.
		 * \param _m0 The first 8 return values.
		 * \param _m1 The second 8 return values.
		 * \param _m2 The third 8 return values.
		 * \param _m3 The fourth 8 return values.
		 **/
		static inline void										uint8x32_to_float32x32( __m256i _mUint8, __m256 &_m0, __m256 &_m1, __m256 &_m2, __m256 &_m3 ) {
			__m128i mLower = _mm256_castsi256_si128( _mUint8 );
			__m128i mUpper = _mm256_extracti128_si256( _mUint8, 1 );

			__m256i mLower16 = _mm256_cvtepu8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepu8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepu16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepu16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepu16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepu16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );

			_m0 = _mm256_cvtepi32_ps( mLower32_1 );
			_m1 = _mm256_cvtepi32_ps( mLower32_2 );
			_m2 = _mm256_cvtepi32_ps( mUpper32_1 );
			_m3 = _mm256_cvtepi32_ps( mUpper32_2 );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int16_t/uint16_t
		// ===============================
#ifdef __AVX512F__
		/**
		 * Casts 32 int16_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										int16x32_to_int8x32_saturated( __m512i _mInt16, int8_t * _pi8Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mInt16, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mInt16, 1 );

			__m128i mLowA = _mm256_castsi256_si128( mLower );
			__m128i mLowB = _mm256_extracti128_si256( mLower, 1 );
			__m128i mPackedLow = _mm_packs_epi16( mLowA, mLowB );

			__m128i mUpA = _mm256_castsi256_si128( mUpper );
			__m128i mUpB = _mm256_extracti128_si256( mUpper, 1 );
			__m128i mPackedUpper = _mm_packs_epi16( mUpA, mUpB );

			__m256i mRes = _mm256_set_m128i( mPackedUpper, mPackedLow );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi8Dst), mRes );
		}

		/**
		 * Casts 32 int16_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int16x32_to_uint8x32_saturated( __m512i _mUint16, uint8_t * _pu8Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mUint16, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mUint16, 1 );

			__m128i mLowA = _mm256_castsi256_si128( mLower );
			__m128i mLowB = _mm256_extracti128_si256( mLower, 1 );
			__m128i mPackedLow = _mm_packus_epi16( mLowA, mLowB );

			__m128i mUpA = _mm256_castsi256_si128( mUpper );
			__m128i mUpB = _mm256_extracti128_si256( mUpper, 1 );
			__m128i mPackedUpper = _mm_packus_epi16( mUpA, mUpB );

			__m256i mRes = _mm256_set_m128i( mPackedUpper, mPackedLow );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu8Dst), mRes );
		}

		/**
		 * Casts 32 uint16_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										uint16x32_to_int8x32_saturated( __m512i _mUint16, int8_t * _pi8Dst ) {
			int16x32_to_int8x32_saturated( _mm512_min_epu16( _mUint16, _mm512_set1_epi16( 0x7F ) ), _pi8Dst );
		}

		/**
		 * Casts 32 uint16_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint16x32_to_uint8x32_saturated( __m512i _mInt16, uint8_t * _pu8Dst ) {
			int16x32_to_uint8x32_saturated( _mm512_min_epu16( _mInt16, _mm512_set1_epi16( 0xFF ) ), _pu8Dst );
		}

		/**
		 * Casts 32 int16_t's to 32 int8_t's without saturation.
		 * Just takes the low 8 bits of each 16-bit integer, ignoring overflow.
		 * 
		 * \param _mInt16 The source values.
		 * \param _pi8Dst The destination buffer (32 bytes).
		 */
		static inline void										int16x32_to_xint8x32( __m512i _mInt16, int8_t * _pi8Dst ) {
			NN9_ALIGN( 64 )
			int16_t i16Tmp[32];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i16Tmp), _mInt16 );

			for ( int i = 0; i < 32; ++i ) {
				_pi8Dst[i] = static_cast<int8_t>(i16Tmp[i]);
			}
		}

		/**
		 * Converts 32 int16_t values in a __m512i to 32 uint16_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt16 Input vector containing 32 int16_t.
		 * \param _pu16Dst Output buffer to store 32 uint16_t.
		 */
		static inline void										int16x32_to_uint16x32_saturated( __m512i _mInt16, uint16_t * _pu16Dst ) {
			__m512i mZero = _mm512_setzero_si512();
			__m512i mClamped = _mm512_max_epi16( _mInt16, mZero );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu16Dst), mClamped );
		}

		/**
		 * Converts 32 uint16_t values in a __m512i to 32 int16_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mUint16 Input vector containing 32 uint16_t.
		 * \param _pi16Dst Output buffer to store 32 int16_t.
		 */
		static inline void										uint16x32_to_int16x32_saturated( __m512i _mUint16, int16_t * _pi16Dst ) {
			__m512i mMax = _mm512_set1_epi16( 0x7FFF );
			__m512i mClamped = _mm512_min_epu16( _mUint16, mMax );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi16Dst), mClamped );
		}

		/**
		 * Casts 32 int16_t's to 32 int32_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi32Dst The destination buffer.
		 **/
		static inline void										int16x32_to_xint32x32( __m512i _mInt16, int32_t * _pi32Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt16, 1 );

			__m512i mLower32 = _mm512_cvtepi16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepi16_epi32( mUpper );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst), mLower32 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst + 16), mUpper32 );
		}

		/**
		 * Casts 32 uint16_t's to 32 uint32_t's.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										uint16x32_to_xint32x32( __m512i _mUint16, uint32_t * _pu32Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mUint16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mUint16, 1 );

			__m512i mLower32 = _mm512_cvtepu16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepu16_epi32( mUpper );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mLower32 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 16), mUpper32 );
		}

		/**
		 * Casts 32 int16_t's to 32 int64_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi64Dst The destination buffer.
		 **/
		static inline void										int16x32_to_xint64x32( __m512i _mInt16, int64_t * _pi64Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt16, 1 );

			__m512i mLower32 = _mm512_cvtepi16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepi16_epi32( mUpper );

			__m512i mLower64 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32, 0 ) );
			__m512i mUpper64 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32, 1 ) );
			__m512i mLower64_upper = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32, 0 ) );
			__m512i mUpper64_upper = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst), mLower64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 8), mUpper64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 16), mLower64_upper );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst + 24), mUpper64_upper );
		}

		/**
		 * Casts 32 uint16_t's to 32 uint64_t's.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										uint16x32_to_xint64x32( __m512i _mUint16, uint64_t * _pu64Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mUint16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mUint16, 1 );

			__m512i mLower32 = _mm512_cvtepu16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepu16_epi32( mUpper );

			__m512i mLower64 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mLower32, 0 ) );
			__m512i mUpper64 = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mLower32, 1 ) );
			__m512i mLower64_upper = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mUpper32, 0 ) );
			__m512i mUpper64_upper = _mm512_cvtepu32_epi64( _mm512_extracti32x8_epi32( mUpper32, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst), mLower64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 8), mUpper64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 16), mLower64_upper );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 24), mUpper64_upper );
		}

		/**
		 * Casts 32 int16_t's to 32 float's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _m0 The first 16 return values.
		 * \param _m1 The second 16 return values.
		 **/
		static inline void										int16x32_to_float32x32( __m512i _mInt16, __m512 &_m0, __m512 &_m1 ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt16, 1 );

			__m512i mLower32 = _mm512_cvtepi16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepi16_epi32( mUpper );

			_m0 = _mm512_cvtepi32_ps( mLower32 );
			_m1 = _mm512_cvtepi32_ps( mUpper32 );
		}

		/**
		 * Casts 32 int16_t's to 32 float's.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _m0 The first 16 return values.
		 * \param _m1 The second 16 return values.
		 **/
		static inline void										uint16x32_to_float32x32( __m512i _mUint16, __m512 &_m0, __m512 &_m1 ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mUint16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mUint16, 1 );

			__m512i mLower32 = _mm512_cvtepu16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepu16_epi32( mUpper );

			_m0 = _mm512_cvtepi32_ps( mLower32 );
			_m1 = _mm512_cvtepi32_ps( mUpper32 );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * Casts 16 int16_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mInt16 The source values (16 int16_t in a __m256i).
		 * \param _pi8Dst The destination buffer (16 int8_t).
		 */
		static inline void										int16x16_to_int8x16_saturated( __m256i _mInt16, int8_t * _pi8Dst ) {
			__m128i mLowA = _mm256_castsi256_si128( _mInt16 );
			__m128i mLowB = _mm256_extracti128_si256( _mInt16, 1 );

			__m128i mPacked = _mm_packs_epi16( mLowA, mLowB );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pi8Dst), mPacked );
		}

		/**
		 * Casts 16 int16_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The source values (16 uint16_t in a __m256i).
		 * \param _pi8Dst The destination buffer (16 uint16_t).
		 */
		static inline void										int16x16_to_uint8x16_saturated( __m256i _mUint16, uint8_t * _pu8Dst ) {
			__m128i mLowA = _mm256_castsi256_si128( _mUint16 );
			__m128i mLowB = _mm256_extracti128_si256( _mUint16, 1 );

			__m128i mPacked = _mm_packus_epi16( mLowA, mLowB );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pu8Dst), mPacked );
		}

		/**
		 * Casts 16 uint16_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										uint16x16_to_int8x16_saturated( __m256i _mUint16, int8_t * _pi8Dst ) {
			int16x16_to_int8x16_saturated( _mm256_min_epu16( _mUint16, _mm256_set1_epi16( 0x7F ) ), _pi8Dst );
		}

		/**
		 * Casts 16 uint16_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint16x16_to_uint8x16_saturated( __m256i _mInt16, uint8_t * _pu8Dst ) {
			int16x16_to_uint8x16_saturated( _mm256_min_epu16( _mInt16, _mm256_set1_epi16( 0xFF ) ), _pu8Dst );
		}

		/**
		 * Casts 16 int16_t's to 16 int8_t's without saturation.
		 * Just takes the low 8 bits of each 16-bit integer, ignoring overflow.
		 * 
		 * \param _mInt16 The source values.
		 * \param _pi8Dst The destination buffer (32 bytes).
		 */
		static inline void										int16x16_to_xint8x16( __m256i _mInt16, int8_t * _pi8Dst ) {
			NN9_ALIGN( 32 )
			int16_t i16Tmp[16];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i16Tmp), _mInt16 );
			int16_t * pi16Tmp = i16Tmp;
			for ( int i = 0; i < (16 / 4); ++i ) {
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
			}
		}

		/**
		 * Converts 16 int16_t values in a __m256i to 16 uint16_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt16 Input vector containing 16 int16_t.
		 * \param _pu16Dst Output buffer to store 16 uint16_t.
		 */
		static inline void										int16x16_to_uint16x16_saturated( __m256i _mInt16, uint16_t * _pu16Dst ) {
			__m256i mZero = _mm256_setzero_si256();
			__m256i mClamped = _mm256_max_epi16( _mInt16, mZero );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), mClamped );
		}

		/**
		 * Converts 16 uint16_t values in a __m256i to 16 int16_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt16 Input vector containing 16 uint16_t.
		 * \param _pu16Dst Output buffer to store 16 int16_t.
		 */
		static inline void										uint16x16_to_int16x16_saturated( __m256i _mInt16, int16_t * _pu16Dst ) {
			__m256i mMax = _mm256_set1_epi16( 0x7FFF );
			__m256i mClamped = _mm256_min_epu16( _mInt16, mMax );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), mClamped );
		}

		/**
		 * Casts 16 int16_t's to 16 int32_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi32Dst The destination buffer.
		 **/
		static inline void										int16x16_to_xint32x16( __m256i _mInt16, int32_t * _pi32Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt16 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt16, 1 );

			__m256i mLower32 = _mm256_cvtepi16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepi16_epi32( mUpper );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst), mLower32 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst + 8), mUpper32 );
		}

		/**
		 * Casts 16 uint16_t's to 16 uint32_t's.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										uint16x16_to_xint32x16( __m256i _mUint16, uint32_t * _pu32Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mUint16 );
			__m128i mUpper = _mm256_extracti128_si256( _mUint16, 1 );

			__m256i mLower32 = _mm256_cvtepu16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepu16_epi32( mUpper );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), mLower32 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 8), mUpper32 );
		}

		/**
		 * Casts 16 int16_t's to 16 int64_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi64Dst The destination buffer.
		 **/
		static inline void										int16x16_to_xint64x16( __m256i _mInt16, int64_t * _pi64Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt16 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt16, 1 );
        
			__m256i mLower32 = _mm256_cvtepi16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepi16_epi32( mUpper );

			__m256i mLower64_1 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mLower32 ) );
			__m256i mLower64_2 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mLower32, 1 ) );
			__m256i mUpper64_1 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mUpper32 ) );
			__m256i mUpper64_2 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mUpper32, 1 ) ); 

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst), mLower64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 4), mLower64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 8), mUpper64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst + 12), mUpper64_2 );
		}

		/**
		 * Casts 16 uint16_t's to 16 uint64_t's.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										uint16x16_to_xint64x16( __m256i _mUint16, uint64_t * _pu64Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mUint16 );
			__m128i mUpper = _mm256_extracti128_si256( _mUint16, 1 );
        
			__m256i mLower32 = _mm256_cvtepu16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepu16_epi32( mUpper );

			__m256i mLower64_1 = _mm256_cvtepu32_epi64( _mm256_castsi256_si128( mLower32 ) );
			__m256i mLower64_2 = _mm256_cvtepu32_epi64( _mm256_extracti128_si256( mLower32, 1 ) );
			__m256i mUpper64_1 = _mm256_cvtepu32_epi64( _mm256_castsi256_si128( mUpper32 ) );
			__m256i mUpper64_2 = _mm256_cvtepu32_epi64( _mm256_extracti128_si256( mUpper32, 1 ) ); 

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst), mLower64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 4), mLower64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 8), mUpper64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 12), mUpper64_2 );
		}

		/**
		 * Casts 16 int16_t's to 16 float's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _m0 The first 16 return values.
		 * \param _m1 The second 16 return values.
		 **/
		static inline void										int16x16_to_float32x16( __m256i _mInt16, __m256 &_m0, __m256 &_m1 ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt16 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt16, 1 );
        
			__m256i mLower32 = _mm256_cvtepi16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepi16_epi32( mUpper );

			_m0 = _mm256_cvtepi32_ps( mLower32 );
			_m1 = _mm256_cvtepi32_ps( mUpper32 );
		}

		/**
		 * Casts 16 int16_t's to 16 float's.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _m0 The first 16 return values.
		 * \param _m1 The second 16 return values.
		 **/
		static inline void										uint16x16_to_float32x16( __m256i _mInt16, __m256 &_m0, __m256 &_m1 ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt16 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt16, 1 );
        
			__m256i mLower32 = _mm256_cvtepu16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepu16_epi32( mUpper );

			_m0 = _mm256_cvtepi32_ps( mLower32 );
			_m1 = _mm256_cvtepi32_ps( mUpper32 );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int32_t/uint32_t
		// ===============================
#ifdef __AVX512F__
		/**
		 * Casts 16 int32_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										int32x16_to_int8x16_saturated( __m512i _mInt16, int8_t * _pi8Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mInt16, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mInt16, 1 );

			__m128i mLowA32 = _mm256_castsi256_si128( mLower );
			__m128i mLowB32 = _mm256_extracti128_si256( mLower, 1 );
			__m128i mPackedLow16 = _mm_packs_epi32( mLowA32, mLowB32 );

			__m128i mUpA32 = _mm256_castsi256_si128( mUpper );
			__m128i mUpB32 = _mm256_extracti128_si256( mUpper, 1 );
			__m128i mPackedUp16 = _mm_packs_epi32( mUpA32, mUpB32 );

			__m256i m16 = _mm256_set_m128i( mPackedUp16, mPackedLow16 );

			__m128i m16Low = _mm256_castsi256_si128( m16 );
			__m128i m16Up = _mm256_extracti128_si256( m16, 1 );
			__m128i mPacked8 = _mm_packs_epi16( m16Low, m16Up );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pi8Dst), mPacked8 );
		}

		/**
		 * Casts 16 int32_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										int32x16_to_uint8x16_saturated( __m512i _mUint16, uint8_t * _pu8Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mUint16, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mUint16, 1 );

			__m128i mLowA32 = _mm256_castsi256_si128( mLower );
			__m128i mLowB32 = _mm256_extracti128_si256( mLower, 1 );
			__m128i mPackedLow16 = _mm_packus_epi32( mLowA32, mLowB32 );

			__m128i mUpA32 = _mm256_castsi256_si128( mUpper );
			__m128i mUpB32 = _mm256_extracti128_si256( mUpper, 1 );
			__m128i mPackedUp16 = _mm_packus_epi32( mUpA32, mUpB32 );

			__m256i m16 = _mm256_set_m128i( mPackedUp16, mPackedLow16 );

			__m128i m16Low = _mm256_castsi256_si128( m16 );
			__m128i m16Up = _mm256_extracti128_si256( m16, 1 );
			__m128i mPacked8 = _mm_packus_epi16( m16Low, m16Up );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pu8Dst), mPacked8 );
		}

		/**
		 * Casts 16 uint32_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										uint32x16_to_int8x16_saturated( __m512i _mUint16, int8_t * _pi8Dst ) {
			int32x16_to_int8x16_saturated( _mm512_min_epu32( _mUint16, _mm512_set1_epi32( 0x7F ) ), _pi8Dst );
		}

		/**
		 * Casts 16 uint32_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint32x16_to_uint8x16_saturated( __m512i _mInt16, uint8_t * _pu8Dst ) {
			int32x16_to_uint8x16_saturated( _mm512_min_epu32( _mInt16, _mm512_set1_epi32( 0xFF ) ), _pu8Dst );
		}

		/**
		 * Casts 16 int32_t's to 16 int8_t's without saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										int32x16_to_xint8x16( __m512i _mInt16, int8_t * _pi8Dst ) {
			NN9_ALIGN( 64 )
			int32_t i32Tmp[16];
			_mm512_store_si512( i32Tmp, _mInt16 );

			int32_t * pi32Tmp = i32Tmp;
			for ( int i = 0; i < 2; ++i ) {
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			}
		}

		/**
		 * Converts 16 int32_t to 16 int16_t with signed saturation using AVX-512.
		 *
		 * \param _mInt32 Input vector of 16 int32_t's in __m512i.
		 * \param _pi16Dst Output pointer to store 16 int16_t's.
		 */
		static inline void										int32x16_to_int16x16_saturated( __m512i _mInt32, int16_t * _pi16Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mInt32, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mInt32, 1 );

			__m128i mLo32A = _mm256_castsi256_si128( mLower );
			__m128i mLo32B = _mm256_extracti128_si256( mLower, 1 );
			__m128i mLo16 = _mm_packs_epi32( mLo32A, mLo32B );

			__m128i mHi32A = _mm256_castsi256_si128( mUpper );
			__m128i mHi32B = _mm256_extracti128_si256( mUpper, 1 );
			__m128i mUp16 = _mm_packs_epi32( mHi32A, mHi32B );

			__m256i mResult = _mm256_set_m128i( mUp16, mLo16 );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi16Dst), mResult );
		}

		/**
		 * Converts 16 int32_t to 16 uint16_t with unsigned saturation using AVX-512.
		 *
		 * \param _mInt32 Input vector of 16 int32_t's in __m512i.
		 * \param _pu16Dst Output pointer to store 16 uint16_t's.
		 */
		static inline void										int32x16_to_uint16x16_saturated( __m512i _mInt32, uint16_t * _pu16Dst ) {
			__m256i mLower = _mm512_extracti64x4_epi64( _mInt32, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( _mInt32, 1 );

			__m128i mLo32A = _mm256_castsi256_si128( mLower );
			__m128i mLo32B = _mm256_extracti128_si256( mLower, 1 );
			__m128i mLo16 = _mm_packus_epi32( mLo32A, mLo32B );

			__m128i mHi32A = _mm256_castsi256_si128( mUpper );
			__m128i mHi32B = _mm256_extracti128_si256( mUpper, 1 );
			__m128i mUp16 = _mm_packus_epi32( mHi32A, mHi32B );

			__m256i mResult = _mm256_set_m128i( mUp16, mLo16 );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), mResult );
		}

		/**
		 * Converts 16 uint32_t to 16 int16_t with signed saturation using AVX-512.
		 *
		 * \param _mUint32 Input vector of 16 uint32_t's in __m512i.
		 * \param _pi16Dst Output pointer to store 16 int16_t's.
		 */
		static inline void										uint32x16_to_int16x16_saturated( __m512i _mUint32, int16_t * _pi16Dst ) {
			int32x16_to_int16x16_saturated( _mm512_min_epu32( _mUint32, _mm512_set1_epi32( 0x7FFF ) ), _pi16Dst );
		}

		/**
		 * Converts 16 uint32_t to 16 uint16_t with unsigned saturation using AVX-512.
		 *
		 * \param _mUint32 Input vector of 16 uint32_t's in __m512i.
		 * \param _pu16Dst Output pointer to store 16 uint16_t's.
		 */
		static inline void										uint32x16_to_uint16x16_saturated( __m512i _mUint32, uint16_t * _pu16Dst ) {
			int32x16_to_uint16x16_saturated( _mm512_min_epu32( _mUint32, _mm512_set1_epi32( 0xFFFF ) ), _pu16Dst );
		}

		/**
		 * Casts 16 int32_t's to 16 int16_t's without saturation.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int32x16_to_xint16x16( __m512i _mInt16, int16_t * _pi16Dst ) {
			NN9_ALIGN( 64 )
			int32_t i32Tmp[16];
			_mm512_store_si512( i32Tmp, _mInt16 );

			int32_t * pi32Tmp = i32Tmp;
			for ( int i = 0; i < 2; ++i ) {
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
				(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			}
		}

		/**
		 * Converts 16 int32_t values in a __m512i to 16 uint32_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt32 Input vector containing 16 int32_t.
		 * \param _pu32Dst Output buffer to store 16 uint32_t.
		 */
		static inline void										int32x16_to_uint32x16_saturated( __m512i _mInt32, uint32_t * _pu32Dst ) {
			__m512i mZero = _mm512_setzero_si512();
			__m512i mClamped = _mm512_max_epi32( _mInt32, mZero );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mClamped );
		}

		/**
		 * Converts 16 uint32_t values in a __m512i to 16 int32_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt32 Input vector containing 16 uint32_t.
		 * \param _pu32Dst Output buffer to store 16 int32_t.
		 */
		static inline void										uint32x16_to_int32x16_saturated( __m512i _mInt32, int32_t * _pu32Dst ) {
			__m512i mMax = _mm512_set1_epi32( 0x7FFFFFFF );
			__m512i mClamped = _mm512_min_epu32( _mInt32, mMax );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mClamped );
		}

		/**
		 * Converts 16 int32_t to 16 int64_t (signed extend) using AVX-512.
		 *
		 * \param _mInt32  Input vector of 16 int32_t in __m512i.
		 * \param _pi64Dst Output pointer to store 16 int64_t.
		 */
		static inline void										int32x16_to_xint64x16( __m512i _mInt32, int64_t *_pi64Dst ) {
			__m256i mLow32 = _mm512_extracti64x4_epi64( _mInt32, 0 );
			__m256i mUpper32 = _mm512_extracti64x4_epi64( _mInt32, 1 );

			__m512i mLow64 = _mm512_cvtepi32_epi64( mLow32 );
			__m512i mUpper64 = _mm512_cvtepi32_epi64( mUpper32 );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(&_pi64Dst[0]), mLow64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(&_pi64Dst[8]), mUpper64 );
		}

		/**
		 * Converts 16 uint32_t to 16 uint64_t (mZero extend) using AVX-512.
		 *
		 * \param _mUint32 Input vector of 16 uint32_t in __m512i.
		 * \param _pu64Dst Output pointer to store 16 uint64_t.
		 */
		static inline void										uint32x16_to_xint64x16( __m512i _mUint32, uint64_t *_pu64Dst ) {
			__m256i mLow32 = _mm512_extracti64x4_epi64( _mUint32, 0 );
			__m256i mUpper32 = _mm512_extracti64x4_epi64( _mUint32, 1 );

			__m512i mLow64 = _mm512_cvtepu32_epi64( mLow32 );
			__m512i mUpper64 = _mm512_cvtepu32_epi64( mUpper32 );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(&_pu64Dst[0]), mLow64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(&_pu64Dst[8]), mUpper64 );
		}

		/**
		 * Casts 16 int32_t's to 16 float's.
		 * 
		 * \param _mInt32 The values to cast.
		 * \param _m0 The first 16 return values.
		 **/
		static inline void										int32x16_to_float32x16( __m512i _mInt32, __m512 &_m0 ) {
			_m0 = _mm512_cvtepi32_ps( _mInt32 );
		}

		/**
		 * Casts 16 int32_t's to 16 float's.
		 * 
		 * \param _mUint32 The values to cast.
		 * \param _m0 The first 16 return values.
		 **/
		static inline void										uint32x16_to_float32x16( __m512i _mUint32, __m512 &_m0 ) {
			_m0 = _mm512_cvtepu32_ps( _mUint32 );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * Casts 8 int32_t's to 8 int8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										int32x8_to_int8x8_saturated( __m256i _mInt16, int8_t * _pi8Dst ) {
			__m128i mLowA32 = _mm256_castsi256_si128( _mInt16 );
			__m128i mLowB32 = _mm256_extracti128_si256( _mInt16, 1 );

			__m128i mPacked16 = _mm_packs_epi32( mLowA32, mLowB32 );

			__m128i mPacked8 = _mm_packs_epi16( mPacked16, mPacked16 );

			NN9_ALIGN( 32 )
			int8_t i8Tmp[16];
			_mm_store_si128( reinterpret_cast<__m128i *>(i8Tmp), mPacked8 );

			(*reinterpret_cast<uint64_t *>(_pi8Dst)) = (*reinterpret_cast<uint64_t *>(i8Tmp));
		}

		/**
		 * Casts 8 int32_t's to 8 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										int32x8_to_uint8x8_saturated( __m256i _mUint16, uint8_t * _pu8Dst ) {
			__m128i mLowA32 = _mm256_castsi256_si128( _mUint16 );
			__m128i mLowB32 = _mm256_extracti128_si256( _mUint16, 1 );

			__m128i mPacked16 = _mm_packus_epi32( mLowA32, mLowB32 );

			__m128i mPacked8 = _mm_packus_epi16( mPacked16, mPacked16 );

			NN9_ALIGN( 32 )
			int8_t i8Tmp[16];
			_mm_store_si128( reinterpret_cast<__m128i *>(i8Tmp), mPacked8 );

			(*reinterpret_cast<uint64_t *>(_pu8Dst)) = (*reinterpret_cast<uint64_t *>(i8Tmp));
		}

		/**
		 * Casts 8 uint32_t's to 8 int8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										uint32x16_to_int8x16_saturated( __m256i _mUint16, int8_t * _pi8Dst ) {
			int32x8_to_int8x8_saturated( _mm256_min_epu32( _mUint16, _mm256_set1_epi32( 0x7F ) ), _pi8Dst );
		}

		/**
		 * Casts 8 int32_t's to 8 uint8_t's with saturation.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint32x16_to_uint8x16_saturated( __m256i _mInt16, uint8_t * _pu8Dst ) {
			int32x8_to_uint8x8_saturated( _mm256_min_epu32( _mInt16, _mm256_set1_epi32( 0xFF ) ), _pu8Dst );
		}

		/**
		 * Casts 8 int32_t's to 8 int8_t's without saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										int32x8_to_xint8x8( __m256i _mInt16, int8_t * _pi8Dst ) {
			NN9_ALIGN( 32 )
			int32_t i32Tmp[8];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i32Tmp), _mInt16 );

			int32_t * pi32Tmp = i32Tmp;
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
			(*_pi8Dst++) = static_cast<int8_t>((*pi32Tmp++));
		}

		/**
		 * Converts 8 int32_t to 8 int16_t with signed saturation using AVX2.
		 *
		 * \param _mInt32  Input vector of 8 int32_t in __m256i.
		 * \param _pi16Dst Output pointer to store 8 int16_t.
		 */
		static inline void										int32x8_to_int16x8_saturated( __m256i _mInt32, int16_t * _pi16Dst ) {
			__m128i mLow = _mm256_castsi256_si128( _mInt32 );
			__m128i mHi = _mm256_extracti128_si256( _mInt32, 1 );

			__m128i mPacked = _mm_packs_epi32( mLow, mHi );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pi16Dst), mPacked );
		}

		/**
		 * Converts 8 uint32_t to 8 uint16_t with unsigned saturation using AVX2.
		 *
		 * \param _mInt32 Input vector of 8 uint32_t in __m256i.
		 * \param _pu16Dst Output pointer to store 8 uint16_t.
		 */
		static inline void										int32x8_to_uint16x8_saturated( __m256i _mInt32, uint16_t * _pu16Dst ) {
			__m128i mLow = _mm256_castsi256_si128( _mInt32 );
			__m128i mHi = _mm256_extracti128_si256( _mInt32, 1 );

			__m128i mPacked = _mm_packus_epi32( mLow, mHi );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pu16Dst), mPacked );
		}

		/**
		 * Converts 8 uint32_t to 8 int16_t with signed saturation using AVX2.
		 *
		 * \param _mUint32 Input vector of 8 uint32_t's in __m256i.
		 * \param _pi16Dst Output pointer to store 8 int16_t's.
		 */
		static inline void										uint32x8_to_int16x8_saturated( __m256i _mUint32, int16_t * _pi16Dst ) {
			int32x8_to_int16x8_saturated( _mm256_min_epu32( _mUint32, _mm256_set1_epi32( 0x7FFF ) ), _pi16Dst );
		}

		/**
		 * Converts 8 uint32_t to 8 uint16_t with unsigned saturation using AVX2.
		 *
		 * \param _mUint32 Input vector of 8 uint32_t's in __m256i.
		 * \param _pu16Dst Output pointer to store 8 uint16_t's.
		 */
		static inline void										uint32x8_to_uint16x8_saturated( __m256i _mUint32, uint16_t * _pu16Dst ) {
			int32x8_to_uint16x8_saturated( _mm256_min_epu32( _mUint32, _mm256_set1_epi32( 0xFFFF ) ), _pu16Dst );
		}

		/**
		 * Casts 8 int32_t's to 8 int16_t's without saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int32x8_to_xint16x8( __m256i _mInt16, int16_t * _pi16Dst ) {
			NN9_ALIGN( 32 )
			int32_t i32Tmp[8];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i32Tmp), _mInt16 );

			int32_t * pi32Tmp = i32Tmp;
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
			(*_pi16Dst++) = static_cast<int16_t>((*pi32Tmp++));
		}

		/**
		 * Converts 8 int32_t values in a __m256i to 8 uint32_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt32 Input vector containing 8 int32_t.
		 * \param _pu32Dst Output buffer to store 8 uint32_t.
		 */
		static inline void										int32x8_to_uint32x8_saturated( __m256i _mInt32, uint32_t * _pu32Dst ) {
			__m256i mZero = _mm256_setzero_si256();
			__m256i mClamped = _mm256_max_epi32( _mInt32, mZero );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), mClamped );
		}

		/**
		 * Converts 8 uint32_t values in a __m256i to 8 int32_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt32 Input vector containing 8 uint32_t.
		 * \param _pu32Dst Output buffer to store 8 int32_t.
		 */
		static inline void										uint32x8_to_int32x8_saturated( __m256i _mInt32, int32_t * _pu32Dst ) {
			__m256i mMax = _mm256_set1_epi32( 0x7FFFFFFF );
			__m256i mClamped = _mm256_min_epu32( _mInt32, mMax );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), mClamped );
		}

		/**
		 * Converts 8 int32_t to 8 int64_t (signed extend) using AVX2 (and SSE4.1 for the conversion).
		 *
		 * \param _mInt32  Input vector of 8 int32_t in __m256i.
		 * \param _pi64Dst Output pointer to store 8 int64_t.
		 */
		static inline void										int32x8_to_xint64x8( __m256i _mInt32, int64_t * _pi64Dst ) {
			__m128i mLo4 = _mm256_castsi256_si128( _mInt32 );
			__m128i mHi4 = _mm256_extracti128_si256( _mInt32, 1 );

			__m128i mI0i1_64 = _mm_cvtepi32_epi64( mLo4 );

			__m128i mI2i3_32 = _mm_shuffle_epi32( mLo4, _MM_SHUFFLE( 1, 0, 3, 2 ) );
			__m128i mI2i3_64 = _mm_cvtepi32_epi64( mI2i3_32 );

			__m128i mI4i5_64 = _mm_cvtepi32_epi64( mHi4 );

			__m128i mI6i7_32 = _mm_shuffle_epi32( mHi4, _MM_SHUFFLE( 1, 0, 3, 2 ) );
			__m128i mI6i7_64 = _mm_cvtepi32_epi64( mI6i7_32 );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pi64Dst[0]), mI0i1_64 );
			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pi64Dst[2]), mI2i3_64 );
			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pi64Dst[4]), mI4i5_64 );
			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pi64Dst[6]), mI6i7_64 );
		}

		/**
		 * Converts 8 uint32_t to 8 uint64_t (mZero extend) using AVX2 (and SSE4.1).
		 *
		 * \param _mUint32 Input vector of 8 uint32_t in __m256i.
		 * \param _pu64Dst Output pointer to store 8 uint64_t.
		 */
		static inline void										uint32x8_to_xint64x8( __m256i _mUint32, uint64_t * _pu64Dst ) {
			__m128i mLo4 = _mm256_castsi256_si128( _mUint32 );
			__m128i mHi4 = _mm256_extracti128_si256( _mUint32, 1 );

			__m128i mI0i1_64 = _mm_cvtepu32_epi64( mLo4 );

			__m128i mI2i3_32 = _mm_shuffle_epi32( mLo4, _MM_SHUFFLE( 1, 0, 3, 2 ) );
			__m128i mI2i3_64 = _mm_cvtepu32_epi64( mI2i3_32 );

			__m128i mI4i5_64 = _mm_cvtepu32_epi64( mHi4 );

			__m128i mI6i7_32 = _mm_shuffle_epi32( mHi4, _MM_SHUFFLE( 1, 0, 3, 2 ) );
			__m128i mI6i7_64 = _mm_cvtepu32_epi64( mI6i7_32 );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pu64Dst[0]), mI0i1_64 );
			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pu64Dst[2]), mI2i3_64 );
			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pu64Dst[4]), mI4i5_64 );
			_mm_storeu_si128( reinterpret_cast<__m128i *>(&_pu64Dst[6]), mI6i7_64 );
		}

		/**
		 * Casts 8 int32_t's to 8 float's.
		 * 
		 * \param _mInt32 The values to cast.
		 * \param _m0 The first 8 return values.
		 **/
		static inline void										int32x8_to_float32x8( __m256i _mInt32, __m256 &_m0 ) {
			_m0 = _mm256_cvtepi32_ps( _mInt32 );
		}

		/**
		 * Casts 8 int32_t's to 8 float's.
		 * 
		 * \param _mUint32 The values to cast.
		 * \param _m0 The first 8 return values.
		 **/
		static inline void										uint32x8_to_float32x8( __m256i _mUint32, __m256 &_m0 ) {
			_m0 = _mm256_cvtepu32_ps( _mUint32 );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int64_t/uint64_t
		// ===============================
#ifdef __AVX512F__
		/**
		 * \brief Casts 8 int64_t's to 8 int8_t's with saturation using AVX-512.
		 *
		 * The input 64-bit integers are mClamped to the int8_t range [-128, 127]
		 * and then stored as int8_t. No further checks are needed after clamping.
		 *
		 * \param _mInt64 The source values (8 int64_t in a __m512i).
		 * \param _pi8Dst The destination buffer (8 int8_t).
		 */
		static inline void										int64x8_to_int8x8_saturated( __m512i _mInt64, int8_t * _pi8Dst ) {
			__m512i minVal = _mm512_set1_epi64( -128 );
			__m512i mMaxVal = _mm512_set1_epi64( 127 );

			__m512i mClamped = _mm512_max_epi64( _mInt64, minVal );
			mClamped = _mm512_min_epi64( mClamped, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi8Dst[i] = static_cast<int8_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 int64_t's to 8 uint8_t's with saturation using AVX-512.
		 *
		 * The input 64-bit integers are first mClamped at the lower bound using max with mZero
		 * to ensure no negatives, then mClamped to [0, 255]. The result is then stored as uint8_t.
		 *
		 * \param _mInt64 The source values (8 int64_t in a __m512i).
		 * \param _pu8Dst The destination buffer (8 uint8_t).
		 */
		static inline void										int64x8_to_uint8x8_saturated( __m512i _mInt64, uint8_t * _pu8Dst ) {
			__m512i mZero = _mm512_setzero_si512();
			__m512i mMaxVal = _mm512_set1_epi64( 255 );

			__m512i mClamped = _mm512_max_epi64( _mInt64, mZero );
			mClamped = _mm512_min_epi64( mClamped, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu8Dst[i] = static_cast<uint8_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 int8_t's with saturation using AVX-512.
		 *
		 * The input unsigned 64-bit integers are mClamped to the int8_t range [-128, 127].
		 * Since we are dealing with uint64_t, no values are negative, so first check if values >127:
		 * - If value < 128, it's fine.
		 * - If value >127, clamp to 127.
		 * Since they're never negative, just clamp to [-128,127] is effectively [0,127].
		 *
		 * \param _mUint64 The source values (8 uint64_t in a __m512i).
		 * \param _pi8Dst The destination buffer (8 int8_t).
		 */
		static inline void										uint64x8_to_int8x8_saturated( __m512i _mUint64, int8_t * _pi8Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( 127 );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi8Dst[i] = static_cast<int8_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 uint8_t's with saturation using AVX-512.
		 *
		 * The input unsigned 64-bit integers are mClamped to [0,255].
		 *
		 * \param _mUint64 The source values (8 uint64_t in a __m512i).
		 * \param _pu8Dst The destination buffer (8 uint8_t).
		 */
		static inline void										uint64x8_to_uint8x8_saturated( __m512i _mUint64, uint8_t * _pu8Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( 255 );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu8Dst[i] = static_cast<uint8_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 int64_t's to 8 int16_t's with saturation using AVX-512.
		 *
		 * Clamps values to [-32768, 32767].
		 *
		 * \param _mInt64 The source values (8 int64_t).
		 * \param _pi16Dst Output buffer (8 int16_t).
		 */
		static inline void										int64x8_to_int16x8_saturated( __m512i _mInt64, int16_t * _pi16Dst ) {
			__m512i minVal = _mm512_set1_epi64( -32768 );
			__m512i mMaxVal = _mm512_set1_epi64( 32767 );

			__m512i mClamped = _mm512_max_epi64( _mInt64, minVal );
			mClamped = _mm512_min_epi64( mClamped, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi16Dst[i] = static_cast<int16_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 int64_t's to 8 uint16_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, 65535].
		 *
		 * \param _mInt64 The source values (8 int64_t).
		 * \param _pu16Dst Output buffer (8 uint16_t).
		 */
		static inline void										int64x8_to_uint16x8_saturated( __m512i _mInt64, uint16_t * _pu16Dst ) {
			__m512i mZero = _mm512_setzero_si512();
			__m512i mMaxVal = _mm512_set1_epi64( 65535 );

			__m512i mClamped = _mm512_max_epi64( _mInt64, mZero );
			mClamped = _mm512_min_epi64( mClamped, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu16Dst[i] = static_cast<uint16_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 int16_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, 32767] (since uint64_t can't be negative).
		 *
		 * \param _mUint64 The source values (8 uint64_t).
		 * \param _pi16Dst Output buffer (8 int16_t).
		 */
		static inline void										uint64x8_to_int16x8_saturated( __m512i _mUint64, int16_t * _pi16Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( 32767 );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi16Dst[i] = static_cast<int16_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 uint16_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, 65535].
		 *
		 * \param _mUint64 The source values (8 uint64_t).
		 * \param _pu16Dst Output buffer (8 uint16_t).
		 */
		static inline void										uint64x8_to_uint16x8_saturated( __m512i _mUint64, uint16_t * _pu16Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( 65535 );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu16Dst[i] = static_cast<uint16_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 int64_t's to 8 int32_t's with saturation using AVX-512.
		 *
		 * Clamps values to [INT32_MIN, INT32_MAX].
		 *
		 * \param _mInt64 The source values (8 int64_t).
		 * \param _pi32Dst Output buffer (8 int32_t).
		 */
		static inline void										int64x8_to_int32x8_saturated( __m512i _mInt64, int32_t * _pi32Dst ) {
			__m512i minVal = _mm512_set1_epi64( static_cast<int64_t>(-2147483648LL) );
			__m512i mMaxVal = _mm512_set1_epi64( static_cast<int64_t>(2147483647LL) );

			__m512i mClamped = _mm512_max_epi64( _mInt64, minVal );
			mClamped = _mm512_min_epi64( mClamped, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi32Dst[i] = static_cast<int32_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 int64_t's to 8 uint32_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, 4294967295].
		 *
		 * \param _mInt64 The source values (8 int64_t).
		 * \param _pu32Dst Output buffer (8 uint32_t).
		 */
		static inline void										int64x8_to_uint32x8_saturated( __m512i _mInt64, uint32_t * _pu32Dst ) {
			__m512i mZero = _mm512_setzero_si512();
			__m512i mMaxVal = _mm512_set1_epi64( 4294967295ULL );

			__m512i mClamped = _mm512_max_epi64( _mInt64, mZero );
			mClamped = _mm512_min_epi64( mClamped, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu32Dst[i] = static_cast<uint32_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 int32_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, INT32_MAX].
		 *
		 * \param _mUint64 The source values (8 uint64_t).
		 * \param _pi32Dst Output buffer (8 int32_t).
		 */
		static inline void										uint64x8_to_int32x8_saturated( __m512i _mUint64, int32_t * _pi32Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( static_cast<int64_t>(2147483647LL) );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi32Dst[i] = static_cast<int32_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 uint32_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, 4294967295].
		 *
		 * \param _mUint64 The source values (8 uint64_t).
		 * \param _pu32Dst Output buffer (8 uint32_t).
		 */
		static inline void										uint64x8_to_uint32x8_saturated( __m512i _mUint64, uint32_t * _pu32Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( 4294967295ULL );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu32Dst[i] = static_cast<uint32_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 int64_t's to 8 uint32_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, INT64_MAX].
		 *
		 * \param _mInt64 The source values (8 uint64_t).
		 * \param _pu32Dst Output buffer (8 uint32_t).
		 */
		static inline void										int64x8_to_uint64x8_saturated( __m512i _mInt64, uint64_t * _pu32Dst ) {
			__m512i mMin = _mm512_setzero_si512();

			__m512i mClamped = _mm512_max_epi64( _mInt64, mMin );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu32Dst[i] = static_cast<uint64_t>(i64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 8 uint64_t's to 8 int32_t's with saturation using AVX-512.
		 *
		 * Clamps values to [0, INT64_MAX].
		 *
		 * \param _mUint64 The source values (8 uint64_t).
		 * \param _pi32Dst Output buffer (8 int32_t).
		 */
		static inline void										uint64x8_to_int64x8_saturated( __m512i _mUint64, int64_t * _pi32Dst ) {
			__m512i mMaxVal = _mm512_set1_epi64( static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL) );

			__m512i mClamped = _mm512_min_epu64( _mUint64, mMaxVal );

			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi32Dst[i] = static_cast<int64_t>(i64Tmp[i]);
			}
		}

#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * \brief Casts 4 int64_t's to 4 int8_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pi8Dst The destination buffer (4 int8_t's).
		 */
		static inline void										int64x4_to_int8x4_saturated( __m256i _mInt64, int8_t * _pi8Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pi8Dst[i] = static_cast<int8_t>(std::clamp( aVal, -128LL, 127LL ));
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 uint8_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pu8Dst The destination buffer (4 uint8_t's).
		 */
		static inline void										int64x4_to_uint8x4_saturated( __m256i _mInt64, uint8_t * _pu8Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pu8Dst[i] = static_cast<uint8_t>(std::clamp( aVal, 0LL, 255LL ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 int8_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pi8Dst The destination buffer (4 int8_t's).
		 */
		static inline void										uint64x4_to_int8x4_saturated( __m256i _mUint64, int8_t * _pi8Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pi8Dst[i] = static_cast<int8_t>(std::min( aVal, 127ULL ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 uint8_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pu8Dst The destination buffer (4 uint8_t's).
		 */
		static inline void										uint64x4_to_uint8x4_saturated( __m256i _mUint64, uint8_t * _pu8Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pu8Dst[i] = static_cast<uint8_t>(std::min( aVal, 255ULL ));
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 int16_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pi16Dst The destination buffer (4 int16_t's).
		 */
		static inline void										int64x4_to_int16x4_saturated( __m256i _mInt64, int16_t * _pi16Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pi16Dst[i] = static_cast<int16_t>(std::clamp( aVal, static_cast<int64_t>(INT16_MIN), static_cast<int64_t>(INT16_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 uint16_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pu16Dst The destination buffer (4 uint16_t's).
		 */
		static inline void										int64x4_to_uint16x4_saturated( __m256i _mInt64, uint16_t * _pu16Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pu16Dst[i] = static_cast<uint16_t>(std::clamp( aVal, 0LL, static_cast<int64_t>(UINT16_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 int16_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pi16Dst The destination buffer (4 int16_t's).
		 */
		static inline void										uint64x4_to_int16x4_saturated( __m256i _mUint64, int16_t * _pi16Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pi16Dst[i] = static_cast<int16_t>(std::min( aVal, static_cast<uint64_t>(INT16_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 uint16_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pu16Dst The destination buffer (4 uint16_t's).
		 */
		static inline void										uint64x4_to_uint16x4_saturated( __m256i _mUint64, uint16_t * _pu16Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pu16Dst[i] = static_cast<uint16_t>(std::min( aVal, static_cast<uint64_t>(UINT16_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 int32_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pi32Dst The destination buffer (4 int32_t's).
		 */
		static inline void										int64x4_to_int32x4_saturated( __m256i _mInt64, int32_t * _pi32Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pi32Dst[i] = static_cast<int32_t>(std::clamp( aVal, static_cast<int64_t>(INT32_MIN), static_cast<int64_t>(INT32_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 uint32_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pu32Dst The destination buffer (4 uint32_t's).
		 */
		static inline void										int64x4_to_uint32x4_saturated( __m256i _mInt64, uint32_t * _pu32Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pu32Dst[i] = static_cast<uint32_t>(std::clamp( aVal, 0LL, static_cast<int64_t>(UINT32_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 int32_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pi32Dst The destination buffer (4 int32_t's).
		 */
		static inline void										uint64x4_to_int32x4_saturated( __m256i _mUint64, int32_t * _pi32Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pi32Dst[i] = static_cast<int32_t>(std::min( aVal, static_cast<uint64_t>(INT32_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 uint32_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pu32Dst The destination buffer (4 uint32_t's).
		 */
		static inline void										uint64x4_to_uint32x4_saturated( __m256i _mUint64, uint32_t * _pu32Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pu32Dst[i] = static_cast<uint32_t>(std::min( aVal, static_cast<uint64_t>(UINT32_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 uint64_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pu64Dst The destination buffer (4 uint64_t's).
		 */
		static inline void										int64x4_to_uint64x4_saturated( __m256i _mInt64, uint64_t * _pu64Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = i64Tmp[i];
				_pu64Dst[i] = static_cast<uint64_t>(std::clamp( aVal, 0LL, static_cast<int64_t>(UINT64_MAX) ));
			}
		}

		/**
		 * \brief Casts 4 uint64_t's to 4 int64_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mUint64 The source values (4 int64_t's in __m256i).
		 * \param _pi64Dst The destination buffer (4 int64_t's).
		 */
		static inline void										uint64x4_to_int64x4_saturated( __m256i _mUint64, int64_t * _pi64Dst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pi64Dst[i] = static_cast<int64_t>(std::min( aVal, static_cast<uint64_t>(INT64_MAX) ));
			}
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int8_t
		// ===============================
		static inline void										int8_scast( int8_t _i8Src, int8_t &_i8Dst ) {
			_i8Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint8_t &_i8Dst ) {
			_i8Dst = static_cast<uint8_t>(std::max( _i8Src, static_cast<int8_t>(0) ));
		}
		static inline void										int8_scast( int8_t _i8Src, int16_t &_i16Dst ) {
			_i16Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint16_t &_i16Dst ) {
			_i16Dst = static_cast<uint16_t>(std::max( _i8Src, static_cast<int8_t>(0) ));
		}
		static inline void										int8_scast( int8_t _i8Src, int32_t &_i32Dst ) {
			_i32Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint32_t &_i32Dst ) {
			_i32Dst = static_cast<uint32_t>(std::max( _i8Src, static_cast<int8_t>(0) ));
		}
		static inline void										int8_scast( int8_t _i8Src, int64_t &_i64Dst ) {
			_i64Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint64_t &_i64Dst ) {
			_i64Dst = static_cast<uint64_t>(std::max( _i8Src, static_cast<int8_t>(0) ));
		}
		static inline void										int8_scast( int8_t _i8Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(_i8Src);
		}
		static inline void										int8_scast( int8_t _i8Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_i8Src);
		}
		static inline void										int8_scast( int8_t _i8Src, float &_fDst ) {
			_fDst = static_cast<float>(_i8Src);
		}
		static inline void										int8_scast( int8_t _i8Src, double &_dDst ) {
			_dDst = static_cast<double>(_i8Src);
		}
		static inline void										int8_scast( int8_t _i8Src, bool &_bDst ) {
			_bDst = _i8Src != 0;
		}
		static inline void										int8_scast( int8_t _i8Src, std::complex<float> & ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<float>." );
		}
		static inline void										int8_scast( int8_t _i8Src, std::complex<double> & ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										int8_scast( __m512i _mInt8, int8_t * _pi8Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi8Dst), _mInt8 );
		}
		static inline void										int8_scast( __m512i _mInt8, uint8_t * _pu8Dst ) {
			int8x64_to_uint8x64_saturated( _mInt8, _pu8Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, int16_t * _pi16Dst ) {
			int8x64_to_xint16x64( _mInt8, _pi16Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, uint16_t * _pu16Dst ) {
			int8x64_to_xint16x64( _mInt8, reinterpret_cast<int16_t *>(_pu16Dst) );
		}
		static inline void										int8_scast( __m512i _mInt8, int32_t * _pi32Dst ) {
			int8x64_to_xint32x64( _mInt8, _pi32Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, uint32_t * _pu32Dst ) {
			int8x64_to_xint32x64( _mInt8, reinterpret_cast<int32_t *>(_pu32Dst) );
		}
		static inline void										int8_scast( __m512i _mInt8, int64_t * _pi64Dst ) {
			int8x64_to_xint64x64( _mInt8, _pi64Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, uint64_t * _pu64Dst ) {
			int8x64_to_xint64x64( _mInt8, reinterpret_cast<int64_t *>(_pu64Dst) );
		}
		static inline void										int8_scast( __m512i _mInt8, nn9::float16 * _pf16Dst ) {
			__m512 m0, m1, m2, m3;
			int8x64_to_float32x64( _mInt8, m0, m1, m2, m3 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 16, m1 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 32, m2 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 48, m3 );
		}
		static inline void										int8_scast( __m512i _mInt8, bfloat16_t * _pf16Dst ) {
			__m512 m0, m1, m2, m3;
			int8x64_to_float32x64( _mInt8, m0, m1, m2, m3 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 16), m1 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 32), m2 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 48), m3 );
		}
		static inline void										int8_scast( __m512i _mInt8, float * _pfDst ) {
			__m512 m0, m1, m2, m3;
			int8x64_to_float32x64( _mInt8, m0, m1, m2, m3 );
			_mm512_storeu_ps( _pfDst, m0 );
			_mm512_storeu_ps( _pfDst + 16, m1 );
			_mm512_storeu_ps( _pfDst + 32, m2 );
			_mm512_storeu_ps( _pfDst + 48, m3 );
		}
		static inline void										int8_scast( __m512i _mInt8, double * _pdDst ) {
			__m512 m0, m1, m2, m3;
			int8x64_to_float32x64( _mInt8, m0, m1, m2, m3 );
			NN9_ALIGN( 64 )
			float fTmp[64];
			_mm512_store_ps( fTmp, m0 );
			_mm512_store_ps( fTmp + 16, m1 );
			_mm512_store_ps( fTmp + 32, m2 );
			_mm512_store_ps( fTmp + 48, m3 );
			for ( int i = 0; i < 64; ++i ) {
				(*_pdDst++) = fTmp[i];
			}
		}
		static inline void										int8_scast( __m512i _mInt8, bool * ) {
			throw std::runtime_error( "int8_scast: I need to implement int8_t -> bool." );
		}
		static inline void										int8_scast( __m512i _mInt8, std::complex<float> * ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<float>." );
		}
		static inline void										int8_scast( __m512i _mInt8, std::complex<double> * ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__
	};

}	// namespace nn9

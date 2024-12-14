/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Common intrinsic operations.
 */

#pragma once

#include "NN9Macros.h"

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
		static inline void										int16x32_to_xint8x32_saturated( __m512i _mInt16, int8_t * _pi8Dst ) {
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
		 * Casts 32 uint16_t's to 16 int8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										uint16x32_to_xint8x32_saturated( __m512i _mUint16, uint8_t * _pu8Dst ) {
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
		 * Casts 32 int16_t's to 32 int8_t's without saturation.
		 * Just takes the low 8 bits of each 16-bit integer, ignoring overflow.
		 * 
		 * \param _mInt16 The source values.
		 * \param _pi8Dst The destination buffer (32 bytes).
		 */
		static inline void										int16x32_to_int8x32( __m512i _mInt16, int8_t * _pi8Dst ) {
			NN9_ALIGN( 64 )
			int16_t i16Tmp[32];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i16Tmp), _mInt16 );

			for ( int i = 0; i < 32; ++i ) {
				_pi8Dst[i] = static_cast<int8_t>(i16Tmp[i]);
			}
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
		static inline void										int16x16_to_xint8x16_saturated( __m256i _mInt16, int8_t * _pi8Dst ) {
			__m128i mLowA = _mm256_castsi256_si128( _mInt16 );
			__m128i mLowB = _mm256_extracti128_si256( _mInt16, 1 );

			__m128i mPacked = _mm_packs_epi16( mLowA, mLowB );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pi8Dst), mPacked );
		}

		/**
		 * Casts 16 uint16_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The source values (16 uint16_t in a __m256i).
		 * \param _pi8Dst The destination buffer (16 uint16_t).
		 */
		static inline void										uint16x16_to_xint8x16_saturated( __m256i _mUint16, uint8_t * _pu8Dst ) {
			__m128i mLowA = _mm256_castsi256_si128( _mUint16 );
			__m128i mLowB = _mm256_extracti128_si256( _mUint16, 1 );

			__m128i mPacked = _mm_packus_epi16( mLowA, mLowB );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pu8Dst), mPacked );
		}

		/**
		 * Casts 16 int16_t's to 16 int8_t's without saturation.
		 * Just takes the low 8 bits of each 16-bit integer, ignoring overflow.
		 * 
		 * \param _mInt16 The source values.
		 * \param _pi8Dst The destination buffer (32 bytes).
		 */
		static inline void										int16x16_to_int8x16( __m256i _mInt16, int8_t * _pi8Dst ) {
			NN9_ALIGN( 32 )
			int16_t i16Tmp[16];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i16Tmp), _mInt16 );
			int16_t * pi16Tmp = i16Tmp;
			for ( int i = 0; i < (16 / 4); i += 4 ) {
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
				(*_pi8Dst++) = static_cast<int8_t>((*pi16Tmp++));
			}
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
		static inline void										int32x16_to_xint8x16_saturated( __m512i _mInt16, int8_t * _pi8Dst ) {
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
		 * Casts 16 uint32_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint32x16_to_xint8x16_saturated( __m512i _mUint16, uint8_t * _pu8Dst ) {
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
			__m128i mPacked8 = _mm_packs_epi16( m16Low, m16Up );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pu8Dst), mPacked8 );
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
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * Casts 8 int32_t's to 8 int8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		static inline void										int32x8_to_xint8x8_saturated( __m256i _mInt16, int8_t * _pi8Dst ) {
			__m128i mLowA32 = _mm256_castsi256_si128( _mInt16 );
			__m128i mLowB32 = _mm256_extracti128_si256( _mInt16, 1 );

			__m128i mPacked16 = _mm_packs_epi32( mLowA32, mLowB32 );

			__m128i mZero = _mm_setzero_si128();
			__m128i mPacked8 = _mm_packs_epi16( mPacked16, mZero );

			NN9_ALIGN( 32 )
			int8_t i8Tmp[16];
			_mm_storeu_si128( reinterpret_cast<__m128i *>(i8Tmp), mPacked8 );

			(*reinterpret_cast<uint64_t *>(_pi8Dst)) = (*reinterpret_cast<uint64_t *>(i8Tmp));
		}

		/**
		 * Casts 8 uint32_t's to 8 uint8_t's with saturation.
		 * 
		 * \param _mUint16 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint32x8_to_xint8x8_saturated( __m256i _mUint16, uint8_t * _pu8Dst ) {
			__m128i mLowA32 = _mm256_castsi256_si128( _mUint16 );
			__m128i mLowB32 = _mm256_extracti128_si256( _mUint16, 1 );

			__m128i mPacked16 = _mm_packus_epi32( mLowA32, mLowB32 );

			__m128i mZero = _mm_setzero_si128();
			__m128i mPacked8 = _mm_packus_epi16( mPacked16, mZero );

			NN9_ALIGN( 32 )
			int8_t i8Tmp[16];
			_mm_store_si128( reinterpret_cast<__m128i *>(i8Tmp), mPacked8 );

			(*reinterpret_cast<uint64_t *>(_pu8Dst)) = (*reinterpret_cast<uint64_t *>(i8Tmp));
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
#endif	// #ifdef __AVX2__
	};

}	// namespace nn9

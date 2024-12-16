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
			__m512i mClamped = _mm512_max_epi8( _mInt8, _mm512_setzero_si512() );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu8Dst), mClamped );
		}

		/**
		 * Converts 64 uint8_t values in a __m512i to 64 int8_t with saturation.
		 *
		 * \param _mUint8 Input vector containing 64 int8_t.
		 * \param _pi8Dst Output pointer to at least 64 int8_t.
		 */
		static inline void										uint8x64_to_int8x64_saturated( __m512i _mUint8, int8_t * _pi8Dst ) {
			__m512i m127 = _mm512_set1_epi8( INT8_MAX );
			__m512i mClamped = _mm512_min_epu8( _mUint8, m127 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi8Dst), mClamped );
		}

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

		/**
		 * Casts 64 int8_t's to 64 uint16_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu16Dst The destination buffer.
		 **/
		static inline void										int8x64_to_uint16x64_saturated( __m512i _mInt8, uint16_t * _pu16Dst ) {
			__m512i mClamped = _mm512_max_epi8( _mInt8, _mm512_setzero_si512() );
			__m256i mLower = _mm512_extracti64x4_epi64( mClamped, 0 );
			__m256i mUpper = _mm512_extracti64x4_epi64( mClamped, 1 );

			_mm512_storeu_si512( _pu16Dst, _mm512_cvtepi8_epi16( mLower ) );
			_mm512_storeu_si512( _pu16Dst + 32, _mm512_cvtepi8_epi16( mUpper ) );
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
		static inline void										int8x64_to_int32x64( __m512i _mInt8, int32_t * _pi32Dst ) {
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
		 * Casts 64 int8_t's to 64 uint32_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										int8x64_to_uint32x64_saturated( __m512i _mInt8, uint32_t * _pu32Dst ) {
			__m512i mClamped = _mm512_max_epi8( _mInt8, _mm512_setzero_si512() );
			__m256i mLower = _mm512_extracti32x8_epi32( mClamped, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( mClamped, 1 );

			__m512i mLower16 = _mm512_cvtepi8_epi16( mLower );
			__m512i mUpper16 = _mm512_cvtepi8_epi16( mUpper );

			__m512i mLower32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );

			__m512i mUpper32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mLower32_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 16), mLower32_2 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 32), mUpper32_1 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 48), mUpper32_2 );
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
		static inline void										int8x64_to_int64x64( __m512i _mInt8, int64_t * _pi64Dst ) {
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
		 * Casts 64 int8_t's to 64 int64_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi64Dst The destination buffer.
		 **/
		static inline void										int8x64_to_uint64x64_saturated( __m512i _mInt8, uint64_t * _pi64Dst ) {
			__m512i mClamped = _mm512_max_epi8( _mInt8, _mm512_setzero_si512() );
			__m256i mLower = _mm512_extracti32x8_epi32( mClamped, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( mClamped, 1 );

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

		/**
		 * \brief Casts 64 int8_t values to 64 double values using AVX-512.
		 *
		 * The input is a __m512i of 64 int8_t. We expand them step by step:
		 * int8 -> int16 -> int32 -> double.
		 *
		 * We do not store intermediate results in registers for final output; we directly
		 * store the 64 doubles into the provided double* array.
		 *
		 * \param _mInt8 The source vector containing 64 int8_t.
		 * \param _pdDst The destination pointer to store 64 double values.
		 */
		static inline void										int8x64_to_float64x64( __m512i _mInt8, double * _pdDst ) {
			__m256i mLower8 = _mm512_extracti32x8_epi32( _mInt8, 0 );
			__m256i mUpper8 = _mm512_extracti32x8_epi32( _mInt8, 1 );

			__m512i mLower16 = _mm512_cvtepi8_epi16( mLower8 );
			__m512i mUpper16 = _mm512_cvtepi8_epi16( mUpper8 );

			__m512i mLower32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );
			__m512i mUpper32_1 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepi16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			__m256i mL32_1a = _mm512_extracti32x8_epi32( mLower32_1, 0 );
			__m256i mL32_1b = _mm512_extracti32x8_epi32( mLower32_1, 1 );
			__m256i mL32_2a = _mm512_extracti32x8_epi32( mLower32_2, 0 );
			__m256i mL32_2b = _mm512_extracti32x8_epi32( mLower32_2, 1 );

			__m256i mU32_1a = _mm512_extracti32x8_epi32( mUpper32_1, 0 );
			__m256i mU32_1b = _mm512_extracti32x8_epi32( mUpper32_1, 1 );
			__m256i mU32_2a = _mm512_extracti32x8_epi32( mUpper32_2, 0 );
			__m256i mU32_2b = _mm512_extracti32x8_epi32( mUpper32_2, 1 );

			__m512d mD0 = _mm512_cvtepi32_pd( mL32_1a );
			__m512d mD1 = _mm512_cvtepi32_pd( mL32_1b );
			__m512d mD2 = _mm512_cvtepi32_pd( mL32_2a );
			__m512d mD3 = _mm512_cvtepi32_pd( mL32_2b );

			__m512d d4 = _mm512_cvtepi32_pd( mU32_1a );
			__m512d d5 = _mm512_cvtepi32_pd( mU32_1b );
			__m512d d6 = _mm512_cvtepi32_pd( mU32_2a );
			__m512d d7 = _mm512_cvtepi32_pd( mU32_2b );

			_mm512_storeu_pd( _pdDst + 0*8, mD0 );
			_mm512_storeu_pd( _pdDst + 1*8, mD1 );
			_mm512_storeu_pd( _pdDst + 2*8, mD2 );
			_mm512_storeu_pd( _pdDst + 3*8, mD3 );
			_mm512_storeu_pd( _pdDst + 4*8, d4 );
			_mm512_storeu_pd( _pdDst + 5*8, d5 );
			_mm512_storeu_pd( _pdDst + 6*8, d6 );
			_mm512_storeu_pd( _pdDst + 7*8, d7 );
		}

		/**
		 * \brief Casts 64 uint8_t values to 64 double values using AVX-512.
		 *
		 * Similar to the int8 version, but uses the unsigned conversion intrinsics.
		 *
		 * \param _mUint8 The source vector containing 64 uint8_t.
		 * \param _pdDst The destination pointer to store 64 double values.
		 */
		static inline void										uint8x64_to_float64x64( __m512i _mUint8, double * _pdDst ) {
			__m256i mLower8 = _mm512_extracti32x8_epi32( _mUint8, 0 );
			__m256i mUpper8 = _mm512_extracti32x8_epi32( _mUint8, 1 );

			__m512i mLower16 = _mm512_cvtepu8_epi16( mLower8 );
			__m512i mUpper16 = _mm512_cvtepu8_epi16( mUpper8 );

			__m512i mLower32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 0 ) );
			__m512i mLower32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mLower16, 1 ) );
			__m512i mUpper32_1 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 0 ) );
			__m512i mUpper32_2 = _mm512_cvtepu16_epi32( _mm512_extracti32x8_epi32( mUpper16, 1 ) );

			__m256i mL32_1a = _mm512_extracti32x8_epi32( mLower32_1, 0 );
			__m256i mL32_1b = _mm512_extracti32x8_epi32( mLower32_1, 1 );
			__m256i mL32_2a = _mm512_extracti32x8_epi32( mLower32_2, 0 );
			__m256i mL32_2b = _mm512_extracti32x8_epi32( mLower32_2, 1 );

			__m256i mU32_1a = _mm512_extracti32x8_epi32( mUpper32_1, 0 );
			__m256i mU32_1b = _mm512_extracti32x8_epi32( mUpper32_1, 1 );
			__m256i mU32_2a = _mm512_extracti32x8_epi32( mUpper32_2, 0 );
			__m256i mU32_2b = _mm512_extracti32x8_epi32( mUpper32_2, 1 );

			__m512d mD0 = _mm512_cvtepi32_pd( mL32_1a );
			__m512d mD1 = _mm512_cvtepi32_pd( mL32_1b );
			__m512d mD2 = _mm512_cvtepi32_pd( mL32_2a );
			__m512d mD3 = _mm512_cvtepi32_pd( mL32_2b );

			__m512d d4 = _mm512_cvtepi32_pd( mU32_1a );
			__m512d d5 = _mm512_cvtepi32_pd( mU32_1b );
			__m512d d6 = _mm512_cvtepi32_pd( mU32_2a );
			__m512d d7 = _mm512_cvtepi32_pd( mU32_2b );

			_mm512_storeu_pd( _pdDst + 0*8, mD0 );
			_mm512_storeu_pd( _pdDst + 1*8, mD1 );
			_mm512_storeu_pd( _pdDst + 2*8, mD2 );
			_mm512_storeu_pd( _pdDst + 3*8, mD3 );
			_mm512_storeu_pd( _pdDst + 4*8, d4 );
			_mm512_storeu_pd( _pdDst + 5*8, d5 );
			_mm512_storeu_pd( _pdDst + 6*8, d6 );
			_mm512_storeu_pd( _pdDst + 7*8, d7 );
		}

		/**
		 * \brief Converts 64 int8_t values to 64 bools.
		 *
		 * Any nonzero int8_t is considered true (1), zero is false (0).
		 *
		 * \param _mInt8 Input vector with 64 int8_t.
		 * \param _pbDst Output buffer to store 64 bools.
		 */
		static inline void										xint8x64_to_boolx64( __m512i _mXint8, bool * _pbDst ) {
			__mmask64 mMask = _mm512_cmpneq_epi8_mask( _mXint8, _mm512_setzero_si512() );
			__m512i mOnes = _mm512_set1_epi8( 1 );
			__m512i mRes = _mm512_maskz_mov_epi8( mMask, mOnes );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pbDst), mRes );
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
			__m256i mRes = _mm256_max_epi8( _mInt8, _mm256_setzero_si256() );
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
			__m256i m127 = _mm256_set1_epi8( INT8_MAX );
			__m256i mRes = _mm256_min_epu8( _mUint8, m127 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi8Dst), mRes );
		}

		/**
		 * Casts 32 int8_t's to 32 int16_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		static inline void										int8x32_to_int16x32( __m256i _mInt8, int16_t * _pi16Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt8 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt8, 1 );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi16Dst), _mm256_cvtepi8_epi16( mLower ) );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi16Dst + 16), _mm256_cvtepi8_epi16( mUpper ) );
		}

		/**
		 * Casts 32 int8_t's to 32 uint16_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu16Dst The destination buffer.
		 **/
		static inline void										int8x32_to_uint16x32_saturated( __m256i _mInt8, uint16_t * _pu16Dst ) {
			__m256i mClamped = _mm256_max_epi8( _mInt8, _mm256_setzero_si256() );
			__m128i mLower = _mm256_castsi256_si128( mClamped );
			__m128i mUpper = _mm256_extracti128_si256( mClamped, 1 );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), _mm256_cvtepi8_epi16( mLower ) );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst + 16), _mm256_cvtepi8_epi16( mUpper ) );
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
		static inline void										int8x32_to_int32x32( __m256i _mInt8, int32_t * _pi32Dst ) {
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
		 * Casts 32 int8_t's to 32 uint32_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										int8x32_to_uint32x32_saturated( __m256i _mInt8, uint32_t * _pu32Dst ) {
			__m256i mClamped = _mm256_max_epi8( _mInt8, _mm256_setzero_si256() );
			__m128i mLower = _mm256_castsi256_si128( mClamped );
			__m128i mUpper = _mm256_extracti128_si256( mClamped, 1 );

			__m256i mLower16 = _mm256_cvtepi8_epi16( mLower );
			__m256i mUpper16 = _mm256_cvtepi8_epi16( mUpper );

			__m256i mLower32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mLower16 ) );
			__m256i mLower32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mLower16, 1 ) );
			__m256i mUpper32_1 = _mm256_cvtepi16_epi32( _mm256_castsi256_si128( mUpper16 ) );
			__m256i mUpper32_2 = _mm256_cvtepi16_epi32( _mm256_extracti128_si256( mUpper16, 1 ) );
        
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), mLower32_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 8), mLower32_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 16), mUpper32_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 24), mUpper32_2 );
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
		static inline void										int8x32_to_int64x32( __m256i _mInt8, int64_t * _pi64Dst ) {
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
		 * Casts 32 int8_t's to 32 uint64_t's.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										int8x32_to_uint64x32_saturated( __m256i _mInt8, uint64_t * _pu64Dst ) {
			__m256i mClamped = _mm256_max_epi8( _mInt8, _mm256_setzero_si256() );
			__m128i mLower = _mm256_castsi256_si128( mClamped );
			__m128i mUpper = _mm256_extracti128_si256( mClamped, 1 );

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

		/**
		 * \brief Casts 32 int8_t values to 32 double-precision floats (double) using AVX2.
		 * 
		 * \param _mInt8 Input vector with 32 int8_t's (in a __m256i).
		 * \param _pdDst Output array of 32 double's.
		 */
		static inline void										int8x32_to_float64x32( __m256i _mInt8, double * _pdDst ) {
			NN9_ALIGN( 16 )
			int32_t iTemp32[32];

			int8x32_to_int32x32( _mInt8, iTemp32 );

			for ( int i = 0; i < 32; i += 4 ) {
				__m128i mSrc = _mm_load_si128( reinterpret_cast<__m128i *>(&iTemp32[i]) );
				__m256d mD = _mm256_cvtepi32_pd( mSrc );
				_mm256_storeu_pd( &_pdDst[i], mD );
			}
		}

		/**
		 * \brief Casts 32 uint8_t values to 32 double-precision floats (double) using AVX2.
		 * 
		 * \param _mUint8 Input vector with 32 uint8_t's (in a __m256i).
		 * \param _pdDst Output array of 32 double's.
		 */
		static inline void										uint8x32_to_float64x32( __m256i _mUint8, double * _pdDst ) {
			NN9_ALIGN( 16 )
			uint32_t uiTemp32[32];

			uint8x32_to_xint32x32( _mUint8, uiTemp32 );

			for ( int i = 0; i < 32; i += 4 ) {
				__m128i mSrc = _mm_load_si128( reinterpret_cast<__m128i *>(&uiTemp32[i]) );
				__m256d mD = _mm256_cvtepi32_pd( mSrc );
				_mm256_storeu_pd( &_pdDst[i], mD );
			}
		}

		/**
		 * \brief Converts 32 int8_t/uint8_t values to 32 bool values using AVX2.
		 * 
		 * True if uint8_t != 0, False if uint8_t == 0.
		 * 
		 * \param _mXint8 Input vector of 32 int8_t's or uint8_t's.
		 * \param _pbDst Output array of 32 bool's.
		 */
		static inline void										xint8x32_to_boolx32( __m256i _mXint8, bool *_pbDst ) {
			__m256i mCmp = _mm256_cmpeq_epi8( _mXint8, _mm256_setzero_si256() );
			__m256i mNotCmp = _mm256_xor_si256( mCmp, _mm256_set1_epi8( static_cast<char>(0xFF) ) );
			__m256i mOnes = _mm256_set1_epi8( 1 );
			__m256i mRes = _mm256_and_si256( mNotCmp, mOnes );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pbDst), mRes );
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
			__m512i mClamped = _mm512_max_epi16( _mInt16, _mm512_setzero_si512() );
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
			__m512i mMax = _mm512_set1_epi16( INT16_MAX );
			__m512i mClamped = _mm512_min_epu16( _mUint16, mMax );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi16Dst), mClamped );
		}

		/**
		 * Casts 32 int16_t's to 32 int32_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi32Dst The destination buffer.
		 **/
		static inline void										int16x32_to_int32x32( __m512i _mInt16, int32_t * _pi32Dst ) {
			__m256i mLower = _mm512_extracti32x8_epi32( _mInt16, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( _mInt16, 1 );

			__m512i mLower32 = _mm512_cvtepi16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepi16_epi32( mUpper );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst), mLower32 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst + 16), mUpper32 );
		}

		/**
		 * Casts 32 int16_t's to 32 uint32_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										int16x32_to_uint32x32_saturated( __m512i _mInt16, uint32_t * _pu32Dst ) {
			__m512i mClamped = _mm512_max_epi16( _mInt16, _mm512_setzero_si512() );
			__m256i mLower = _mm512_extracti32x8_epi32( mClamped, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( mClamped, 1 );

			__m512i mLower32 = _mm512_cvtepi16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepi16_epi32( mUpper );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mLower32 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst + 16), mUpper32 );
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
		static inline void										int16x32_to_int64x32( __m512i _mInt16, int64_t * _pi64Dst ) {
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
		 * Casts 32 int16_t's to 32 uint64_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										int16x32_to_uint64x32_saturated( __m512i _mInt16, uint64_t * _pu64Dst ) {
			__m512i mClamped = _mm512_max_epi16( _mInt16, _mm512_setzero_si512() );
			__m256i mLower = _mm512_extracti32x8_epi32( mClamped, 0 );
			__m256i mUpper = _mm512_extracti32x8_epi32( mClamped, 1 );

			__m512i mLower32 = _mm512_cvtepi16_epi32( mLower );
			__m512i mUpper32 = _mm512_cvtepi16_epi32( mUpper );

			__m512i mLower64 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32, 0 ) );
			__m512i mUpper64 = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mLower32, 1 ) );
			__m512i mLower64_upper = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32, 0 ) );
			__m512i mUpper64_upper = _mm512_cvtepi32_epi64( _mm512_extracti32x8_epi32( mUpper32, 1 ) );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst), mLower64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 8), mUpper64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 16), mLower64_upper );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst + 24), mUpper64_upper );
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

		/**
		 * @brief Converts 32 int16_t values in a __m512i to 32 double values using AVX-512.
		 *
		 * Steps:
		 * 1. Split into two __m256i halves if needed (using _mm512_extracti32x8_epi32).
		 * 2. Convert int16 to int32 with _mm512_cvtepi16_epi32.
		 * 3. From int32, convert to double in chunks of 8 elements using _mm512_cvtepi32_pd on __m256i parts.
		 *
		 * @param _mInt16 Input vector (32 int16_t's).
		 * @param _pdDst Output pointer to 32 double's.
		 */
		static inline void										int16x32_to_float64x32( __m512i _mInt16, double * _pdDst ) {
			__m256i mLow16 = _mm512_extracti32x8_epi32( _mInt16, 0 );
			__m256i mHigh16 = _mm512_extracti32x8_epi32( _mInt16, 1 );

			__m512i mLow32  = _mm512_cvtepi16_epi32( mLow16 );
			__m512i mHigh32 = _mm512_cvtepi16_epi32( mHigh16 );

			__m256i mLow32A  = _mm512_extracti32x8_epi32( mLow32, 0 );
			__m256i mLow32B  = _mm512_extracti32x8_epi32( mLow32, 1 );
			__m256i mHigh32A = _mm512_extracti32x8_epi32( mHigh32, 0 );
			__m256i mHigh32B = _mm512_extracti32x8_epi32( mHigh32, 1 );

			__m512d mD0 = _mm512_cvtepi32_pd( mLow32A );
			__m512d mD1 = _mm512_cvtepi32_pd( mLow32B );
			__m512d mD2 = _mm512_cvtepi32_pd( mHigh32A );
			__m512d mD3 = _mm512_cvtepi32_pd( mHigh32B );

			_mm512_storeu_pd( _pdDst + 0 * 8, mD0 );
			_mm512_storeu_pd( _pdDst + 1 * 8, mD1 );
			_mm512_storeu_pd( _pdDst + 2 * 8, mD2 );
			_mm512_storeu_pd( _pdDst + 3 * 8, mD3 );
		}

		/**
		 * @brief Converts 32 uint16_t values in a __m512i to 32 double values using AVX-512.
		 *
		 * Similar to int16, but uses unsigned conversion _mm512_cvtepu16_epi32.
		 *
		 * @param _mXint16 Input vector (32 uint16_t's).
		 * @param _pdDst Output pointer to 32 double's.
		 */
		static inline void										uint16x32_to_float64x32( __m512i _mXint16, double * _pdDst ) {
			__m256i mLow16 = _mm512_extracti32x8_epi32( _mXint16, 0 );
			__m256i mHigh16 = _mm512_extracti32x8_epi32( _mXint16, 1 );

			__m512i mLow32  = _mm512_cvtepu16_epi32( mLow16 );
			__m512i mHigh32 = _mm512_cvtepu16_epi32( mHigh16 );

			__m256i mLow32A  = _mm512_extracti32x8_epi32( mLow32, 0 );
			__m256i mLow32B  = _mm512_extracti32x8_epi32( mLow32, 1 );
			__m256i mHigh32A = _mm512_extracti32x8_epi32( mHigh32, 0 );
			__m256i mHigh32B = _mm512_extracti32x8_epi32( mHigh32, 1 );

			__m512d mD0 = _mm512_cvtepi32_pd( mLow32A );
			__m512d mD1 = _mm512_cvtepi32_pd( mLow32B );
			__m512d mD2 = _mm512_cvtepi32_pd( mHigh32A );
			__m512d mD3 = _mm512_cvtepi32_pd( mHigh32B );

			_mm512_storeu_pd( _pdDst + 0 * 8, mD0 );
			_mm512_storeu_pd( _pdDst + 1 * 8, mD1 );
			_mm512_storeu_pd( _pdDst + 2 * 8, mD2 );
			_mm512_storeu_pd( _pdDst + 3 * 8, mD3 );
		}

		/**
		 * @brief Converts 32 int16_t to bool using AVX-512.
		 *
		 * Nonzero = 1, zero = 0.
		 *
		 * @param _mInt16 Input vector.
		 * @param _pbDst Output array of 32 bool.
		 */
		static inline void										xint16x32_to_boolx32( __m512i _mInt16, bool * _pbDst ) {
			__mmask32 mask = _mm512_cmpneq_epi16_mask( _mInt16, _mm512_setzero_si512() );
			__m512i result = _mm512_maskz_mov_epi16( mask, _mm512_set1_epi16( 1 ) );
			static_assert( sizeof( bool ) == sizeof( int8_t ) );
			int16x32_to_xint8x32( result, reinterpret_cast<int8_t *>(_pbDst) );
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
			__m256i mClamped = _mm256_max_epi16( _mInt16, _mm256_setzero_si256() );
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
			__m256i mMax = _mm256_set1_epi16( INT16_MAX );
			__m256i mClamped = _mm256_min_epu16( _mInt16, mMax );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), mClamped );
		}

		/**
		 * Casts 16 int16_t's to 16 int32_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi32Dst The destination buffer.
		 **/
		static inline void										int16x16_to_int32x16( __m256i _mInt16, int32_t * _pi32Dst ) {
			__m128i mLower = _mm256_castsi256_si128( _mInt16 );
			__m128i mUpper = _mm256_extracti128_si256( _mInt16, 1 );

			__m256i mLower32 = _mm256_cvtepi16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepi16_epi32( mUpper );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst), mLower32 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst + 8), mUpper32 );
		}

		/**
		 * Casts 16 int16_t's to 16 uint32_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pu32Dst The destination buffer.
		 **/
		static inline void										int16x16_to_uint32x16_saturated( __m256i _mInt16, uint32_t * _pu32Dst ) {
			__m256i mClamped = _mm256_max_epi16( _mInt16, _mm256_setzero_si256() );
			__m128i mLower = _mm256_castsi256_si128( mClamped );
			__m128i mUpper = _mm256_extracti128_si256( mClamped, 1 );

			__m256i mLower32 = _mm256_cvtepi16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepi16_epi32( mUpper );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), mLower32 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst + 8), mUpper32 );
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
		static inline void										int16x16_to_int64x16( __m256i _mInt16, int64_t * _pi64Dst ) {
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
		 * Casts 16 int16_t's to 16 uint64_t's.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pu64Dst The destination buffer.
		 **/
		static inline void										int16x16_to_uint64x16_saturated( __m256i _mInt16, uint64_t * _pu64Dst ) {
			__m256i mClamped = _mm256_max_epi16( _mInt16, _mm256_setzero_si256() );
			__m128i mLower = _mm256_castsi256_si128( mClamped );
			__m128i mUpper = _mm256_extracti128_si256( mClamped, 1 );
        
			__m256i mLower32 = _mm256_cvtepi16_epi32( mLower );
			__m256i mUpper32 = _mm256_cvtepi16_epi32( mUpper );

			__m256i mLower64_1 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mLower32 ) );
			__m256i mLower64_2 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mLower32, 1 ) );
			__m256i mUpper64_1 = _mm256_cvtepi32_epi64( _mm256_castsi256_si128( mUpper32 ) );
			__m256i mUpper64_2 = _mm256_cvtepi32_epi64( _mm256_extracti128_si256( mUpper32, 1 ) ); 

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst), mLower64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 4), mLower64_2 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 8), mUpper64_1 );
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst + 12), mUpper64_2 );
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

		/**
		 * @brief Converts 16 int16_t to 16 double using AVX2.
		 *
		 * Fallback:
		 * 1. Store 16 int16.
		 * 2. Convert each to int32_t scalar.
		 * 3. Convert in chunks of 4 int32 to double using _mm256_cvtepi32_pd.
		 *
		 * @param _mInt16 Input (16 int16_t)
		 * @param _pdDst Output array of 16 double
		 */
		static inline void										int16x16_to_float64x16( __m256i _mInt16, double * _pdDst ) {
			NN9_ALIGN( 16 )
			int32_t iTemp32[16];

			int16x16_to_int32x16( _mInt16, iTemp32 );

			for ( int i = 0; i < 16; i += 4 ) {
				__m128i mSrc = _mm_loadu_si128( reinterpret_cast<__m128i *>(&iTemp32[i]) );
				__m256d mD = _mm256_cvtepi32_pd( mSrc );
				_mm256_storeu_pd( &_pdDst[i], mD );
			}
		}

		/**
		 * @brief Converts 16 uint16_t to 16 double using AVX2.
		 *
		 * Same fallback approach, but zero-extension is trivial.
		 *
		 * @param _mUint16 Input (16 uint16_t)
		 * @param _pdDst Output array of 16 double
		 */
		static inline void										uint16x16_to_float64x16( __m256i _mUint16, double * _pdDst ) {
			NN9_ALIGN( 16 )
			uint32_t uiTemp32[16];

			uint16x16_to_xint32x16( _mUint16, uiTemp32 );

			for ( int i = 0; i < 16; i += 4 ) {
				__m128i mSrc = _mm_loadu_si128( reinterpret_cast<__m128i *>(&uiTemp32[i]) );
				__m256d mD = _mm256_cvtepi32_pd( mSrc );
				_mm256_storeu_pd( &_pdDst[i], mD );
			}
		}

		/**
		 * @brief Converts 16 int16_t to bool using AVX2.
		 *
		 * Nonzero = 1, zero = 0.
		 * Compare with zero using _mm256_cmpeq_epi16,
		 * invert, and mask with ones.
		 *
		 * @param _mInt16 Input (16 int16_t)
		 * @param _pbDst Output array of 16 bool
		 */
		static inline void										xint16x16_to_boolx16( __m256i _mInt16, bool * _pbDst ) {
			__m256i mCmp = _mm256_cmpeq_epi16( _mInt16, _mm256_setzero_si256() );
			__m256i mNotCmp = _mm256_xor_si256( mCmp, _mm256_set1_epi8( static_cast<char>(0xFF) ) );
			__m256i mRes = _mm256_and_si256( mNotCmp, _mm256_set1_epi16( 1 ) );
			int16x16_to_xint8x16( mRes, reinterpret_cast<int8_t *>(_pbDst) );
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
			int32x16_to_int16x16_saturated( _mm512_min_epu32( _mUint32, _mm512_set1_epi32( INT16_MAX ) ), _pi16Dst );
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
			__m512i mMax = _mm512_set1_epi32( INT32_MAX );
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
			int32x8_to_int16x8_saturated( _mm256_min_epu32( _mUint32, _mm256_set1_epi32( INT16_MAX ) ), _pi16Dst );
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
			__m256i mMax = _mm256_set1_epi32( INT32_MAX );
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
			__m512i minVal = _mm512_set1_epi64( INT8_MIN );
			__m512i mMaxVal = _mm512_set1_epi64( INT8_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( UINT_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( INT8_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( UINT_MAX );

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
			__m512i minVal = _mm512_set1_epi64( INT16_MIN );
			__m512i mMaxVal = _mm512_set1_epi64( INT16_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( INT16_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( UINT32_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( UINT32_MAX );

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
			__m512i mMaxVal = _mm512_set1_epi64( INT64_MAX );

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
				_pi8Dst[i] = static_cast<int8_t>(std::clamp<int64_t>( aVal, INT8_MIN, INT8_MAX ));
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
				_pu8Dst[i] = static_cast<uint8_t>(std::clamp<int64_t>( aVal, 0LL, UINT_MAX ));
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
				_pi8Dst[i] = static_cast<int8_t>(std::min<uint64_t>( aVal, INT8_MAX ));
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
				_pu8Dst[i] = static_cast<uint8_t>(std::min<uint64_t>( aVal, UINT_MAX ));
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
			_i8Dst = static_cast<uint8_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										int8_scast( int8_t _i8Src, int16_t &_i16Dst ) {
			_i16Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint16_t &_i16Dst ) {
			_i16Dst = static_cast<uint16_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										int8_scast( int8_t _i8Src, int32_t &_i32Dst ) {
			_i32Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint32_t &_i32Dst ) {
			_i32Dst = static_cast<uint32_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										int8_scast( int8_t _i8Src, int64_t &_i64Dst ) {
			_i64Dst = _i8Src;
		}
		static inline void										int8_scast( int8_t _i8Src, uint64_t &_i64Dst ) {
			_i64Dst = static_cast<uint64_t>(std::max<int8_t>( _i8Src, 0 ));
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
			int8x64_to_int16x64( _mInt8, _pi16Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, uint16_t * _pu16Dst ) {
			int8x64_to_uint16x64_saturated( _mInt8, reinterpret_cast<uint16_t *>(_pu16Dst) );
		}
		static inline void										int8_scast( __m512i _mInt8, int32_t * _pi32Dst ) {
			int8x64_to_int32x64( _mInt8, _pi32Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, uint32_t * _pu32Dst ) {
			int8x64_to_uint32x64_saturated( _mInt8, _pu32Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, int64_t * _pi64Dst ) {
			int8x64_to_int64x64( _mInt8, _pi64Dst );
		}
		static inline void										int8_scast( __m512i _mInt8, uint64_t * _pu64Dst ) {
			int8x64_to_uint64x64_saturated( _mInt8, _pu64Dst );
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
			int8x64_to_float64x64( _mInt8, _pdDst );
		}
		static inline void										int8_scast( __m512i _mInt8, bool * _pbDst ) {
			xint8x64_to_boolx64( _mInt8, _pbDst );
		}
		static inline void										int8_scast( __m512i _mInt8, std::complex<float> * ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<float>." );
		}
		static inline void										int8_scast( __m512i _mInt8, std::complex<double> * ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										int8_scast( __m256i _mInt8, int8_t * _pi8Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi8Dst), _mInt8 );
		}
		static inline void										int8_scast( __m256i _mInt8, uint8_t * _pu8Dst ) {
			int8x32_to_uint8x32_saturated( _mInt8, _pu8Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, int16_t * _pi16Dst ) {
			int8x32_to_int16x32( _mInt8, _pi16Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, uint16_t * _pu16Dst ) {
			int8x32_to_uint16x32_saturated( _mInt8, _pu16Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, int32_t * _pi32Dst ) {
			int8x32_to_int32x32( _mInt8, _pi32Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, uint32_t * _pu32Dst ) {
			int8x32_to_uint32x32_saturated( _mInt8, _pu32Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, int64_t * _pi64Dst ) {
			int8x32_to_int64x32( _mInt8, _pi64Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, uint64_t * _pu64Dst ) {
			int8x32_to_uint64x32_saturated( _mInt8, _pu64Dst );
		}
		static inline void										int8_scast( __m256i _mInt8, nn9::float16 * _pf16Dst ) {
			__m256 m0, m1, m2, m3;
			int8x32_to_float32x32( _mInt8, m0, m1, m2, m3 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 8, m1 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 16, m2 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 24, m3 );
		}
		static inline void										int8_scast( __m256i _mInt8, bfloat16_t * _pf16Dst ) {
			__m256 m0, m1, m2, m3;
			int8x32_to_float32x32( _mInt8, m0, m1, m2, m3 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 8), m1 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 16), m2 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 24), m3 );
		}
		static inline void										int8_scast( __m256i _mInt8, float * _pfDst ) {
			__m256 m0, m1, m2, m3;
			int8x32_to_float32x32( _mInt8, m0, m1, m2, m3 );
			_mm256_storeu_ps( _pfDst, m0 );
			_mm256_storeu_ps( _pfDst + 8, m1 );
			_mm256_storeu_ps( _pfDst + 16, m2 );
			_mm256_storeu_ps( _pfDst + 24, m3 );
		}
		static inline void										int8_scast( __m256i _mInt8, double * _pdDst ) {
			int8x32_to_float64x32( _mInt8, _pdDst );
		}
		static inline void										int8_scast( __m256i _mInt8, bool * _pbDst ) {
			xint8x32_to_boolx32( _mInt8, _pbDst );
		}
		static inline void										int8_scast( __m256i _mInt8, std::complex<float> * ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<float>." );
		}
		static inline void										int8_scast( __m256i _mInt8, std::complex<double> * ) {
			throw std::runtime_error( "int8_scast: No conversion available for int8_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// uint8_t
		// ===============================
		static inline void										uint8_scast( uint8_t _u8Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::min<uint8_t>( _u8Src, INT8_MAX ));
		}
		static inline void										uint8_scast( uint8_t _u8Src, uint8_t &_i8Dst ) {
			_i8Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, int16_t &_i16Dst ) {
			_i16Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, uint16_t &_i16Dst ) {
			_i16Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, int32_t &_i32Dst ) {
			_i32Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, uint32_t &_i32Dst ) {
			_i32Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, int64_t &_i64Dst ) {
			_i64Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, uint64_t &_i64Dst ) {
			_i64Dst = _u8Src;
		}
		static inline void										uint8_scast( uint8_t _u8Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(_u8Src);
		}
		static inline void										uint8_scast( uint8_t _u8Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_u8Src);
		}
		static inline void										uint8_scast( uint8_t _u8Src, float &_fDst ) {
			_fDst = static_cast<float>(_u8Src);
		}
		static inline void										uint8_scast( uint8_t _u8Src, double &_dDst ) {
			_dDst = static_cast<double>(_u8Src);
		}
		static inline void										uint8_scast( uint8_t _u8Src, bool &_bDst ) {
			_bDst = _u8Src != 0;
		}
		static inline void										uint8_scast( uint8_t _u8Src, std::complex<float> & ) {
			throw std::runtime_error( "uint8_scast: No conversion available for uint8_t -> std::complex<float>." );
		}
		static inline void										uint8_scast( uint8_t _u8Src, std::complex<double> & ) {
			throw std::runtime_error( "uint8_scast: No conversion available for uint8_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										uint8_scast( __m512i _mUint8, int8_t * _pi8Dst ) {
			uint8x64_to_int8x64_saturated( _mUint8, _pi8Dst );
		}
		static inline void										uint8_scast( __m512i _mUint8, uint8_t * _pu8Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu8Dst), _mUint8 );
		}
		static inline void										uint8_scast( __m512i _mUint8, int16_t * _pi16Dst ) {
			uint8x64_to_xint16x64( _mUint8, reinterpret_cast<uint16_t *>(_pi16Dst) );
		}
		static inline void										uint8_scast( __m512i _mUint8, uint16_t * _pu16Dst ) {
			uint8x64_to_xint16x64( _mUint8, _pu16Dst );
		}
		static inline void										uint8_scast( __m512i _mUint8, int32_t * _pi32Dst ) {
			uint8x64_to_xint32x64( _mUint8, reinterpret_cast<uint32_t *>(_pi32Dst) );
		}
		static inline void										uint8_scast( __m512i _mUint8, uint32_t * _pu32Dst ) {
			uint8x64_to_xint32x64( _mUint8, _pu32Dst );
		}
		static inline void										uint8_scast( __m512i _mUint8, int64_t * _pi64Dst ) {
			uint8x64_to_xint64x64( _mUint8, reinterpret_cast<uint64_t *>(_pi64Dst) );
		}
		static inline void										uint8_scast( __m512i _mUint8, uint64_t * _pu64Dst ) {
			uint8x64_to_xint64x64( _mUint8, _pu64Dst );
		}
		static inline void										uint8_scast( __m512i _mUint8, nn9::float16 * _pf16Dst ) {
			__m512 m0, m1, m2, m3;
			uint8x64_to_float32x64( _mUint8, m0, m1, m2, m3 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 16, m1 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 32, m2 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 48, m3 );
		}
		static inline void										uint8_scast( __m512i _mUint8, bfloat16_t * _pf16Dst ) {
			__m512 m0, m1, m2, m3;
			uint8x64_to_float32x64( _mUint8, m0, m1, m2, m3 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 16), m1 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 32), m2 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 48), m3 );
		}
		static inline void										uint8_scast( __m512i _mUint8, float * _pfDst ) {
			__m512 m0, m1, m2, m3;
			uint8x64_to_float32x64( _mUint8, m0, m1, m2, m3 );
			_mm512_storeu_ps( _pfDst, m0 );
			_mm512_storeu_ps( _pfDst + 16, m1 );
			_mm512_storeu_ps( _pfDst + 32, m2 );
			_mm512_storeu_ps( _pfDst + 48, m3 );
		}
		static inline void										uint8_scast( __m512i _mUint8, double * _pdDst ) {
			uint8x64_to_float64x64( _mUint8, _pdDst );
		}
		static inline void										uint8_scast( __m512i _mUint8, bool * _pbDst ) {
			xint8x64_to_boolx64( _mUint8, _pbDst );
		}
		static inline void										uint8_scast( __m512i _mUint8, std::complex<float> * ) {
			throw std::runtime_error( "uint8_scast: No conversion available for uint8_t -> std::complex<float>." );
		}
		static inline void										uint8_scast( __m512i _mUint8, std::complex<double> * ) {
			throw std::runtime_error( "uint8_scast: No conversion available for uint8_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										uint8_scast( __m256i _mUint8, int8_t * _pi8Dst ) {
			uint8x32_to_int8x32_saturated( _mUint8, _pi8Dst );
		}
		static inline void										uint8_scast( __m256i _mUint8, uint8_t * _pu8Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu8Dst), _mUint8 );
		}
		static inline void										uint8_scast( __m256i _mUint8, int16_t * _pi16Dst ) {
			uint8x32_to_xint16x32( _mUint8, reinterpret_cast<uint16_t *>(_pi16Dst) );
		}
		static inline void										uint8_scast( __m256i _mUint8, uint16_t * _pu16Dst ) {
			uint8x32_to_xint16x32( _mUint8, _pu16Dst );
		}
		static inline void										uint8_scast( __m256i _mUint8, int32_t * _pi32Dst ) {
			uint8x32_to_xint32x32( _mUint8, reinterpret_cast<uint32_t *>(_pi32Dst) );
		}
		static inline void										uint8_scast( __m256i _mUint8, uint32_t * _pu32Dst ) {
			uint8x32_to_xint32x32( _mUint8, _pu32Dst );
		}
		static inline void										uint8_scast( __m256i _mUint8, int64_t * _pi64Dst ) {
			uint8x32_to_xint64x32( _mUint8, reinterpret_cast<uint64_t *>(_pi64Dst) );
		}
		static inline void										uint8_scast( __m256i _mUint8, uint64_t * _pu64Dst ) {
			uint8x32_to_xint64x32( _mUint8, _pu64Dst );
		}
		static inline void										uint8_scast( __m256i _mUint8, nn9::float16 * _pf16Dst ) {
			__m256 m0, m1, m2, m3;
			uint8x32_to_float32x32( _mUint8, m0, m1, m2, m3 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 8, m1 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 16, m2 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 48, m3 );
		}
		static inline void										uint8_scast( __m256i _mUint8, bfloat16_t * _pf16Dst ) {
			__m256 m0, m1, m2, m3;
			uint8x32_to_float32x32( _mUint8, m0, m1, m2, m3 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 8), m1 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 16), m2 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 24), m3 );
		}
		static inline void										uint8_scast( __m256i _mUint8, float * _pfDst ) {
			__m256 m0, m1, m2, m3;
			uint8x32_to_float32x32( _mUint8, m0, m1, m2, m3 );
			_mm256_storeu_ps( _pfDst, m0 );
			_mm256_storeu_ps( _pfDst + 8, m1 );
			_mm256_storeu_ps( _pfDst + 16, m2 );
			_mm256_storeu_ps( _pfDst + 24, m3 );
		}
		static inline void										uint8_scast( __m256i _mUint8, double * _pdDst ) {
			uint8x32_to_float64x32( _mUint8, _pdDst );
		}
		static inline void										uint8_scast( __m256i _mUint8, bool * _pbDst ) {
			xint8x32_to_boolx32( _mUint8, _pbDst );
		}
		static inline void										uint8_scast( __m256i _mUint8, std::complex<float> * ) {
			throw std::runtime_error( "uint8_scast: No conversion available for uint8_t -> std::complex<float>." );
		}
		static inline void										uint8_scast( __m256i _mUint8, std::complex<double> * ) {
			throw std::runtime_error( "uint8_scast: No conversion available for uint8_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int16_t
		// ===============================
		static inline void										int16_scast( int16_t _i16Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<uint8_t>(std::clamp<int16_t>( _i16Src, INT8_MIN, INT8_MAX ));
		}
		static inline void										int16_scast( int16_t _i16Src, uint8_t &_i8Dst ) {
			_i8Dst = static_cast<uint8_t>(std::clamp<int16_t>( _i16Src, 0, UINT8_MAX ));
		}
		static inline void										int16_scast( int16_t _i16Src, int16_t &_i16Dst ) {
			_i16Dst = _i16Src;
		}
		static inline void										int16_scast( int16_t _i16Src, uint16_t &_i16Dst ) {
			_i16Dst = static_cast<uint32_t>(std::max<int16_t>( _i16Src, 0 ));
		}
		static inline void										int16_scast( int16_t _i16Src, int32_t &_i32Dst ) {
			_i32Dst = _i16Src;
		}
		static inline void										int16_scast( int16_t _i16Src, uint32_t &_i32Dst ) {
			_i32Dst = static_cast<uint32_t>(std::max<int16_t>( _i16Src, 0 ));
		}
		static inline void										int16_scast( int16_t _i16Src, int64_t &_i64Dst ) {
			_i64Dst = _i16Src;
		}
		static inline void										int16_scast( int16_t _i16Src, uint64_t &_i64Dst ) {
			_i64Dst = static_cast<uint32_t>(std::max<int16_t>( _i16Src, 0 ));
		}
		static inline void										int16_scast( int16_t _i16Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(_i16Src);
		}
		static inline void										int16_scast( int16_t _i16Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_i16Src);
		}
		static inline void										int16_scast( int16_t _i16Src, float &_fDst ) {
			_fDst = static_cast<float>(_i16Src);
		}
		static inline void										int16_scast( int16_t _i16Src, double &_dDst ) {
			_dDst = static_cast<double>(_i16Src);
		}
		static inline void										int16_scast( int16_t _i16Src, bool &_bDst ) {
			_bDst = _i16Src != 0;
		}
		static inline void										int16_scast( int16_t _i16Src, std::complex<float> & ) {
			throw std::runtime_error( "int16_scast: No conversion available for int16_t -> std::complex<float>." );
		}
		static inline void										int16_scast( int16_t _i16Src, std::complex<double> & ) {
			throw std::runtime_error( "int16_scast: No conversion available for int16_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										int16_scast( __m512i _mInt16, int8_t * _pi8Dst ) {
			int16x32_to_int8x32_saturated( _mInt16, _pi8Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, uint8_t * _pu8Dst ) {
			int16x32_to_uint8x32_saturated( _mInt16, _pu8Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, int16_t * _pi16Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi16Dst), _mInt16 );
		}
		static inline void										int16_scast( __m512i _mInt16, uint16_t * _pu16Dst ) {
			int16x32_to_uint16x32_saturated( _mInt16, _pu16Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, int32_t * _pi32Dst ) {
			int16x32_to_int32x32( _mInt16, _pi32Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, uint32_t * _pu32Dst ) {
			int16x32_to_uint32x32_saturated( _mInt16, _pu32Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, int64_t * _pi64Dst ) {
			int16x32_to_int64x32( _mInt16, _pi64Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, uint64_t * _pu64Dst ) {
			int16x32_to_uint64x32_saturated( _mInt16, _pu64Dst );
		}
		static inline void										int16_scast( __m512i _mInt16, nn9::float16 * _pf16Dst ) {
			__m512 m0, m1;
			int16x32_to_float32x32( _mInt16, m0, m1 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 16, m1 );
		}
		static inline void										int16_scast( __m512i _mInt16, bfloat16_t * _pf16Dst ) {
			__m512 m0, m1;
			int16x32_to_float32x32( _mInt16, m0, m1 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 16), m1 );
		}
		static inline void										int16_scast( __m512i _mInt16, float * _pfDst ) {
			__m512 m0, m1;
			int16x32_to_float32x32( _mInt16, m0, m1 );
			_mm512_storeu_ps( _pfDst, m0 );
			_mm512_storeu_ps( _pfDst + 16, m1 );
		}
		static inline void										int16_scast( __m512i _mInt16, double * _pdDst ) {
			int16x32_to_float64x32( _mInt16, _pdDst );
			/*__m512 m0, m1;
			int16x32_to_float32x32( _mInt16, m0, m1 );
			NN9_ALIGN( 64 )
			float fTmp[32];
			_mm512_store_ps( fTmp, m0 );
			_mm512_store_ps( fTmp + 16, m1 );
			for ( int i = 0; i < 32; ++i ) {
				(*_pdDst++) = fTmp[i];
			}*/
		}
		static inline void										int16_scast( __m512i _mInt16, bool * _pbDst ) {
			xint16x32_to_boolx32( _mInt16, _pbDst );
		}
		static inline void										int16_scast( __m512i _mInt16, std::complex<float> * ) {
			throw std::runtime_error( "int16_scast: No conversion available for int16_t -> std::complex<float>." );
		}
		static inline void										int16_scast( __m512i _mInt16, std::complex<double> * ) {
			throw std::runtime_error( "int16_scast: No conversion available for int16_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										int16_scast( __m256i _mInt16, int8_t * _pi8Dst ) {
			int16x16_to_int8x16_saturated( _mInt16, _pi8Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, uint8_t * _pu8Dst ) {
			int16x16_to_uint8x16_saturated( _mInt16, _pu8Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, int16_t * _pi16Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi16Dst), _mInt16 );
		}
		static inline void										int16_scast( __m256i _mInt16, uint16_t * _pu16Dst ) {
			int16x16_to_uint16x16_saturated( _mInt16, _pu16Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, int32_t * _pi32Dst ) {
			int16x16_to_int32x16( _mInt16, _pi32Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, uint32_t * _pu32Dst ) {
			int16x16_to_uint32x16_saturated( _mInt16, _pu32Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, int64_t * _pi64Dst ) {
			int16x16_to_int64x16( _mInt16, _pi64Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, uint64_t * _pu64Dst ) {
			int16x16_to_uint64x16_saturated( _mInt16, _pu64Dst );
		}
		static inline void										int16_scast( __m256i _mInt16, nn9::float16 * _pf16Dst ) {
			__m256 m0, m1;
			int16x16_to_float32x16( _mInt16, m0, m1 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 8, m1 );
		}
		static inline void										int16_scast( __m256i _mInt16, bfloat16_t * _pf16Dst ) {
			__m256 m0, m1;
			int16x16_to_float32x16( _mInt16, m0, m1 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf16Dst + 8), m1 );
		}
		static inline void										int16_scast( __m256i _mInt16, float * _pfDst ) {
			__m256 m0, m1;
			int16x16_to_float32x16( _mInt16, m0, m1 );
			_mm256_storeu_ps( _pfDst, m0 );
			_mm256_storeu_ps( _pfDst + 8, m1 );
		}
		static inline void										int16_scast( __m256i _mInt16, double * _pdDst ) {
			int16x16_to_float64x16( _mInt16, _pdDst );
		}
		static inline void										int16_scast( __m256i _mInt16, bool * _pbDst ) {
			xint16x16_to_boolx16( _mInt16, _pbDst );
		}
		static inline void										int16_scast( __m256i _mInt16, std::complex<float> * ) {
			throw std::runtime_error( "int16_scast: No conversion available for int16_t -> std::complex<float>." );
		}
		static inline void										int16_scast( __m256i _mInt16, std::complex<double> * ) {
			throw std::runtime_error( "int16_scast: No conversion available for int16_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__
	};

}	// namespace nn9

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
		 * \brief Converts 32 int16_t values in a __m512i to 32 double values using AVX-512.
		 *
		 * Steps:
		 * 1. Split into two __m256i halves if needed (using _mm512_extracti32x8_epi32).
		 * 2. Convert int16 to int32 with _mm512_cvtepi16_epi32.
		 * 3. From int32, convert to double in chunks of 8 elements using _mm512_cvtepi32_pd on __m256i parts.
		 *
		 * \param _mInt16 Input vector (32 int16_t's).
		 * \param _pdDst Output pointer to 32 double's.
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
		 * \brief Converts 32 uint16_t values in a __m512i to 32 double values using AVX-512.
		 *
		 * Similar to int16, but uses unsigned conversion _mm512_cvtepu16_epi32.
		 *
		 * \param _mXint16 Input vector (32 uint16_t's).
		 * \param _pdDst Output pointer to 32 double's.
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
		 * \brief Converts 32 int16_t to bool using AVX-512.
		 *
		 * Nonzero = 1, zero = 0.
		 *
		 * \param _mInt16 Input vector.
		 * \param _pbDst Output array of 32 bool.
		 */
		static inline void										xint16x32_to_boolx32( __m512i _mInt16, bool * _pbDst ) {
			__mmask32 mMask = _mm512_cmpneq_epi16_mask( _mInt16, _mm512_setzero_si512() );
			__m512i result = _mm512_maskz_mov_epi16( mMask, _mm512_set1_epi16( 1 ) );
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
		 * \brief Converts 16 int16_t to 16 double using AVX2.
		 *
		 * Fallback:
		 * 1. Store 16 int16.
		 * 2. Convert each to int32_t scalar.
		 * 3. Convert in chunks of 4 int32 to double using _mm256_cvtepi32_pd.
		 *
		 * \param _mInt16 Input (16 int16_t)
		 * \param _pdDst Output array of 16 double
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
		 * \brief Converts 16 uint16_t to 16 double using AVX2.
		 *
		 * Same fallback approach, but zero-extension is trivial.
		 *
		 * \param _mUint16 Input (16 uint16_t)
		 * \param _pdDst Output array of 16 double
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
		 * \brief Converts 16 int16_t to bool using AVX2.
		 *
		 * Nonzero = 1, zero = 0.
		 * Compare with zero using _mm256_cmpeq_epi16,
		 * invert, and mMask with ones.
		 *
		 * \param _mInt16 Input (16 int16_t)
		 * \param _pbDst Output array of 16 bool
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
			int32x16_to_int8x16_saturated( _mm512_min_epu32( _mUint16, _mm512_set1_epi32( INT8_MAX ) ), _pi8Dst );
		}

		/**
		 * Casts 16 uint32_t's to 16 uint8_t's with saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint32x16_to_uint8x16_saturated( __m512i _mInt16, uint8_t * _pu8Dst ) {
			int32x16_to_uint8x16_saturated( _mm512_min_epu32( _mInt16, _mm512_set1_epi32( UINT8_MAX ) ), _pu8Dst );
		}

		/**
		 * Casts 16 int32_t's to 16 int8_t's without saturation.
		 * 
		 * \param _mInt8 The values to cast.
		 * \param _pi8Dst The destination buffer.
		 **/
		/*static inline void										int32x16_to_xint8x16( __m512i _mInt16, int8_t * _pi8Dst ) {
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
		}*/

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
			int32x16_to_uint16x16_saturated( _mm512_min_epu32( _mUint32, _mm512_set1_epi32( UINT16_MAX ) ), _pu16Dst );
		}

		/**
		 * Casts 16 int32_t's to 16 int16_t's without saturation.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pi16Dst The destination buffer.
		 **/
		/*static inline void										int32x16_to_int16x16( __m512i _mInt16, int16_t * _pi16Dst ) {
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
		}*/

		/**
		 * Converts 16 int32_t values in a __m512i to 16 uint32_t with saturation.
		 * Negative values are mClamped to 0, positive values remain unchanged.
		 *
		 * \param _mInt32 Input vector containing 16 int32_t.
		 * \param _pu32Dst Output buffer to store 16 uint32_t.
		 */
		static inline void										int32x16_to_uint32x16_saturated( __m512i _mInt32, uint32_t * _pu32Dst ) {
			__m512i mClamped = _mm512_max_epi32( _mInt32, _mm512_setzero_si512() );
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
		static inline void										int32x16_to_int64x16( __m512i _mInt32, int64_t *_pi64Dst ) {
			__m256i mLow32 = _mm512_extracti64x4_epi64( _mInt32, 0 );
			__m256i mUpper32 = _mm512_extracti64x4_epi64( _mInt32, 1 );

			__m512i mLow64 = _mm512_cvtepi32_epi64( mLow32 );
			__m512i mUpper64 = _mm512_cvtepi32_epi64( mUpper32 );

			_mm512_storeu_si512( reinterpret_cast<__m512i *>(&_pi64Dst[0]), mLow64 );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(&_pi64Dst[8]), mUpper64 );
		}

		/**
		 * Converts 16 int32_t to 16 uint64_t (signed extend) using AVX-512.
		 *
		 * \param _mInt32  Input vector of 16 int32_t in __m512i.
		 * \param _pi64Dst Output pointer to store 16 uint64_t.
		 */
		static inline void										int32x16_to_uint64x16_saturated( __m512i _mInt32, uint64_t *_pi64Dst ) {
			__m512i mClamped = _mm512_max_epi32( _mInt32, _mm512_setzero_si512() );
			__m256i mLow32 = _mm512_extracti64x4_epi64( mClamped, 0 );
			__m256i mUpper32 = _mm512_extracti64x4_epi64( mClamped, 1 );

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

		/**
		 * \brief Converts 16 int32_t values in a __m512i to 16 double values using AVX-512.
		 *
		 * Direct approach:
		 * - Extract two __m256i halves (8 int32 each)
		 * - Use _mm512_cvtepi32_pd on each __m256i half
		 *
		 * \param _mInt32 Input vector (16 int32_t).
		 * \param _pdDst Output pointer to 16 double.
		 */
		static inline void										int32x16_to_float64x16( __m512i _mInt32, double * _pdDst ) {
			__m256i mLo32 = _mm512_extracti32x8_epi32( _mInt32, 0 );
			__m256i mHi32 = _mm512_extracti32x8_epi32( _mInt32, 1 );

			__m512d mD0 = _mm512_cvtepi32_pd( mLo32 );
			__m512d mD1 = _mm512_cvtepi32_pd( mHi32 );

			_mm512_storeu_pd( _pdDst, mD0 );
			_mm512_storeu_pd( _pdDst + 8, mD1 );
		}

		/**
		 * \brief Converts 16 uint32_t values in a __m512i to 16 double values using AVX-512.
		 *
		 * Use the same intrinsic _mm512_cvtepi32_pd because the values are unsigned but 
		 * representable in signed domain for conversion. Just treat them as int32_t.
		 *
		 * \param _mUint32 Input vector (16 uint32_t).
		 * \param _pdDst Output pointer to 16 double.
		 */
		static inline void										uint32x16_to_float64x16( __m512i _mUint32, double * _pdDst ) {
			__m256i mLo32 = _mm512_extracti32x8_epi32( _mUint32, 0 );
			__m256i mHi32 = _mm512_extracti32x8_epi32( _mUint32, 1 );

			__m512d mD0 = _mm512_cvtepu32_pd( mLo32 );
			__m512d mD1 = _mm512_cvtepu32_pd( mHi32 );

			_mm512_storeu_pd( _pdDst, mD0 );
			_mm512_storeu_pd( _pdDst + 8, mD1 );
		}

		/**
		 * \brief Converts 16 int32_t to bool using AVX-512.
		 *
		 * \param _mInt32 Input vector.
		 * \param _pbDst Output array of 16 bool.
		 */
		static inline void										xint32x16_to_boolx16( __m512i _mInt32, bool * _pbDst ) {
			__mmask16 mMask = _mm512_cmpneq_epi32_mask(_mInt32, _mm512_setzero_si512() );
			__m512i mRes = _mm512_maskz_mov_epi32( mMask, _mm512_set1_epi32( 1 ) );
			int32x16_to_int8x16_saturated( mRes,  reinterpret_cast<int8_t *>(_pbDst) );
		}
#endif	// #ifdef __AVX512F__

#if defined( __AVX2__ ) || defined( __AVX512F__ )
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
		static inline void										uint32x8_to_int8x8_saturated( __m256i _mUint16, int8_t * _pi8Dst ) {
			int32x8_to_int8x8_saturated( _mm256_min_epu32( _mUint16, _mm256_set1_epi32( 0x7F ) ), _pi8Dst );
		}

		/**
		 * Casts 8 int32_t's to 8 uint8_t's with saturation.
		 * 
		 * \param _mInt16 The values to cast.
		 * \param _pu8Dst The destination buffer.
		 **/
		static inline void										uint32x8_to_uint8x8_saturated( __m256i _mInt16, uint8_t * _pu8Dst ) {
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
			__m256i mClamped = _mm256_max_epi32( _mInt32, _mm256_setzero_si256() );
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
		static inline void										int32x8_to_int64x8( __m256i _mInt32, int64_t * _pi64Dst ) {
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
		 * Converts 8 int32_t's to 8 uint64_t's (signed extend) using AVX2 (and SSE4.1 for the conversion).
		 *
		 * \param _mInt32  Input vector of 8 int32_t in __m256i.
		 * \param _pi64Dst Output pointer to store 8 uint64_t's.
		 */
		static inline void										int32x8_to_uint64x8_saturated( __m256i _mInt32, uint64_t * _pi64Dst ) {
			__m256i mClamped = _mm256_max_epi32( _mInt32, _mm256_setzero_si256() );
			__m128i mLo4 = _mm256_castsi256_si128( mClamped );
			__m128i mHi4 = _mm256_extracti128_si256( mClamped, 1 );

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

		/**
		 * \brief Converts 8 int32_t to 8 double using AVX2 without intermediate store/load.
		 *
		 * Steps:
		 * 1. Extract the low and high 128-bit halves of the __m256i.
		 * 2. Convert each 128-bit half (4 int32) to double using _mm256_cvtepi32_pd.
		 * 3. Store the results.
		 *
		 * \param _mInt32 Input (8 int32_t in __m256i)
		 * \param _pdDst Output array of 8 double
		 */
		static inline void										int32x8_to_float64x8( __m256i _mInt32, double * _pdDst ) {
			__m128i mLow128 = _mm256_castsi256_si128( _mInt32 );
			__m128i mHi128 = _mm256_extractf128_si256( _mInt32, 1 );

			__m256d mD0 = _mm256_cvtepi32_pd( mLow128 );
			__m256d mD1 = _mm256_cvtepi32_pd( mHi128 );

			_mm256_storeu_pd( &_pdDst[0], mD0 );
			_mm256_storeu_pd( &_pdDst[4], mD1 );
		}

		/**
		 * \brief Converts 8 uint32_t to 8 double using AVX2.
		 *
		 * Same approach as int32.
		 *
		 * \param _mUint32 Input (8 uint32_t)
		 * \param _pdDst Output array of 8 double
		 */
		static inline void										uint32x8_to_float64x8( __m256i _mUint32, double * _pdDst ) {
			__m128i mLow128 = _mm256_castsi256_si128( _mUint32 );
			__m128i mHi128 = _mm256_extractf128_si256( _mUint32, 1 );

			__m256d mD0 = _mm256_cvtepu32_pd( mLow128 );
			__m256d mD1 = _mm256_cvtepu32_pd( mHi128 );

			_mm256_storeu_pd( &_pdDst[0], mD0 );
			_mm256_storeu_pd( &_pdDst[4], mD1 );
		}

		/**
		 * \brief Converts 8 int32_t to bool using AVX2.
		 *
		 * \param _mInt32 Input (8 int32_t)
		 * \param _pbDst Output array of 8 bool
		 */
		static inline void										xint32x8_to_boolx8( __m256i _mInt32, bool * _pbDst ) {
			__m256i mCmp = _mm256_cmpeq_epi32( _mInt32, _mm256_setzero_si256() );
			__m256i mNotCmp = _mm256_xor_si256( mCmp, _mm256_set1_epi8( static_cast<char>(0xFF) ) );
			__m256i mRes = _mm256_and_si256( mNotCmp, _mm256_set1_epi32( 1 ) );
			int32x8_to_xint8x8( mRes, reinterpret_cast<int8_t *>(_pbDst) );
		}
#endif	// #if defined( __AVX2__ ) || defined( __AVX512F__ )


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
		 * \brief Casts 8 uint64_t's to 8 uint8_t's with saturation using AVX-512.
		 *
		 *
		 * \param _mUint64 The source values (8 uint64_t in a __m512i).
		 * \param _pu8Dst The destination buffer (8 uint8_t).
		 */
		static inline void										uint64x8_to_uint8x8( __m512i _mUint64, uint8_t * _pu8Dst ) {
			NN9_ALIGN( 64 )
			int64_t i64Tmp[8];
			_mm512_store_si512( reinterpret_cast<__m512i *>(i64Tmp), _mUint64 );

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

		/**
		 * \brief Converts 8 int64_t values in a __m512i to 8 float values using AVX-512.
		 *
		 * Steps:
		 * 1. Convert int64_t to double using _mm512_cvtepi64_pd.
		 * 2. Convert double to float using _mm512_cvtpd_ps.
		 * 
		 * \param mInt64 The source vector (__m512i) containing 8 int64_t values.
		 * \param pfDst Pointer to a float array of size at least 8 where results are stored.
		 */
		static inline void										int64x8_to_float32x8( __m512i _mInt64, __m256 &_mFloatX8 ) {
			__m512d mDouble = _mm512_cvtepi64_pd( _mInt64 );
			_mFloatX8 = _mm512_cvtpd_ps( mDouble );
			//_mm256_storeu_ps( _pfDst, _mFloat );
		}

		/**
		 * \brief Converts 8 uint64_t values in a __m512i to 8 float values using AVX-512.
		 *
		 * Steps:
		 * 1. Convert uint64_t to double using _mm512_cvtepu64_pd.
		 * 2. Convert double to float using _mm512_cvtpd_ps.
		 * 
		 * \param mInt64 The source vector (__m512i) containing 8 uint64_t values.
		 * \param pfDst Pointer to a float array of size at least 8 where results are stored.
		 */
		static inline void										uint64x8_to_float32x8( __m512i _mInt64, __m256 &_mFloatX8 ) {
			__m512d mDouble = _mm512_cvtepu64_pd( _mInt64 );
			_mFloatX8 = _mm512_cvtpd_ps( mDouble );
			//_mm256_storeu_ps( _pfDst, _mFloat );
		}

		/**
		 * \brief Converts 8 int64_t values in a __m512i to 8 double values using AVX-512.
		 *
		 * Directly use _mm512_cvtepi64_pd.
		 *
		 * \param _mInt64 Input vector (8 int64_t).
		 * \param _pdDst Output pointer to 8 double.
		 */
		static inline void										int64x8_to_float64x8( __m512i _mInt64, double *_pdDst ) {
			__m512d mD = _mm512_cvtepi64_pd( _mInt64 );
			_mm512_storeu_pd( _pdDst, mD );
		}

		/**
		 * \brief Converts 8 uint64_t values in a __m512i to 8 double values using AVX-512.
		 *
		 * There's no direct unsigned 64 to double intrinsic. But since all uint64_t fit in signed 64-bit,
		 * _mm512_cvtepi64_pd can be used directly. The interpretation is the same bit-wise,
		 * just ensure values fit in double range if needed.
		 *
		 * \param _mUint64 Input vector (8 uint64_t).
		 * \param _pdDst Output pointer to 8 double.
		 */
		static inline void										uint64x8_to_float64x8( __m512i _mUint64, double * _pdDst ) {
			__m512d mD = _mm512_cvtepu64_pd( _mUint64 );
			_mm512_storeu_pd( _pdDst, mD );
		}

		/**
		 * \brief Converts 8 int64_t to bool using AVX-512.
		 *
		 * \param _mInt64 Input vector.
		 * \param _pbDst Output array of 8 bool.
		 */
		static inline void										xint64x8_to_boolx8( __m512i _mInt64, bool * _pbDst ) {
			__mmask8 mMask = _mm512_cmpneq_epi64_mask( _mInt64, _mm512_setzero_si512() );
			__m512i mRes = _mm512_maskz_mov_epi64( mMask, _mm512_set1_epi64( 1 ) );
			uint64x8_to_uint8x8( _mInt64, reinterpret_cast<uint8_t *>(_pbDst) );
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
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( INT8_MAX ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_set1_epi64x( INT8_MIN ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi8Dst[i] = static_cast<int8_t>(i64Tmp[i]);
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
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( UINT8_MAX ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_setzero_si256() );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pu8Dst[i] = static_cast<uint8_t>(i64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( INT8_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi8Dst[i] = static_cast<int8_t>(ui64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( UINT8_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				_pu8Dst[i] = static_cast<uint8_t>(ui64Tmp[i]);
			}
		}

		/**
		 * \brief Casts 4 int64_t's to 4 int8_t's with saturation using AVX2 (scalar fallback).
		 * This stores the __m256i values to memory, then clamps each value in scalar code.
		 * 
		 * \param _mInt64 The source values (4 int64_t's in __m256i).
		 * \param _pi8Dst The destination buffer (4 int8_t's).
		 */
		static inline void										uint64x4_to_xint8x4( __m256i _mInt64, int8_t * _pi8Dst ) {
			NN9_ALIGN( 32 )
			int64_t i64Tmp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi8Dst[i] = static_cast<int8_t>(i64Tmp[i]);
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
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( INT16_MAX ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_set1_epi64x( INT16_MIN ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi16Dst[i] = static_cast<int16_t>(i64Tmp[i]);
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
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( UINT16_MAX ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_setzero_si256() );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pu16Dst[i] = static_cast<int32_t>(i64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( INT16_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi16Dst[i] = static_cast<int16_t>(ui64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( UINT16_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				_pu16Dst[i] = static_cast<uint16_t>(ui64Tmp[i]);
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
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( INT32_MAX ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_set1_epi64x( INT32_MIN ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi32Dst[i] = static_cast<int32_t>(i64Tmp[i]);
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
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( UINT32_MAX ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_setzero_si256() );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pu32Dst[i] = static_cast<uint32_t>(i64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( UINT32_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				_pi32Dst[i] = static_cast<int32_t>(ui64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( UINT32_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			for ( int i = 0; i < 4; i++ ) {
				auto aVal = ui64Tmp[i];
				_pu32Dst[i] = static_cast<uint32_t>(ui64Tmp[i]);
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
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_setzero_si256() );
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Tmp), _mInt64 );

			for ( int i = 0; i < 4; i++ ) {
				_pu64Dst[i] = static_cast<uint64_t>(i64Tmp[i]);
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
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( INT64_MAX ) );
			_mm256_store_si256( reinterpret_cast<__m256i *>(ui64Tmp), _mUint64 );

			_pi64Dst[0] = static_cast<int64_t>(ui64Tmp[0]);
			_pi64Dst[1] = static_cast<int64_t>(ui64Tmp[1]);
			_pi64Dst[2] = static_cast<int64_t>(ui64Tmp[2]);
			_pi64Dst[3] = static_cast<int64_t>(ui64Tmp[3]);
		}

		/**
		 * \brief Converts 4 int64_t values in a __m256i to 4 float values using AVX2.
		 *
		 * Method:
		 * 1. Store int64_t values to an array.
		 * 2. Convert each scalar to float using a simple cast.
		 * 3. Store results to pfDst.
		 *
		 * This is simpler and often the only method due to lack of direct int64->float intrinsics in AVX2.
		 *
		 * \param mInt64 The source vector (__m256i) with 4 int64_t values.
		 * \param pfDst  Pointer to a float array of size at least 4 for the output.
		 */
		static inline void										int64x4_to_float32x4( __m256i _mInt64, float * _pfDst ) {
			NN9_ALIGN( 32 )
			int64_t i64Temp[4];
			_mm256_store_si256( reinterpret_cast<__m256i *>(i64Temp), _mInt64 );

			_pfDst[0] = static_cast<float>(i64Temp[0]);
			_pfDst[1] = static_cast<float>(i64Temp[1]);
			_pfDst[2] = static_cast<float>(i64Temp[2]);
			_pfDst[3] = static_cast<float>(i64Temp[3]);
		}

		/**
		 * \brief Converts 4 uint64_t values in a __m256i to 4 float values using AVX2.
		 *
		 * Similar fallback as int64:
		 * 1. Store uint64_t to array.
		 * 2. Convert each scalar to float.
		 *
		 * Large values may lose precision. For extremely large numbers, consider a double intermediate.
		 *
		 * \param mUint64 The source vector (__m256i) with 4 uint64_t values.
		 * \param pfDst   Pointer to a float array of size at least 4 for the output.
		 */
		static inline void										uint64x4_to_float32x4( __m256i _mUint64, float * _pfDst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Temp[4];
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(ui64Temp), _mUint64 );

			_pfDst[0] = static_cast<float>(ui64Temp[0]);
			_pfDst[1] = static_cast<float>(ui64Temp[1]);
			_pfDst[2] = static_cast<float>(ui64Temp[2]);
			_pfDst[3] = static_cast<float>(ui64Temp[3]);
		}

		/**
		 * \brief Converts 4 int64_t to 4 double using AVX2.
		 *
		 * No direct 64->double in AVX2. Fallback:
		 * 1. Store 4 int64_t.
		 * 2. Convert each to double in scalar (or SSE):
		 *    - There's no _mm256_cvtepi64_pd in AVX2. We'll do scalar conversion in a loop.
		 *
		 * \param _mInt64 Input (4 int64_t).
		 * \param _pdDst Output array of 4 double.
		 */
		static inline void										int64x4_to_float64x4( __m256i _mInt64, double * _pdDst ) {
			NN9_ALIGN( 32 )
			int64_t i64Temp[4];
			_mm256_store_si256( (__m256i *)i64Temp, _mInt64 );

			_pdDst[0] = static_cast<double>(i64Temp[0]);
			_pdDst[1] = static_cast<double>(i64Temp[1]);
			_pdDst[2] = static_cast<double>(i64Temp[2]);
			_pdDst[3] = static_cast<double>(i64Temp[3]);
		}

		/**
		 * \brief Converts 4 uint64_t to 4 double using AVX2.
		 *
		 * Same fallback approach as int64.
		 *
		 * \param _mUint64 Input (4 uint64_t).
		 * \param _pdDst Output array of 4 double.
		 */
		static inline void										uint64x4_to_float64x4( __m256i _mUint64, double * _pdDst ) {
			NN9_ALIGN( 32 )
			uint64_t ui64Temp[4];
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(ui64Temp), _mUint64 );

			_pdDst[0] = static_cast<double>(ui64Temp[0]);
			_pdDst[1] = static_cast<double>(ui64Temp[1]);
			_pdDst[2] = static_cast<double>(ui64Temp[2]);
			_pdDst[3] = static_cast<double>(ui64Temp[3]);
		}

		/**
		 * \brief Converts 4 int64_t to bool using AVX2.
		 *
		 * _mm256_cmpeq_epi64 is available with AVX2 for 64-bit compares.
		 *
		 * \param _mInt64 Input (4 int64_t).
		 * \param _pbDst Output array of 4 bool.
		 */
		static inline void										xint64x4_to_boolx4( __m256i _mInt64, bool * _pbDst ) {
			__m256i cmp = _mm256_cmpeq_epi64( _mInt64, _mm256_setzero_si256() );
			__m256i notcmp = _mm256_xor_si256( cmp, _mm256_set1_epi8( static_cast<char>(0xFF) ) );
			__m256i result = _mm256_and_si256( notcmp, _mm256_set1_epi64x( 1 ) );
			uint64x4_to_xint8x4( _mInt64, reinterpret_cast<int8_t *>(_pbDst) );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// float/float16/bfloat16_t
		// ===============================
#ifdef __AVX512F__
		/**
		 * \brief Converts 16 floats in a __m512 to 16 int8_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pi8Dst Output array of 16 int8_t's.
		 */
		static inline void										float32x16_to_int8x16( __m512 _mFloat, int8_t * _pi8Dst ) {
			__m512 fMin = _mm512_set1_ps( -128.0f );
			__m512 fMax = _mm512_set1_ps( 127.0f );
			__m512 fClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, fMin ), fMax );

			__m512i mInt32 = _mm512_cvtps_epi32( fClamped );

			__m256i low32 = _mm512_extracti32x8_epi32( mInt32, 0 );
			__m256i high32 = _mm512_extracti32x8_epi32( mInt32, 1 );

			__m128i low16 = _mm_packs_epi32( _mm256_castsi256_si128( low32 ), _mm256_extracti128_si256( low32, 1 ) );
			__m128i high16 = _mm_packs_epi32( _mm256_castsi256_si128( high32 ), _mm256_extracti128_si256( high32, 1 ) );

			__m128i i8_16 = _mm_packs_epi16( low16, high16 );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pi8Dst), i8_16 );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 uint8_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pu8Dst Output array of 16 uint8_t's.
		 */
		static inline void										float32x16_to_uint8x16( __m512 _mFloat, uint8_t * _pu8Dst ) {
			__m512 fZero = _mm512_setzero_ps();
			__m512 fMax = _mm512_set1_ps( 255.0f );
			__m512 fClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, fZero ), fMax );

			__m512i mInt32 = _mm512_cvtps_epi32( fClamped );

			__m256i low32 = _mm512_extracti32x8_epi32( mInt32, 0 );
			__m256i high32 = _mm512_extracti32x8_epi32( mInt32, 1 );

			__m128i low16 = _mm_packus_epi32( _mm256_castsi256_si128( low32 ), _mm256_extracti128_si256( low32, 1 ) );
			__m128i high16 = _mm_packus_epi32( _mm256_castsi256_si128( high32 ), _mm256_extracti128_si256( high32, 1 ) );

			__m128i u8_16 = _mm_packus_epi16( low16, high16 );

			_mm_storeu_si128( reinterpret_cast<__m128i *>(_pu8Dst), u8_16 );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 int16_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pi16Dst Output array of 16 int16_t's.
		 */
		static inline void										float32x16_to_int16x16( __m512 _mFloat, int16_t * _pi16Dst ) {
			__m512 fMin = _mm512_set1_ps( -32768.0f );
			__m512 fMax = _mm512_set1_ps( 32767.0f );
			__m512 fClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, fMin ), fMax );

			__m512i mInt32 = _mm512_cvtps_epi32( fClamped );
			int32x16_to_int16x16_saturated( mInt32, _pi16Dst );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 uint16_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pu16Dst Output array of 16 uint16_t's.
		 */
		static inline void										float32x16_to_uint16x16( __m512 _mFloat, uint16_t * _pu16Dst ) {
			__m512 zero = _mm512_setzero_ps();
			__m512 maxVal = _mm512_set1_ps( 65535.0f );
			__m512 clamped = _mm512_min_ps( _mm512_max_ps( _mFloat, zero ), maxVal );

			__m512i mInt32 = _mm512_cvtps_epi32( clamped );

			int32x16_to_uint16x16_saturated( mInt32, _pu16Dst );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 int32_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pi32Dst Output array of 16 int32_t's.
		 */
		static inline void										float32x16_to_int32x16( __m512 _mFloat, int32_t * _pi32Dst ) {
			__m512 fMin = _mm512_set1_ps( -2147483648.0f );
			__m512 fMax = _mm512_set1_ps( 2147483520.0f );
			__m512 fClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, fMin ), fMax );

			__m512i mInt32 = _mm512_cvtps_epi32( fClamped );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst), mInt32 );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 uint32_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pu32Dst Output array of 16 uint32_t's.
		 */
		static inline void										float32x16_to_uint32x16( __m512 _mFloat, uint32_t * _pu32Dst ) {
			__m512 zero = _mm512_setzero_ps();
			__m512 maxVal = _mm512_set1_ps( static_cast<float>(UINT32_MAX) );
			__m512 clamped = _mm512_min_ps( _mm512_max_ps( _mFloat, zero ), maxVal );

			__m512i mInt32 = _mm512_cvtps_epu32( clamped );
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), mInt32 );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 int64_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pu32Dst Output array of 16 int64_t's.
		 */
		static inline void										float32x16_to_int64x16( __m512 _mFloat, int64_t * _pi64Dst ) {
			__m512 mMin = _mm512_set1_ps( -9223372036854775808.0f );
			__m512 mMax = _mm512_set1_ps( 9223371487098961920.0f );
			__m512 mClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, mMin ), mMax );

			NN9_ALIGN( 64 )
			float fTemp[16];
			_mm512_store_ps( fTemp, mClamped );

			for ( int i = 0; i < 16; i++ ) {
				_pi64Dst[i] = static_cast<int64_t>(fTemp[i]);
			}
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 uint64_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pu32Dst Output array of 16 uint64_t's.
		 */
		static inline void										float32x16_to_uint64x16( __m512 _mFloat, uint64_t * _pu64Dst ) {
			__m512 mMaxVal = _mm512_set1_ps( 18446742974197923840.0f );
			__m512 mClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, _mm512_setzero_ps() ), mMaxVal );

			NN9_ALIGN( 64 )
			float fTemp[16];
			_mm512_store_ps( fTemp, mClamped );

			for ( int i = 0; i < 16; i++ ) {
				_pu64Dst[i] = static_cast<uint64_t>(fTemp[i]);
			}
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 double's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pdDst Output array of 16 double's.
		 */
		static inline void										float32x16_to_float64x16( __m512 _mFloat, double * _pdDst ) {
			__m256 mLow = _mm512_castps512_ps256( _mFloat );
			__m256 mHi = _mm512_extractf32x8_ps( _mFloat, 1 );

			__m512d mD0 = _mm512_cvtps_pd( mLow );
			__m512d mD1 = _mm512_cvtps_pd( mHi );

			_mm512_storeu_pd( _pdDst, mD0 );
			_mm512_storeu_pd( _pdDst + 8, mD1 );
		}

		/**
		 * \brief Converts 16 floats in a __m512 to 16 bool's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pbDst Output array of 16 bool's.
		 */
		static inline void										float32x16_to_boolx16( __m512 _mFloat, bool * _pbDst ) {
			__m512 fMin = _mm512_set1_ps( -1.0f );
			__m512 fMax = _mm512_set1_ps( 1.0f );
			__m512 fClamped = _mm512_min_ps( _mm512_max_ps( _mFloat, fMin ), fMax );

			__m512i mInt32 = _mm512_cvtps_epi32( fClamped );
			xint32x16_to_boolx16( mInt32, _pbDst );
		}

#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * \brief Converts 8 floats in a __m512 to 8 int8_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pi8Dst Output array of 8 int8_t's.
		 */
		static inline void										float32x8_to_int8x8( __m256 _mFloat, int8_t * _pi8Dst ) {
			__m256 fMin = _mm256_set1_ps( static_cast<float>(INT8_MIN) );
			__m256 fMax = _mm256_set1_ps( static_cast<float>(INT8_MAX) );
			__m256 mClamped = _mm256_min_ps( _mm256_max_ps( _mFloat, fMin ), fMax );

			int32x8_to_int8x8_saturated( _mm256_cvtps_epi32( mClamped ), _pi8Dst );
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 uint8_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pu8Dst Output array of 8 uint8_t's.
		 */
		static inline void										float32x8_to_uint8x8( __m256 _mFloat, uint8_t * _pu8Dst ) {
			__m256 fMax = _mm256_set1_ps( static_cast<float>(UINT8_MAX) );
			__m256 mClamped = _mm256_min_ps(_mm256_max_ps( _mFloat, _mm256_setzero_ps() ), fMax );

			int32x8_to_uint8x8_saturated( _mm256_cvtps_epu32( mClamped ), _pu8Dst );
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 int16_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pi16Dst Output array of 8 int16_t's.
		 */
		static inline void										float32x8_to_int16x8( __m256 _mFloat, int16_t * _pi16Dst ) {
			__m256 fMin = _mm256_set1_ps( static_cast<float>(INT16_MIN) );
			__m256 fMax = _mm256_set1_ps( static_cast<float>(INT16_MAX) );
			__m256 mClamped = _mm256_min_ps( _mm256_max_ps( _mFloat, fMin ), fMax );

			int32x8_to_int16x8_saturated( _mm256_cvtps_epi32( mClamped ), _pi16Dst );
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 uint16_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pu16Dst Output array of 8 uint16_t's.
		 */
		static inline void										float32x8_to_uint16x8( __m256 _mFloat, uint16_t * _pu16Dst ) {
			__m256 fMax = _mm256_set1_ps( static_cast<float>(UINT16_MAX) );
			__m256 mClamped = _mm256_min_ps(_mm256_max_ps( _mFloat, _mm256_setzero_ps() ), fMax );

			int32x8_to_uint16x8_saturated( _mm256_cvtps_epu32( mClamped ), _pu16Dst );
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 int32_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pi32Dst Output array of 8 int32_t's.
		 */
		static inline void										float32x8_to_int32x8( __m256 _mFloat, int32_t * _pi32Dst ) {
			__m256 fMin = _mm256_set1_ps( -2147483648.0f );
			__m256 fMax = _mm256_set1_ps( 2147483520.0f );
			__m256 mClamped = _mm256_min_ps( _mm256_max_ps( _mFloat, fMin ), fMax );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst), _mm256_cvtps_epi32( mClamped ) );
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 uint32_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pu32Dst Output array of 8 uint32_t's.
		 */
		static inline void										float32x8_to_uint32x8( __m256 _mFloat, uint32_t * _pu32Dst ) {
			__m256 fMax = _mm256_set1_ps( static_cast<float>(UINT32_MAX) );
			__m256 mClamped = _mm256_min_ps(_mm256_max_ps( _mFloat, _mm256_setzero_ps() ), fMax );

			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), _mm256_cvtps_epu32( mClamped ) );
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 int64_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pi64Dst Output array of 8 int64_t's.
		 */
		static inline void										float32x8_to_int64x8( __m256 _mFloat, int64_t * _pi64Dst ) {
			__m256 fMin = _mm256_set1_ps( -9223372036854775808.0f );
			__m256 fMax = _mm256_set1_ps( 9223371487098961920.0f );
			__m256 mClamped = _mm256_min_ps( _mm256_max_ps( _mFloat, fMin ), fMax );

			NN9_ALIGN( 32 )
			float fTemp[8];
			_mm256_store_ps( fTemp, mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pi64Dst[i] = static_cast<int64_t>(fTemp[i]);
			}
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 uint64_t's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pu64Dst Output array of 8 uint64_t's.
		 */
		static inline void										float32x8_to_uint64x8( __m256 _mFloat, uint64_t * _pu64Dst ) {
			__m256 fMax = _mm256_set1_ps( 18446742974197923840.0f );
			__m256 mClamped = _mm256_min_ps(_mm256_max_ps( _mFloat, _mm256_setzero_ps() ), fMax );

			NN9_ALIGN( 32 )
			float fTemp[8];
			_mm256_store_ps( fTemp, mClamped );

			for ( int i = 0; i < 8; i++ ) {
				_pu64Dst[i] = static_cast<uint64_t>(fTemp[i]);
			}
		}

		/**
		 * \brief Converts 8 floats in a __m512 to 8 double's with saturation using AVX-512.
		 * 
		 * \param _mFloat Input vector with 8 floats.
		 * \param _pdDst Output array of 8 double's.
		 */
		static inline void										float32x8_to_float64x8( __m256 _mFloat, double * _pdDst ) {
			__m128 low4 = _mm256_castps256_ps128( _mFloat );
			__m128 high4 = _mm256_extractf128_ps( _mFloat, 1 );

			__m256d d0 = _mm256_cvtps_pd( low4 );
			__m256d d1 = _mm256_cvtps_pd( high4 );

			_mm256_storeu_pd( _pdDst, d0 );
			_mm256_storeu_pd( _pdDst + 4, d1 );
		}

		/**
		 * \brief Converts 8 floats in a __m256 to 8 bool's with saturation using AVX-256.
		 * 
		 * \param _mFloat Input vector with 16 floats.
		 * \param _pbDst Output array of 8 bool's.
		 */
		static inline void										float32x8_to_boolx8( __m256 _mFloat, bool * _pbDst ) {
			__m256 fMin = _mm256_set1_ps( -1.0f );
			__m256 fMax = _mm256_set1_ps( 1.0f );
			__m256 fClamped = _mm256_min_ps( _mm256_max_ps( _mFloat, fMin ), fMax );

			__m256i mInt32 = _mm256_cvtps_epi32( fClamped );
			xint32x8_to_boolx8( mInt32, _pbDst );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// double
		// ===============================
#ifdef __AVX512F__
		/**
		 * \brief Converts 8 doubles in a __m512d to 8 int8_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pi8Dst Output array of 8 int8_t's.
		 */
		static inline void										float64x8_to_int8x8( __m512d _mDouble, int8_t * _pi8Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_set1_pd( static_cast<double>(INT8_MIN) ) ), _mm512_set1_pd( static_cast<double>(INT8_MAX) ) );

			__m256i mI32 = _mm512_cvtpd_epi32( mClamped );

			int32x8_to_int8x8_saturated( mI32, _pi8Dst );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 uint8_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pu8Dst Output array of 8 uint8_t's.
		 */
		static inline void										float64x8_to_uint8x8( __m512d _mDouble, uint8_t * _pu8Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_setzero_pd() ), _mm512_set1_pd( static_cast<double>(UINT8_MAX) ) );

			__m256i mI32 = _mm512_cvtpd_epi32( mClamped );

			int32x8_to_uint8x8_saturated( mI32, _pu8Dst );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 int16_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pi16Dst Output array of 8 int16_t's.
		 */
		static inline void										float64x8_to_int16x8( __m512d _mDouble, int16_t * _pi16Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_set1_pd( static_cast<double>(INT16_MIN) ) ), _mm512_set1_pd( static_cast<double>(INT16_MAX) ) );

			__m256i mI32 = _mm512_cvtpd_epi32( mClamped );

			int32x8_to_int16x8_saturated( mI32, _pi16Dst );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 uint16_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pu16Dst Output array of 8 uint16_t's.
		 */
		static inline void										float64x8_to_uint16x8( __m512d _mDouble, uint16_t * _pu16Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_setzero_pd() ), _mm512_set1_pd( static_cast<double>(UINT16_MAX) ) );

			__m256i mI32 = _mm512_cvtpd_epu32( mClamped );

			uint32x8_to_uint16x8_saturated( mI32, _pu16Dst );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 int32_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pi32Dst Output array of 8 int32_t's.
		 */
		static inline void										float64x8_to_int32x8( __m512d _mDouble, int32_t * _pi32Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_set1_pd( static_cast<double>(INT32_MIN) ) ), _mm512_set1_pd( static_cast<double>(INT32_MAX) ) );

			__m256i mI32 = _mm512_cvtpd_epi32( mClamped );

			_mm256_storeu_epi32( _pi32Dst, mI32 );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 uint32_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pu32Dst Output array of 8 uint32_t's.
		 */
		static inline void										float64x8_to_uint32x8( __m512d _mDouble, uint32_t * _pu32Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_setzero_pd() ), _mm512_set1_pd( static_cast<double>(UINT32_MAX) ) );

			__m256i mI32 = _mm512_cvtpd_epu32( mClamped );

			_mm256_storeu_epi32( _pu32Dst, mI32 );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 int64_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pi64Dst Output array of 8 int64_t's.
		 */
		static inline void										float64x8_to_int64x8( __m512d _mDouble, int64_t * _pi64Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_set1_pd( static_cast<double>(INT64_MIN) ) ), _mm512_set1_pd( 9223372036854774784.0 ) );

			__m512i i64 = _mm512_cvttpd_epi64( mClamped );
		    _mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst), i64 );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 uint64_t's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pu64Dst Output array of 8 uint64_t's.
		 */
		static inline void										float64x8_to_uint64x8( __m512d _mDouble, uint64_t * _pu64Dst ) {
			__m512d mClamped = _mm512_min_pd( _mm512_max_pd( _mDouble, _mm512_setzero_pd() ), _mm512_set1_pd( 18446744073709549568.0 ) );

			__m512i i64 = _mm512_cvttpd_epu64( mClamped );
		    _mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst), i64 );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 float's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pfDst Output array of 8 float's.
		 */
		static inline void										float64x8_to_float32x8( __m512d _mDouble, float * _pfDst ) {
			_mm256_storeu_ps( _pfDst, _mm512_cvtpd_ps( _mDouble ) );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 float's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _fDst Output array of 8 float's stored in a __m256 register.
		 */
		static inline void										float64x8_to_float32x8( __m512d _mDouble, __m256 &_fDst ) {
			_fDst = _mm512_cvtpd_ps( _mDouble );
		}

		/**
		 * \brief Converts 8 doubles in a __m512d to 8 bool's using AVX-512.
		 * 
		 * \param _mDouble Input vector with 8 doubles.
		 * \param _pbDst Output array of 8 bool's.
		 */
		static inline void										float64x8_to_boolx8( __m512d _mDouble, bool * _pbDst ) {
			__mmask8 mMask = _mm512_cmp_pd_mask( _mDouble, _mm512_setzero_pd(), _CMP_NEQ_OQ );

			for ( int i = 0; i < 8; i++ ) {
				_pbDst[i] = (mMask & (1 << i)) != 0;
			}
		}


#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * \brief Converts 4 doubles in a __m256d to 4 int8_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pi8Dst Output array of 4 int8_t's.
		 */
		static inline void										float64x4_to_int8x4( __m256d _mDouble, int8_t * _pi8Dst ) {
			__m256d dMin = _mm256_set1_pd( static_cast<double>(INT8_MIN) );
			__m256d dMax = _mm256_set1_pd( static_cast<double>(INT8_MAX) );
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, dMin ), dMax );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pi8Dst[i] = static_cast<int8_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 uint8_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pu8Dst Output array of 4 uint8_t's.
		 */
		static inline void										float64x4_to_uint8x4( __m256d _mDouble, uint8_t * _pu8Dst ) {
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, _mm256_setzero_pd() ), _mm256_set1_pd( static_cast<double>(UINT8_MAX) ) );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pu8Dst[i] = static_cast<uint8_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 int16_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pi16Dst Output array of 4 int16_t's.
		 */
		static inline void										float64x4_to_int16x4( __m256d _mDouble, int16_t * _pi16Dst ) {
			__m256d dMin = _mm256_set1_pd( static_cast<double>(INT16_MIN) );
			__m256d dMax = _mm256_set1_pd( static_cast<double>(INT16_MAX) );
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, dMin ), dMax );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pi16Dst[i] = static_cast<int16_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 uint16_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pu16Dst Output array of 4 uint16_t's.
		 */
		static inline void										float64x4_to_uint16x4( __m256d _mDouble, uint16_t * _pu16Dst ) {
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, _mm256_setzero_pd() ), _mm256_set1_pd( static_cast<double>(UINT16_MAX) ) );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pu16Dst[i] = static_cast<uint16_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 int32_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pi32Dst Output array of 4 int32_t's.
		 */
		static inline void										float64x4_to_int32x4( __m256d _mDouble, int32_t * _pi32Dst ) {
			__m256d dMin = _mm256_set1_pd( static_cast<double>(INT32_MIN) );
			__m256d dMax = _mm256_set1_pd( static_cast<double>(INT32_MAX) );
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, dMin ), dMax );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pi32Dst[i] = static_cast<int32_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 uint32_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pu32Dst Output array of 4 uint32_t's.
		 */
		static inline void										float64x4_to_uint32x4( __m256d _mDouble, uint32_t * _pu32Dst ) {
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, _mm256_setzero_pd() ), _mm256_set1_pd( static_cast<double>(UINT32_MAX) ) );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pu32Dst[i] = static_cast<uint32_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 int64_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pi64Dst Output array of 4 int64_t's.
		 */
		static inline void										float64x4_to_int64x4( __m256d _mDouble, int64_t * _pi64Dst ) {
			__m256d dMin = _mm256_set1_pd( static_cast<double>(INT64_MIN) );
			__m256d dMax = _mm256_set1_pd( 9223372036854774784.0 );
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, dMin ), dMax );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pi64Dst[i] = static_cast<int64_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 uint64_t's with saturation using AVX2.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pu64Dst Output array of 4 uint64_t's.
		 */
		static inline void										float64x4_to_uint64x4( __m256d _mDouble, uint64_t * _pu64Dst ) {
			__m256d mClamped = _mm256_min_pd( _mm256_max_pd( _mDouble, _mm256_setzero_pd() ), _mm256_set1_pd( 18446744073709549568.0 ) );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mClamped );

			for ( int i = 0; i < 4; i++ ) {
				_pu64Dst[i] = static_cast<uint64_t>(dTemp[i]);
			}
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 float's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pfDst Output array of 4 float's.
		 */
		static inline void										float64x4_to_float32x4( __m256d _mDouble, float * _pfDst ) {
			_mm_storeu_ps( _pfDst, _mm256_cvtpd_ps( _mDouble ) );
		}

		/**
		 * \brief Converts 4 doubles in a __m256d to 4 float's with saturation using AVX-512.
		 * 
		 * \param _mDouble Input vector with 4 doubles.
		 * \param _pfDst Output array of 4 float's.
		 */
		static inline void										float64x4_to_float32x4( __m256d _mDouble, __m128 &_mDst ) {
			_mDst = _mm256_cvtpd_ps( _mDouble );
		}

		static inline void										float64x4_to_boolx4( __m256d _mDouble, bool * _pbDst ) {
			__m256d mCmp = _mm256_cmp_pd( _mDouble, _mm256_setzero_pd(), _CMP_NEQ_OQ );

			NN9_ALIGN( 32 )
			double dTemp[4];
			_mm256_storeu_pd( dTemp, mCmp );

			for (int i = 0; i < 4; i++) {
				_pbDst[i] = (dTemp[i] != 0.0);
			}
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int8_t
		// ===============================
		static inline void										scast( int8_t _i8Src, int8_t &_i8Dst ) {
			_i8Dst = _i8Src;
		}
		static inline void										scast( int8_t _i8Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										scast( int8_t _i8Src, int16_t &_i16Dst ) {
			_i16Dst = _i8Src;
		}
		static inline void										scast( int8_t _i8Src, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										scast( int8_t _i8Src, int32_t &_i32Dst ) {
			_i32Dst = _i8Src;
		}
		static inline void										scast( int8_t _i8Src, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										scast( int8_t _i8Src, int64_t &_i64Dst ) {
			_i64Dst = _i8Src;
		}
		static inline void										scast( int8_t _i8Src, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::max<int8_t>( _i8Src, 0 ));
		}
		static inline void										scast( int8_t _i8Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(_i8Src);
		}
		static inline void										scast( int8_t _i8Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_i8Src);
		}
		static inline void										scast( int8_t _i8Src, float &_fDst ) {
			_fDst = static_cast<float>(_i8Src);
		}
		static inline void										scast( int8_t _i8Src, double &_dDst ) {
			_dDst = static_cast<double>(_i8Src);
		}
		static inline void										scast( int8_t _i8Src, bool &_bDst ) {
			_bDst = _i8Src != 0;
		}
		static inline void										scast( int8_t _i8Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for int8_t -> std::complex<float>." );
		}
		static inline void										scast( int8_t _i8Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for int8_t -> std::complex<double>." );
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
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 16), m1 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 32), m2 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 48), m3 );
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
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 8), m1 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 16), m2 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 24), m3 );
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
		static inline void										scast( uint8_t _u8Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::min<uint8_t>( _u8Src, INT8_MAX ));
		}
		static inline void										scast( uint8_t _u8Src, uint8_t &_u8Dst ) {
			_u8Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, int16_t &_i16Dst ) {
			_i16Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, uint16_t &_u16Dst ) {
			_u16Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, int32_t &_i32Dst ) {
			_i32Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, uint32_t &_u32Dst ) {
			_u32Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, int64_t &_i64Dst ) {
			_i64Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, uint64_t &_u64Dst ) {
			_u64Dst = _u8Src;
		}
		static inline void										scast( uint8_t _u8Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(_u8Src);
		}
		static inline void										scast( uint8_t _u8Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_u8Src);
		}
		static inline void										scast( uint8_t _u8Src, float &_fDst ) {
			_fDst = static_cast<float>(_u8Src);
		}
		static inline void										scast( uint8_t _u8Src, double &_dDst ) {
			_dDst = static_cast<double>(_u8Src);
		}
		static inline void										scast( uint8_t _u8Src, bool &_bDst ) {
			_bDst = _u8Src != 0;
		}
		static inline void										scast( uint8_t _u8Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for uint8_t -> std::complex<float>." );
		}
		static inline void										scast( uint8_t _u8Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for uint8_t -> std::complex<double>." );
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
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 16), m1 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 32), m2 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 48), m3 );
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
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 8), m1 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 16), m2 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 24), m3 );
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
		static inline void										scast( int16_t _i16Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<int16_t>( _i16Src, INT8_MIN, INT8_MAX ));
		}
		static inline void										scast( int16_t _i16Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<int16_t>( _i16Src, 0, UINT8_MAX ));
		}
		static inline void										scast( int16_t _i16Src, int16_t &_i16Dst ) {
			_i16Dst = _i16Src;
		}
		static inline void										scast( int16_t _i16Src, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint32_t>(std::max<int16_t>( _i16Src, 0 ));
		}
		static inline void										scast( int16_t _i16Src, int32_t &_i32Dst ) {
			_i32Dst = _i16Src;
		}
		static inline void										scast( int16_t _i16Src, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::max<int16_t>( _i16Src, 0 ));
		}
		static inline void										scast( int16_t _i16Src, int64_t &_i64Dst ) {
			_i64Dst = _i16Src;
		}
		static inline void										scast( int16_t _i16Src, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint32_t>(std::max<int16_t>( _i16Src, 0 ));
		}
		static inline void										scast( int16_t _i16Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(_i16Src);
		}
		static inline void										scast( int16_t _i16Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_i16Src);
		}
		static inline void										scast( int16_t _i16Src, float &_fDst ) {
			_fDst = static_cast<float>(_i16Src);
		}
		static inline void										scast( int16_t _i16Src, double &_dDst ) {
			_dDst = static_cast<double>(_i16Src);
		}
		static inline void										scast( int16_t _i16Src, bool &_bDst ) {
			_bDst = _i16Src != 0;
		}
		static inline void										scast( int16_t _i16Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for int16_t -> std::complex<float>." );
		}
		static inline void										scast( int16_t _i16Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for int16_t -> std::complex<double>." );
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
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 16), m1 );
		}
		static inline void										int16_scast( __m512i _mInt16, float * _pfDst ) {
			__m512 m0, m1;
			int16x32_to_float32x32( _mInt16, m0, m1 );
			_mm512_storeu_ps( _pfDst, m0 );
			_mm512_storeu_ps( _pfDst + 16, m1 );
		}
		static inline void										int16_scast( __m512i _mInt16, double * _pdDst ) {
			int16x32_to_float64x32( _mInt16, _pdDst );
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
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 8), m1 );
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


		// ===============================
		// uint16_t
		// ===============================
		static inline void										scast( uint16_t _u16Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::min<uint16_t>( _u16Src, INT8_MAX ));
		}
		static inline void										scast( uint16_t _u16Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::min<uint16_t>( _u16Src, UINT8_MAX ));
		}
		static inline void										scast( uint16_t _u16Src, int16_t &_i16Dst ) {
			_i16Dst = std::min<uint16_t>( _u16Src, INT16_MAX );
		}
		static inline void										scast( uint16_t _u16Src, uint16_t &_u16Dst ) {
			_u16Dst = _u16Dst;
		}
		static inline void										scast( uint16_t _u16Src, int32_t &_i32Dst ) {
			_i32Dst = _u16Src;
		}
		static inline void										scast( uint16_t _u16Src, uint32_t &_u32Dst ) {
			_u32Dst = _u16Src;
		}
		static inline void										scast( uint16_t _u16Src, int64_t &_i64Dst ) {
			_i64Dst = _u16Src;
		}
		static inline void										scast( uint16_t _u16Src, uint64_t &_u64Dst ) {
			_u64Dst = _u16Src;
		}
		static inline void										scast( uint16_t _u16Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(std::min<uint16_t>( _u16Src, 65504 ));
		}
		static inline void										scast( uint16_t _u16Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_u16Src);
		}
		static inline void										scast( uint16_t _u16Src, float &_fDst ) {
			_fDst = static_cast<float>(_u16Src);
		}
		static inline void										scast( uint16_t _u16Src, double &_dDst ) {
			_dDst = static_cast<double>(_u16Src);
		}
		static inline void										scast( uint16_t _u16Src, bool &_bDst ) {
			_bDst = _u16Src != 0;
		}
		static inline void										scast( uint16_t _u16Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for uint16_t -> std::complex<float>." );
		}
		static inline void										scast( uint16_t _u16Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for uint16_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										uint16_scast( __m512i _mUint8, int8_t * _pi8Dst ) {
			uint16x32_to_int8x32_saturated( _mUint8, _pi8Dst );
		}
		static inline void										uint16_scast( __m512i _mUint8, uint8_t * _pu8Dst ) {
			uint16x32_to_uint8x32_saturated( _mUint8, _pu8Dst );
		}
		static inline void										uint16_scast( __m512i _mUint8, int16_t * _pi16Dst ) {
			uint16x32_to_int16x32_saturated( _mUint8, _pi16Dst );
		}
		static inline void										uint16_scast( __m512i _mUint8, uint16_t * _pu16Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu16Dst), _mUint8 );
		}
		static inline void										uint16_scast( __m512i _mUint8, int32_t * _pi32Dst ) {
			uint16x32_to_xint32x32( _mUint8, reinterpret_cast<uint32_t *>(_pi32Dst) );
		}
		static inline void										uint16_scast( __m512i _mUint8, uint32_t * _pu32Dst ) {
			uint16x32_to_xint32x32( _mUint8, _pu32Dst );
		}
		static inline void										uint16_scast( __m512i _mUint8, int64_t * _pi64Dst ) {
			uint16x32_to_xint64x32( _mUint8, reinterpret_cast<uint64_t *>(_pi64Dst) );
		}
		static inline void										uint16_scast( __m512i _mUint8, uint64_t * _pu64Dst ) {
			uint16x32_to_xint64x32( _mUint8, _pu64Dst );
		}
		static inline void										uint16_scast( __m512i _mUint8, nn9::float16 * _pf16Dst ) {
			__m512 m0, m1;
			uint16x32_to_float32x32( _mUint8, m0, m1 );
			__m512 mMax = _mm512_set1_ps( 65504.0f );
			m0 = _mm512_min_ps( m0, mMax );
			m1 = _mm512_min_ps( m1, mMax );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst + 16, m1 );
		}
		static inline void										uint16_scast( __m512i _mUint8, bfloat16_t * _pf16Dst ) {
			__m512 m0, m1;
			uint16x32_to_float32x32( _mUint8, m0, m1 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 16), m1 );
		}
		static inline void										uint16_scast( __m512i _mUint8, float * _pfDst ) {
			__m512 m0, m1;
			uint16x32_to_float32x32( _mUint8, m0, m1 );
			_mm512_storeu_ps( _pfDst, m0 );
			_mm512_storeu_ps( _pfDst + 16, m1 );
		}
		static inline void										uint16_scast( __m512i _mUint8, double * _pdDst ) {
			uint16x32_to_float64x32( _mUint8, _pdDst );
		}
		static inline void										uint16_scast( __m512i _mUint8, bool * _pbDst ) {
			xint16x32_to_boolx32( _mUint8, _pbDst );
		}
		static inline void										uint16_scast( __m512i _mUint8, std::complex<float> * ) {
			throw std::runtime_error( "uint16_scast: No conversion available for uint16_t -> std::complex<float>." );
		}
		static inline void										uint16_scast( __m512i _mUint8, std::complex<double> * ) {
			throw std::runtime_error( "uint16_scast: No conversion available for uint16_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										uint16_scast( __m256i _mUint8, int8_t * _pi8Dst ) {
			uint16x16_to_int8x16_saturated( _mUint8, _pi8Dst );
		}
		static inline void										uint16_scast( __m256i _mUint8, uint8_t * _pu8Dst ) {
			uint16x16_to_uint8x16_saturated( _mUint8, _pu8Dst );
		}
		static inline void										uint16_scast( __m256i _mUint8, int16_t * _pi16Dst ) {
			uint16x16_to_int16x16_saturated( _mUint8, _pi16Dst );
		}
		static inline void										uint16_scast( __m256i _mUint8, uint16_t * _pu16Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu16Dst), _mUint8 );
		}
		static inline void										uint16_scast( __m256i _mUint8, int32_t * _pi32Dst ) {
			uint16x16_to_xint32x16( _mUint8, reinterpret_cast<uint32_t *>(_pi32Dst) );
		}
		static inline void										uint16_scast( __m256i _mUint8, uint32_t * _pu32Dst ) {
			uint16x16_to_xint32x16( _mUint8, _pu32Dst );
		}
		static inline void										uint16_scast( __m256i _mUint8, int64_t * _pi64Dst ) {
			uint16x16_to_xint64x16( _mUint8, reinterpret_cast<uint64_t *>(_pi64Dst) );
		}
		static inline void										uint16_scast( __m256i _mUint8, uint64_t * _pu64Dst ) {
			uint16x16_to_xint64x16( _mUint8, _pu64Dst );
		}
		static inline void										uint16_scast( __m256i _mUint8, nn9::float16 * _pf16Dst ) {
			__m256 m0, m1;
			uint16x16_to_float32x16( _mUint8, m0, m1 );
			__m256 mMax = _mm256_set1_ps( 65504.0f );
			m0 = _mm256_min_ps( m0, mMax );
			m1 = _mm256_min_ps( m1, mMax );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst + 8, m1 );
		}
		static inline void										uint16_scast( __m256i _mUint8, bfloat16_t * _pf16Dst ) {
			__m256 m0, m1;
			uint16x16_to_float32x16( _mUint8, m0, m1 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst + 8), m1 );
		}
		static inline void										uint16_scast( __m256i _mUint8, float * _pfDst ) {
			__m256 m0, m1;
			uint16x16_to_float32x16( _mUint8, m0, m1 );
			_mm256_storeu_ps( _pfDst, m0 );
			_mm256_storeu_ps( _pfDst + 8, m1 );
		}
		static inline void										uint16_scast( __m256i _mUint8, double * _pdDst ) {
			uint16x16_to_float64x16( _mUint8, _pdDst );
		}
		static inline void										uint16_scast( __m256i _mUint8, bool * _pbDst ) {
			xint16x16_to_boolx16( _mUint8, _pbDst );
		}
		static inline void										uint16_scast( __m256i _mUint8, std::complex<float> * ) {
			throw std::runtime_error( "uint16_scast: No conversion available for uint16_t -> std::complex<float>." );
		}
		static inline void										uint16_scast( __m256i _mUint8, std::complex<double> * ) {
			throw std::runtime_error( "uint16_scast: No conversion available for uint16_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int32_t
		// ===============================
		static inline void										scast( int32_t _i32Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<int32_t>( _i32Src, INT8_MIN, INT8_MAX ));
		}
		static inline void										scast( int32_t _i32Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<int32_t>( _i32Src, 0, UINT8_MAX ));
		}
		static inline void										scast( int32_t _i32Src, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::clamp<int32_t>( _i32Src, INT16_MIN, INT16_MAX ));
		}
		static inline void										scast( int32_t _i32Src, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::clamp<int32_t>( _i32Src, 0, UINT16_MAX ));
		}
		static inline void										scast( int32_t _i32Src, int32_t &_i32Dst ) {
			_i32Dst = _i32Src;
		}
		static inline void										scast( int32_t _i32Src, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::max<int32_t>( _i32Src, 0 ));
		}
		static inline void										scast( int32_t _i32Src, int64_t &_i64Dst ) {
			_i64Dst = _i32Src;
		}
		static inline void										scast( int32_t _i32Src, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::max<int32_t>( _i32Src, 0 ));
		}
		static inline void										scast( int32_t _i32Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(std::clamp<float>( static_cast<float>(_i32Src), -65504.0f, 65504.0f ));
		}
		static inline void										scast( int32_t _i32Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_i32Src);
		}
		static inline void										scast( int32_t _i32Src, float &_fDst ) {
			_fDst = static_cast<float>(_i32Src);
		}
		static inline void										scast( int32_t _i32Src, double &_dDst ) {
			_dDst = static_cast<double>(_i32Src);
		}
		static inline void										scast( int32_t _i32Src, bool &_bDst ) {
			_bDst = _i32Src != 0;
		}
		static inline void										scast( int32_t _i32Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for int32_t -> std::complex<float>." );
		}
		static inline void										scast( int32_t _i32Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for int32_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										int32_scast( __m512i _mInt32, int8_t * _pi8Dst ) {
			int32x16_to_int8x16_saturated( _mInt32, _pi8Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, uint8_t * _pu8Dst ) {
			int32x16_to_uint8x16_saturated( _mInt32, _pu8Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, int16_t * _pi16Dst ) {
			int32x16_to_int16x16_saturated( _mInt32, _pi16Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, uint16_t * _pu16Dst ) {
			int32x16_to_uint16x16_saturated( _mInt32, _pu16Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, int32_t * _pi32Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi32Dst), _mInt32 );
		}
		static inline void										int32_scast( __m512i _mInt32, uint32_t * _pu32Dst ) {
			int32x16_to_uint32x16_saturated( _mInt32, _pu32Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, int64_t * _pi64Dst ) {
			int32x16_to_int64x16( _mInt32, _pi64Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, uint64_t * _pu64Dst ) {
			int32x16_to_uint64x16_saturated( _mInt32, _pu64Dst );
		}
		static inline void										int32_scast( __m512i _mInt32, nn9::float16 * _pf16Dst ) {
			__m512 m0;
			int32x16_to_float32x16( _mInt32, m0 );
			m0 = _mm512_min_ps( m0, _mm512_set1_ps( 65504.0f ) );
			m0 = _mm512_max_ps( m0, _mm512_set1_ps( -65504.0f ) );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst, m0 );
		}
		static inline void										int32_scast( __m512i _mInt32, bfloat16_t * _pf16Dst ) {
			__m512 m0;
			int32x16_to_float32x16( _mInt32, m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
		}
		static inline void										int32_scast( __m512i _mInt32, float * _pfDst ) {
			__m512 m0;
			int32x16_to_float32x16( _mInt32, m0 );
			_mm512_storeu_ps( _pfDst, m0 );
		}
		static inline void										int32_scast( __m512i _mInt32, double * _pdDst ) {
			int32x16_to_float64x16( _mInt32, _pdDst );
		}
		static inline void										int32_scast( __m512i _mInt32, bool * _pbDst ) {
			xint32x16_to_boolx16( _mInt32, _pbDst );
		}
		static inline void										int32_scast( __m512i _mInt32, std::complex<float> * ) {
			throw std::runtime_error( "int32_scast: No conversion available for int32_t -> std::complex<float>." );
		}
		static inline void										int32_scast( __m512i _mInt32, std::complex<double> * ) {
			throw std::runtime_error( "int32_scast: No conversion available for int32_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										int32_scast( __m256i _mInt32, int8_t * _pi8Dst ) {
			int32x8_to_int8x8_saturated( _mInt32, _pi8Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, uint8_t * _pu8Dst ) {
			int32x8_to_uint8x8_saturated( _mInt32, _pu8Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, int16_t * _pi16Dst ) {
			int32x8_to_int16x8_saturated( _mInt32, _pi16Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, uint16_t * _pu16Dst ) {
			int32x8_to_uint16x8_saturated( _mInt32, _pu16Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, int32_t * _pi32Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi32Dst), _mInt32 );
		}
		static inline void										int32_scast( __m256i _mInt32, uint32_t * _pu32Dst ) {
			int32x8_to_uint32x8_saturated( _mInt32, _pu32Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, int64_t * _pi64Dst ) {
			int32x8_to_int64x8( _mInt32, _pi64Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, uint64_t * _pu64Dst ) {
			int32x8_to_uint64x8_saturated( _mInt32, _pu64Dst );
		}
		static inline void										int32_scast( __m256i _mInt32, nn9::float16 * _pf16Dst ) {
			__m256 m0;
			int32x8_to_float32x8( _mInt32, m0 );
			m0 = _mm256_min_ps( m0, _mm256_set1_ps( 65504.0f ) );
			m0 = _mm256_max_ps( m0, _mm256_set1_ps( -65504.0f ) );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
		}
		static inline void										int32_scast( __m256i _mInt32, bfloat16_t * _pf16Dst ) {
			__m256 m0;
			int32x8_to_float32x8( _mInt32, m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
		}
		static inline void										int32_scast( __m256i _mInt32, float * _pfDst ) {
			__m256 m0;
			int32x8_to_float32x8( _mInt32, m0 );
			_mm256_storeu_ps( _pfDst, m0 );
		}
		static inline void										int32_scast( __m256i _mInt32, double * _pdDst ) {
			int32x8_to_float64x8( _mInt32, _pdDst );
		}
		static inline void										int32_scast( __m256i _mInt32, bool * _pbDst ) {
			xint32x8_to_boolx8( _mInt32, _pbDst );
		}
		static inline void										int32_scast( __m256i _mInt32, std::complex<float> * ) {
			throw std::runtime_error( "int32_scast: No conversion available for int32_t -> std::complex<float>." );
		}
		static inline void										int32_scast( __m256i _mInt32, std::complex<double> * ) {
			throw std::runtime_error( "int32_scast: No conversion available for int32_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__

		
		// ===============================
		// uint32_t
		// ===============================
		static inline void										scast( uint32_t _u32Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::min<uint32_t>( _u32Src, INT8_MAX ));
		}
		static inline void										scast( uint32_t _u32Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::min<uint32_t>( _u32Src, UINT8_MAX ));
		}
		static inline void										scast( uint32_t _u32Src, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::min<uint32_t>( _u32Src, INT16_MAX ));
		}
		static inline void										scast( uint32_t _u32Src, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::min<uint32_t>( _u32Src, UINT16_MAX ));
		}
		static inline void										scast( uint32_t _u32Src, int32_t &_i32Dst ) {
			_i32Dst = std::min<uint32_t>( _u32Src, INT16_MAX );
		}
		static inline void										scast( uint32_t _u32Src, uint32_t &_u32Dst ) {
			_u32Dst = _u32Src;
		}
		static inline void										scast( uint32_t _u32Src, int64_t &_i64Dst ) {
			_i64Dst = _u32Src;
		}
		static inline void										scast( uint32_t _u32Src, uint64_t &_u64Dst ) {
			_u64Dst = _u32Src;
		}
		static inline void										scast( uint32_t _u32Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(std::min<float>( static_cast<float>(_u32Src), 65504.0f ));
		}
		static inline void										scast( uint32_t _u32Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_u32Src);
		}
		static inline void										scast( uint32_t _u32Src, float &_fDst ) {
			_fDst = static_cast<float>(_u32Src);
		}
		static inline void										scast( uint32_t _u32Src, double &_dDst ) {
			_dDst = static_cast<double>(_u32Src);
		}
		static inline void										scast( uint32_t _u32Src, bool &_bDst ) {
			_bDst = _u32Src != 0;
		}
		static inline void										scast( uint32_t _u32Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for uint32_t -> std::complex<float>." );
		}
		static inline void										scast( uint32_t _u32Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for uint32_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										uint32_scast( __m512i _mUint32, int8_t * _pi8Dst ) {
			uint32x16_to_int8x16_saturated( _mUint32, _pi8Dst );
		}
		static inline void										uint32_scast( __m512i _mUint32, uint8_t * _pu8Dst ) {
			uint32x16_to_uint8x16_saturated( _mUint32, _pu8Dst );
		}
		static inline void										uint32_scast( __m512i _mUint32, int16_t * _pi16Dst ) {
			uint32x16_to_int16x16_saturated( _mUint32, _pi16Dst );
		}
		static inline void										uint32_scast( __m512i _mUint32, uint16_t * _pu16Dst ) {
			uint32x16_to_uint16x16_saturated( _mUint32, _pu16Dst );
		}
		static inline void										uint32_scast( __m512i _mUint32, int32_t * _pi32Dst ) {
			uint32x16_to_int32x16_saturated( _mUint32, _pi32Dst );
		}
		static inline void										uint32_scast( __m512i _mUint32, uint32_t * _pu32Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu32Dst), _mUint32 );
		}
		static inline void										uint32_scast( __m512i _mUint32, int64_t * _pi64Dst ) {
			uint32x16_to_xint64x16( _mUint32, reinterpret_cast<uint64_t *>(_pi64Dst) );
		}
		static inline void										uint32_scast( __m512i _mUint32, uint64_t * _pu64Dst ) {
			uint32x16_to_xint64x16( _mUint32, _pu64Dst );
		}
		static inline void										uint32_scast( __m512i _mUint32, nn9::float16 * _pf16Dst ) {
			__m512 m0;
			uint32x16_to_float32x16( _mUint32, m0 );
			__m512 mMax = _mm512_set1_ps( 65504.0f );
			m0 = _mm512_min_ps( m0, mMax );
			nn9::float16::Convert16Float32ToFloat16( _pf16Dst, m0 );
		}
		static inline void										uint32_scast( __m512i _mUint32, bfloat16_t * _pf16Dst ) {
			__m512 m0;
			uint32x16_to_float32x16( _mUint32, m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
		}
		static inline void										uint32_scast( __m512i _mUint32, float * _pfDst ) {
			__m512 m0;
			uint32x16_to_float32x16( _mUint32, m0 );
			_mm512_storeu_ps( _pfDst, m0 );
		}
		static inline void										uint32_scast( __m512i _mUint32, double * _pdDst ) {
			uint32x16_to_float64x16( _mUint32, _pdDst );
		}
		static inline void										uint32_scast( __m512i _mUint32, bool * _pbDst ) {
			xint32x16_to_boolx16( _mUint32, _pbDst );
		}
		static inline void										uint32_scast( __m512i _mUint32, std::complex<float> * ) {
			throw std::runtime_error( "uint32_scast: No conversion available for uint32_t -> std::complex<float>." );
		}
		static inline void										uint32_scast( __m512i _mUint32, std::complex<double> * ) {
			throw std::runtime_error( "uint32_scast: No conversion available for uint32_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										uint32_scast( __m256i _mUint32, int8_t * _pi8Dst ) {
			uint32x8_to_int8x8_saturated( _mUint32, _pi8Dst );
		}
		static inline void										uint32_scast( __m256i _mUint32, uint8_t * _pu8Dst ) {
			uint32x8_to_uint8x8_saturated( _mUint32, _pu8Dst );
		}
		static inline void										uint32_scast( __m256i _mUint32, int16_t * _pi16Dst ) {
			uint32x8_to_int16x8_saturated( _mUint32, _pi16Dst );
		}
		static inline void										uint32_scast( __m256i _mUint32, uint16_t * _pu16Dst ) {
			uint32x8_to_uint16x8_saturated( _mUint32, _pu16Dst );
		}
		static inline void										uint32_scast( __m256i _mUint32, int32_t * _pi32Dst ) {
			uint32x8_to_int32x8_saturated( _mUint32, _pi32Dst );
		}
		static inline void										uint32_scast( __m256i _mUint32, uint32_t * _pu32Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu32Dst), _mUint32 );
		}
		static inline void										uint32_scast( __m256i _mUint32, int64_t * _pi64Dst ) {
			uint32x8_to_xint64x8( _mUint32, reinterpret_cast<uint64_t *>(_pi64Dst) );
		}
		static inline void										uint32_scast( __m256i _mUint32, uint64_t * _pu64Dst ) {
			uint32x8_to_xint64x8( _mUint32, _pu64Dst );
		}
		static inline void										uint32_scast( __m256i _mUint32, nn9::float16 * _pf16Dst ) {
			__m256 m0;
			uint32x8_to_float32x8( _mUint32, m0 );
			__m256 mMax = _mm256_set1_ps( 65504.0f );
			m0 = _mm256_min_ps( m0, mMax );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
		}
		static inline void										uint32_scast( __m256i _mUint32, bfloat16_t * _pf16Dst ) {
			__m256 m0;
			uint32x8_to_float32x8( _mUint32, m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
		}
		static inline void										uint32_scast( __m256i _mUint32, float * _pfDst ) {
			__m256 m0;
			uint32x8_to_float32x8( _mUint32, m0 );
			_mm256_storeu_ps( _pfDst, m0 );
		}
		static inline void										uint32_scast( __m256i _mUint32, double * _pdDst ) {
			uint32x8_to_float64x8( _mUint32, _pdDst );
		}
		static inline void										uint32_scast( __m256i _mUint32, bool * _pbDst ) {
			xint32x8_to_boolx8( _mUint32, _pbDst );
		}
		static inline void										uint32_scast( __m256i _mUint32, std::complex<float> * ) {
			throw std::runtime_error( "uint32_scast: No conversion available for uint32_t -> std::complex<float>." );
		}
		static inline void										uint32_scast( __m256i _mUint32, std::complex<double> * ) {
			throw std::runtime_error( "uint32_scast: No conversion available for uint32_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// int64_t
		// ===============================
		static inline void										scast( int64_t _i64Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<int64_t>( _i64Src, INT8_MIN, INT8_MAX ));
		}
		static inline void										scast( int64_t _i64Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<int64_t>( _i64Src, 0, UINT8_MAX ));
		}
		static inline void										scast( int64_t _i64Src, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::clamp<int64_t>( _i64Src, INT16_MIN, INT16_MAX ));
		}
		static inline void										scast( int64_t _i64Src, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::clamp<int64_t>( _i64Src, 0, UINT16_MAX ));
		}
		static inline void										scast( int64_t _i64Src, int32_t &_i32Dst ) {
			_i32Dst = static_cast<int16_t>(std::clamp<int64_t>( _i64Src, INT32_MIN, INT32_MAX ));
		}
		static inline void										scast( int64_t _i64Src, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint16_t>(std::clamp<int64_t>( _i64Src, 0, UINT32_MAX ));
		}
		static inline void										scast( int64_t _i64Src, int64_t &_i64Dst ) {
			_i64Dst = _i64Src;
		}
		static inline void										scast( int64_t _i64Src, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::max<int64_t>( _i64Src, 0 ));
		}
		static inline void										scast( int64_t _i64Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(std::clamp<float>( static_cast<float>(_i64Src), -65504.0f, 65504.0f ));
		}
		static inline void										scast( int64_t _i64Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_i64Src);
		}
		static inline void										scast( int64_t _i64Src, float &_fDst ) {
			_fDst = static_cast<float>(_i64Src);
		}
		static inline void										scast( int64_t _i64Src, double &_dDst ) {
			_dDst = static_cast<double>(_i64Src);
		}
		static inline void										scast( int64_t _i64Src, bool &_bDst ) {
			_bDst = _i64Src != 0;
		}
		static inline void										scast( int64_t _i64Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for int64_t -> std::complex<float>." );
		}
		static inline void										scast( int64_t _i64Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for int64_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										int64_scast( __m512i _mInt64, int8_t * _pi8Dst ) {
			int64x8_to_int8x8_saturated( _mInt64, _pi8Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, uint8_t * _pu8Dst ) {
			int64x8_to_uint8x8_saturated( _mInt64, _pu8Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, int16_t * _pi16Dst ) {
			int64x8_to_int16x8_saturated( _mInt64, _pi16Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, uint16_t * _pu16Dst ) {
			int64x8_to_uint16x8_saturated( _mInt64, _pu16Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, int32_t * _pi32Dst ) {
			int64x8_to_int32x8_saturated( _mInt64, _pi32Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, uint32_t * _pu32Dst ) {
			int64x8_to_uint32x8_saturated( _mInt64, _pu32Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, int64_t * _pi64Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pi64Dst), _mInt64 );
		}
		static inline void										int64_scast( __m512i _mInt64, uint64_t * _pu64Dst ) {
			int64x8_to_uint64x8_saturated( _mInt64, _pu64Dst );
		}
		static inline void										int64_scast( __m512i _mInt64, nn9::float16 * _pf16Dst ) {
			__m256 m0;
			int64x8_to_float32x8( _mInt64, m0 );
			m0 = _mm256_min_ps( m0, _mm256_set1_ps( 65504.0f ) );
			m0 = _mm256_max_ps( m0, _mm256_set1_ps( -65504.0f ) );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
		}
		static inline void										int64_scast( __m512i _mInt64, bfloat16_t * _pf16Dst ) {
			__m256 m0;
			int64x8_to_float32x8( _mInt64, m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
		}
		static inline void										int64_scast( __m512i _mInt64, float * _pfDst ) {
			__m256 m0;
			int64x8_to_float32x8( _mInt64, m0 );
			_mm256_storeu_ps( _pfDst, m0 );
		}
		static inline void										int64_scast( __m512i _mInt64, double * _pdDst ) {
			int64x8_to_float64x8( _mInt64, _pdDst );
		}
		static inline void										int64_scast( __m512i _mInt64, bool * _pbDst ) {
			xint64x8_to_boolx8( _mInt64, _pbDst );
		}
		static inline void										int64_scast( __m512i _mInt64, std::complex<float> * ) {
			throw std::runtime_error( "int64_scast: No conversion available for int64_t -> std::complex<float>." );
		}
		static inline void										int64_scast( __m512i _mInt64, std::complex<double> * ) {
			throw std::runtime_error( "int64_scast: No conversion available for int64_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										int64_scast( __m256i _mInt64, int8_t * _pi8Dst ) {
			int64x4_to_int8x4_saturated( _mInt64, _pi8Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, uint8_t * _pu8Dst ) {
			int64x4_to_uint8x4_saturated( _mInt64, _pu8Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, int16_t * _pi16Dst ) {
			int64x4_to_int16x4_saturated( _mInt64, _pi16Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, uint16_t * _pu16Dst ) {
			int64x4_to_uint16x4_saturated( _mInt64, _pu16Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, int32_t * _pi32Dst ) {
			int64x4_to_int32x4_saturated( _mInt64, _pi32Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, uint32_t * _pu32Dst ) {
			int64x4_to_uint32x4_saturated( _mInt64, _pu32Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, int64_t * _pi64Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pi64Dst), _mInt64 );
		}
		static inline void										int64_scast( __m256i _mInt64, uint64_t * _pu64Dst ) {
			int64x4_to_uint64x4_saturated( _mInt64, _pu64Dst );
		}
		static inline void										int64_scast( __m256i _mInt64, nn9::float16 * _pf16Dst ) {
			NN9_ALIGN( 32 )
			float fTmp[4];
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( 65504 ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_set1_epi64x( -65504 ) );
			int64x4_to_float32x4( _mInt64, fTmp );
			(*_pf16Dst++) = fTmp[0];
			(*_pf16Dst++) = fTmp[1];
			(*_pf16Dst++) = fTmp[2];
			(*_pf16Dst++) = fTmp[3];
		}
		static inline void										int64_scast( __m256i _mInt64, bfloat16_t * _pf16Dst ) {
			_mInt64 = _mm256_min_epi64( _mInt64, _mm256_set1_epi64x( static_cast<int64_t>(65504.0f) ) );
			_mInt64 = _mm256_max_epi64( _mInt64, _mm256_set1_epi64x( static_cast<int64_t>(-65504.0f) ) );
			NN9_ALIGN( 32 )
			float fTmp[4];
			int64x4_to_float32x4( _mInt64, fTmp );
			(*_pf16Dst++) = fTmp[0];
			(*_pf16Dst++) = fTmp[1];
			(*_pf16Dst++) = fTmp[2];
			(*_pf16Dst++) = fTmp[3];
		}
		static inline void										int64_scast( __m256i _mInt64, float * _pfDst ) {
			int64x4_to_float32x4( _mInt64, _pfDst );
		}
		static inline void										int64_scast( __m256i _mInt64, double * _pdDst ) {
			int64x4_to_float64x4( _mInt64, _pdDst );
		}
		static inline void										int64_scast( __m256i _mInt64, bool * _pbDst ) {
			xint64x4_to_boolx4( _mInt64, _pbDst );
		}
		static inline void										int64_scast( __m256i _mInt64, std::complex<float> * ) {
			throw std::runtime_error( "int64_scast: No conversion available for int64_t -> std::complex<float>." );
		}
		static inline void										int64_scast( __m256i _mInt64, std::complex<double> * ) {
			throw std::runtime_error( "int64_scast: No conversion available for int64_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// uint64_t
		// ===============================
		static inline void										scast( uint64_t _u64Src, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::min<uint64_t>( _u64Src, INT8_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::min<uint64_t>( _u64Src, UINT8_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::min<uint64_t>( _u64Src, INT16_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::min<uint64_t>( _u64Src, UINT16_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, int32_t &_i32Dst ) {
			_i32Dst = static_cast<int32_t>(std::min<uint64_t>( _u64Src, INT32_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::min<uint64_t>( _u64Src, UINT32_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, int64_t &_i64Dst ) {
			_i64Dst = static_cast<int64_t>(std::min<uint64_t>( _u64Src, INT64_MAX ));
		}
		static inline void										scast( uint64_t _u64Src, uint64_t &_u64Dst ) {
			_u64Dst = _u64Src;
		}
		static inline void										scast( uint64_t _u64Src, nn9::float16 &_f16Dst ) {
			_f16Dst = static_cast<float>(std::min<double>( static_cast<double>(_u64Src), 65504.0f ));
		}
		static inline void										scast( uint64_t _u64Src, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_u64Src);
		}
		static inline void										scast( uint64_t _u64Src, float &_fDst ) {
			_fDst = static_cast<float>(_u64Src);
		}
		static inline void										scast( uint64_t _u64Src, double &_dDst ) {
			_dDst = static_cast<double>(_u64Src);
		}
		static inline void										scast( uint64_t _u64Src, bool &_bDst ) {
			_bDst = _u64Src != 0;
		}
		static inline void										scast( uint64_t _u64Src, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for uint64_t -> std::complex<float>." );
		}
		static inline void										scast( uint64_t _u64Src, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for uint64_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										uint64_scast( __m512i _mUint64, int8_t * _pi8Dst ) {
			uint64x8_to_int8x8_saturated( _mUint64, _pi8Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, uint8_t * _pu8Dst ) {
			uint64x8_to_uint8x8_saturated( _mUint64, _pu8Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, int16_t * _pi16Dst ) {
			uint64x8_to_int16x8_saturated( _mUint64, _pi16Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, uint16_t * _pu16Dst ) {
			uint64x8_to_uint16x8_saturated( _mUint64, _pu16Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, int32_t * _pi32Dst ) {
			uint64x8_to_int32x8_saturated( _mUint64, _pi32Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, uint32_t * _pu32Dst ) {
			uint64x8_to_uint32x8_saturated( _mUint64, _pu32Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, int64_t * _pi64Dst ) {
			uint64x8_to_int64x8_saturated( _mUint64, _pi64Dst );
		}
		static inline void										uint64_scast( __m512i _mUint64, uint64_t * _pu64Dst ) {
			_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pu64Dst), _mUint64 );
		}
		static inline void										uint64_scast( __m512i _mUint64, nn9::float16 * _pf16Dst ) {
			__m256 m0;
			uint64x8_to_float32x8( _mUint64, m0 );
			__m256 mMax = _mm256_set1_ps( 65504.0f );
			m0 = _mm256_min_ps( m0, mMax );
			nn9::float16::Convert8Float32ToFloat16( _pf16Dst, m0 );
		}
		static inline void										uint64_scast( __m512i _mUint64, bfloat16_t * _pf16Dst ) {
			__m256 m0;
			uint64x8_to_float32x8( _mUint64, m0 );
			bfloat16_t::storeu_fp32_to_bf16( (_pf16Dst), m0 );
		}
		static inline void										uint64_scast( __m512i _mUint64, float * _pfDst ) {
			__m256 m0;
			uint64x8_to_float32x8( _mUint64, m0 );
			_mm256_storeu_ps( _pfDst, m0 );
		}
		static inline void										uint64_scast( __m512i _mUint64, double * _pdDst ) {
			uint64x8_to_float64x8( _mUint64, _pdDst );
		}
		static inline void										uint64_scast( __m512i _mUint64, bool * _pbDst ) {
			xint64x8_to_boolx8( _mUint64, _pbDst );
		}
		static inline void										uint64_scast( __m512i _mUint64, std::complex<float> * ) {
			throw std::runtime_error( "uint64_scast: No conversion available for uint64_t -> std::complex<float>." );
		}
		static inline void										uint64_scast( __m512i _mUint64, std::complex<double> * ) {
			throw std::runtime_error( "uint64_scast: No conversion available for uint64_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										uint64_scast( __m256i _mUint64, int8_t * _pi8Dst ) {
			uint64x4_to_int8x4_saturated( _mUint64, _pi8Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, uint8_t * _pu8Dst ) {
			uint64x4_to_uint8x4_saturated( _mUint64, _pu8Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, int16_t * _pi16Dst ) {
			uint64x4_to_int16x4_saturated( _mUint64, _pi16Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, uint16_t * _pu16Dst ) {
			uint64x4_to_uint16x4_saturated( _mUint64, _pu16Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, int32_t * _pi32Dst ) {
			uint64x4_to_int32x4_saturated( _mUint64, _pi32Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, uint32_t * _pu32Dst ) {
			uint64x4_to_uint32x4_saturated( _mUint64, _pu32Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, int64_t * _pi64Dst ) {
			uint64x4_to_int64x4_saturated( _mUint64, _pi64Dst );
		}
		static inline void										uint64_scast( __m256i _mUint64, uint64_t * _pu64Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pu64Dst), _mUint64 );
		}
		static inline void										uint64_scast( __m256i _mUint64, nn9::float16 * _pf16Dst ) {
			_mUint64 = _mm256_min_epu64( _mUint64, _mm256_set1_epi64x( static_cast<uint64_t>(65504.0f) ) );
			NN9_ALIGN( 32 )
			float fTmp[4];
			uint64x4_to_float32x4( _mUint64, fTmp );

			(*_pf16Dst++) = fTmp[0];
			(*_pf16Dst++) = fTmp[1];
			(*_pf16Dst++) = fTmp[2];
			(*_pf16Dst++) = fTmp[3];
		}
		static inline void										uint64_scast( __m256i _mUint64, bfloat16_t * _pf16Dst ) {
			NN9_ALIGN( 32 )
			float fTmp[4];
			uint64x4_to_float32x4( _mUint64, fTmp );

			(*_pf16Dst++) = fTmp[0];
			(*_pf16Dst++) = fTmp[1];
			(*_pf16Dst++) = fTmp[2];
			(*_pf16Dst++) = fTmp[3];
		}
		static inline void										uint64_scast( __m256i _mUint64, float * _pfDst ) {
			uint64x4_to_float32x4( _mUint64, _pfDst );
		}
		static inline void										uint64_scast( __m256i _mUint64, double * _pdDst ) {
			uint64x4_to_float64x4( _mUint64, _pdDst );
		}
		static inline void										uint64_scast( __m256i _mUint64, bool * _pbDst ) {
			xint64x4_to_boolx4( _mUint64, _pbDst );
		}
		static inline void										uint64_scast( __m256i _mUint64, std::complex<float> * ) {
			throw std::runtime_error( "uint64_scast: No conversion available for uint64_t -> std::complex<float>." );
		}
		static inline void										uint64_scast( __m256i _mUint64, std::complex<double> * ) {
			throw std::runtime_error( "uint64_scast: No conversion available for uint64_t -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// float/nn9::float16/bfloat16_t
		// ===============================
		static inline void										scast( float _fSrc, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<float>( _fSrc, static_cast<float>(INT8_MIN), static_cast<float>(INT8_MAX) ));
		}
		static inline void										scast( float _fSrc, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT8_MAX) ));
		}
		static inline void										scast( float _fSrc, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::clamp<float>( _fSrc, static_cast<float>(INT16_MIN), static_cast<float>(INT16_MAX) ));
		}
		static inline void										scast( float _fSrc, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT16_MAX) ));
		}
		static inline void										scast( float _fSrc, int32_t &_i32Dst ) {
			_i32Dst = static_cast<int32_t>(std::clamp<float>( _fSrc, -2147483648.0f, 2147483520.0f ));
		}
		static inline void										scast( float _fSrc, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT32_MAX) ));
		}
		static inline void										scast( float _fSrc, int64_t &_i64Dst ) {
			_i64Dst = static_cast<int64_t>(std::clamp<float>( _fSrc, -9223372036854775808.0f, 9223371487098961920.0f ));
		}
		static inline void										scast( float _fSrc, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT64_MAX) ));
		}
		static inline void										scast( float _fSrc, nn9::float16 &_f16Dst ) {
			_f16Dst = std::clamp<float>( _fSrc, -65504.0f, 65504.0f );
		}
		static inline void										scast( float _fSrc, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_fSrc);
		}
		static inline void										scast( float _fSrc, float &_fDst ) {
			_fDst = _fSrc;
		}
		static inline void										scast( float _fSrc, double &_dDst ) {
			_dDst = static_cast<double>(_fSrc);
		}
		static inline void										scast( float _fSrc, bool &_bDst ) {
			_bDst = _fSrc != 0.0f;
		}
		static inline void										scast( float _fSrc, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for float -> std::complex<float>." );
		}
		static inline void										scast( float _fSrc, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for float -> std::complex<double>." );
		}

		static inline void										scast( nn9::float16 _fSrc, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<float>( _fSrc, static_cast<float>(INT8_MIN), static_cast<float>(INT8_MAX) ));
		}
		static inline void										scast( nn9::float16 _fSrc, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT8_MAX) ));
		}
		static inline void										scast( nn9::float16 _fSrc, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::clamp<float>( _fSrc, static_cast<float>(INT16_MIN), static_cast<float>(INT16_MAX) ));
		}
		static inline void										scast( nn9::float16 _fSrc, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::max<float>( _fSrc, 0.0f ));
		}
		static inline void										scast( nn9::float16 _fSrc, int32_t &_i32Dst ) {
			_i32Dst = static_cast<int32_t>(static_cast<float>(_fSrc));
		}
		static inline void										scast( nn9::float16 _fSrc, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::max<float>( _fSrc, 0.0f ));
		}
		static inline void										scast( nn9::float16 _fSrc, int64_t &_i64Dst ) {
			_i64Dst = static_cast<int64_t>(static_cast<float>(_fSrc));
		}
		static inline void										scast( nn9::float16 _fSrc, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::max<float>( _fSrc, 0.0f ));
		}
		static inline void										scast( nn9::float16 _fSrc, nn9::float16 &_f16Dst ) {
			_f16Dst.m_u16Value = _fSrc.m_u16Value;
		}
		static inline void										scast( nn9::float16 _fSrc, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_fSrc);
		}
		static inline void										scast( nn9::float16 _fSrc, float &_fDst ) {
			_fDst = _fSrc;
		}
		static inline void										scast( nn9::float16 _fSrc, double &_dDst ) {
			_dDst = static_cast<double>(_fSrc);
		}
		static inline void										scast( nn9::float16 _fSrc, bool &_bDst ) {
			_bDst = static_cast<float>(_fSrc) != 0.0f;
		}
		static inline void										scast( nn9::float16 _fSrc, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for nn9::float16 -> std::complex<float>." );
		}
		static inline void										scast( nn9::float16 _fSrc, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for nn9::float16 -> std::complex<double>." );
		}

		static inline void										scast( bfloat16_t _fSrc, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<float>( _fSrc, static_cast<float>(INT8_MIN), static_cast<float>(INT8_MAX) ));
		}
		static inline void										scast( bfloat16_t _fSrc, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT8_MAX) ));
		}
		static inline void										scast( bfloat16_t _fSrc, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::clamp<float>( _fSrc, static_cast<float>(INT16_MIN), static_cast<float>(INT16_MAX) ));
		}
		static inline void										scast( bfloat16_t _fSrc, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT16_MAX) ));
		}
		static inline void										scast( bfloat16_t _fSrc, int32_t &_i32Dst ) {
			_i32Dst = static_cast<int32_t>(std::clamp<float>( _fSrc, -2147483648.0f, 2147483520.0f ));
		}
		static inline void										scast( bfloat16_t _fSrc, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT32_MAX) ));
		}
		static inline void										scast( bfloat16_t _fSrc, int64_t &_i64Dst ) {
			_i64Dst = static_cast<int64_t>(std::clamp<float>( _fSrc, -9223372036854775808.0f, 9223371487098961920.0f ));
		}
		static inline void										scast( bfloat16_t _fSrc, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::clamp<float>( _fSrc, 0.0f, static_cast<float>(UINT64_MAX) ));
		}
		static inline void										scast( bfloat16_t _fSrc, nn9::float16 &_f16Dst ) {
			_f16Dst.m_u16Value = _fSrc.m_u16Value;
		}
		static inline void										scast( bfloat16_t _fSrc, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_fSrc);
		}
		static inline void										scast( bfloat16_t _fSrc, float &_fDst ) {
			_fDst = _fSrc;
		}
		static inline void										scast( bfloat16_t _fSrc, double &_dDst ) {
			_dDst = static_cast<double>(_fSrc);
		}
		static inline void										scast( bfloat16_t _fSrc, bool &_bDst ) {
			_bDst = static_cast<float>(_fSrc) != 0.0f;
		}
		static inline void										scast( bfloat16_t _fSrc, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for bfloat16_t -> std::complex<float>." );
		}
		static inline void										scast( bfloat16_t _fSrc, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for bfloat16_t -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										float_scast( __m512 _mFloat, int8_t * _pi8Dst ) {
			float32x16_to_int8x16( _mFloat, _pi8Dst );
		}
		static inline void										float_scast( __m512 _mFloat, uint8_t * _pu8Dst ) {
			float32x16_to_uint8x16( _mFloat, _pu8Dst );
		}
		static inline void										float_scast( __m512 _mFloat, int16_t * _pi16Dst ) {
			float32x16_to_int16x16( _mFloat, _pi16Dst );
		}
		static inline void										float_scast( __m512 _mFloat, uint16_t * _pu16Dst ) {
			float32x16_to_uint16x16( _mFloat, _pu16Dst );
		}
		static inline void										float_scast( __m512 _mFloat, int32_t * _pi32Dst ) {
			float32x16_to_int32x16( _mFloat, _pi32Dst );
		}
		static inline void										float_scast( __m512 _mFloat, uint32_t * _pu32Dst ) {
			float32x16_to_uint32x16( _mFloat, _pu32Dst );
		}
		static inline void										float_scast( __m512 _mFloat, int64_t * _pi64Dst ) {
			float32x16_to_int64x16( _mFloat, _pi64Dst );
		}
		static inline void										float_scast( __m512 _mFloat, uint64_t * _pu64Dst ) {
			float32x16_to_uint64x16( _mFloat, _pu64Dst );
		}
		static inline void										float_scast( __m512 _mFloat, nn9::float16 * _pf32Dst ) {
			nn9::float16::Convert16Float32ToFloat16( _pf32Dst, _mFloat );
		}
		static inline void										float_scast( __m512 _mFloat, bfloat16_t * _pf32Dst ) {
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf32Dst), _mFloat );
		}
		static inline void										float_scast( __m512 _mFloat, float * _pfDst ) {
			_mm512_storeu_ps( _pfDst, _mFloat );
		}
		static inline void										float_scast( __m512 _mFloat, double * _pdDst ) {
			float32x16_to_float64x16( _mFloat, _pdDst );
		}
		static inline void										float_scast( __m512 _mFloat, bool * _pbDst ) {
			float32x16_to_boolx16( _mFloat, _pbDst );
		}
		static inline void										float_scast( __m512 _mUint64, std::complex<float> * ) {
			throw std::runtime_error( "float_scast: No conversion available for float -> std::complex<float>." );
		}
		static inline void										float_scast( __m512 _mUint64, std::complex<double> * ) {
			throw std::runtime_error( "float_scast: No conversion available for float -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										float_scast( __m256 _mFloat, int8_t * _pi8Dst ) {
			float32x8_to_int8x8( _mFloat, _pi8Dst );
		}
		static inline void										float_scast( __m256 _mFloat, uint8_t * _pu8Dst ) {
			float32x8_to_uint8x8( _mFloat, _pu8Dst );
		}
		static inline void										float_scast( __m256 _mFloat, int16_t * _pi16Dst ) {
			float32x8_to_int16x8( _mFloat, _pi16Dst );
		}
		static inline void										float_scast( __m256 _mFloat, uint16_t * _pu16Dst ) {
			float32x8_to_uint16x8( _mFloat, _pu16Dst );
		}
		static inline void										float_scast( __m256 _mFloat, int32_t * _pi32Dst ) {
			float32x8_to_int32x8( _mFloat, _pi32Dst );
		}
		static inline void										float_scast( __m256 _mFloat, uint32_t * _pu32Dst ) {
			float32x8_to_uint32x8( _mFloat, _pu32Dst );
		}
		static inline void										float_scast( __m256 _mFloat, int64_t * _pi64Dst ) {
			float32x8_to_int64x8( _mFloat, _pi64Dst );
		}
		static inline void										float_scast( __m256 _mFloat, uint64_t * _pu64Dst ) {
			float32x8_to_uint64x8( _mFloat, _pu64Dst );
		}
		static inline void										float_scast( __m256 _mFloat, nn9::float16 * _pf32Dst ) {
			nn9::float16::Convert8Float32ToFloat16( _pf32Dst, _mFloat );
		}
		static inline void										float_scast( __m256 _mFloat, bfloat16_t * _pf32Dst ) {
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf32Dst), _mFloat );
		}
		static inline void										float_scast( __m256 _mFloat, float * _pfDst ) {
			_mm256_storeu_ps( _pfDst, _mFloat );
		}
		static inline void										float_scast( __m256 _mFloat, double * _pdDst ) {
			float32x8_to_float64x8( _mFloat, _pdDst );
		}
		static inline void										float_scast( __m256 _mFloat, bool * _pbDst ) {
			float32x8_to_boolx8( _mFloat, _pbDst );
		}
		static inline void										float_scast( __m256 _mUint64, std::complex<float> * ) {
			throw std::runtime_error( "float_scast: No conversion available for float -> std::complex<float>." );
		}
		static inline void										float_scast( __m256 _mUint64, std::complex<double> * ) {
			throw std::runtime_error( "float_scast: No conversion available for float -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// double
		// ===============================
		static inline void										scast( double _dSrc, int8_t &_i8Dst ) {
			_i8Dst = static_cast<int8_t>(std::clamp<double>( _dSrc, static_cast<double>(INT8_MIN), static_cast<double>(INT8_MAX) ));
		}
		static inline void										scast( double _dSrc, uint8_t &_u8Dst ) {
			_u8Dst = static_cast<uint8_t>(std::clamp<double>( _dSrc, 0.0, static_cast<float>(UINT8_MAX) ));
		}
		static inline void										scast( double _dSrc, int16_t &_i16Dst ) {
			_i16Dst = static_cast<int16_t>(std::clamp<double>( _dSrc, static_cast<double>(INT16_MIN), static_cast<double>(INT16_MAX) ));
		}
		static inline void										scast( double _dSrc, uint16_t &_u16Dst ) {
			_u16Dst = static_cast<uint16_t>(std::clamp<double>( _dSrc, 0.0, static_cast<double>(UINT16_MAX) ));
		}
		static inline void										scast( double _dSrc, int32_t &_i32Dst ) {
			_i32Dst = static_cast<int32_t>(std::clamp<double>( _dSrc, static_cast<double>(INT32_MIN), static_cast<double>(INT32_MAX) ));
		}
		static inline void										scast( double _dSrc, uint32_t &_u32Dst ) {
			_u32Dst = static_cast<uint32_t>(std::clamp<double>( _dSrc, 0.0, static_cast<double>(UINT32_MAX) ));
		}
		static inline void										scast( double _dSrc, int64_t &_i64Dst ) {
			_i64Dst = static_cast<int64_t>(std::clamp<double>( _dSrc, static_cast<double>(INT64_MIN), 9223372036854774784.0 ));
		}
		static inline void										scast( double _dSrc, uint64_t &_u64Dst ) {
			_u64Dst = static_cast<uint64_t>(std::clamp<double>( _dSrc, 0.0, 18446744073709549568.0 ));
		}
		static inline void										scast( double _dSrc, nn9::float16 &_f16Dst ) {
			_f16Dst = std::clamp<double>( _dSrc, -65504.0, 65504.0 );
		}
		static inline void										scast( double _dSrc, bfloat16_t &_f16Dst ) {
			_f16Dst = static_cast<float>(_dSrc);
		}
		static inline void										scast( double _dSrc, float &_fDst ) {
			_fDst = static_cast<float>(_dSrc);
		}
		static inline void										scast( double _dSrc, double &_dDst ) {
			_dDst = _dSrc;
		}
		static inline void										scast( double _dSrc, bool &_bDst ) {
			_bDst = _dSrc != 0.0;
		}
		static inline void										scast( double _dSrc, std::complex<float> & ) {
			throw std::runtime_error( "scast: No conversion available for double -> std::complex<float>." );
		}
		static inline void										scast( double _dSrc, std::complex<double> & ) {
			throw std::runtime_error( "scast: No conversion available for double -> std::complex<double>." );
		}

#ifdef __AVX512F__
		static inline void										double_scast( __m512d _mDouble, int8_t * _pi8Dst ) {
			float64x8_to_int8x8( _mDouble, _pi8Dst );
		}
		static inline void										double_scast( __m512d _mDouble, uint8_t * _pu8Dst ) {
			float64x8_to_uint8x8( _mDouble, _pu8Dst );
		}
		static inline void										double_scast( __m512d _mDouble, int16_t * _pi16Dst ) {
			float64x8_to_int16x8( _mDouble, _pi16Dst );
		}
		static inline void										double_scast( __m512d _mDouble, uint16_t * _pu16Dst ) {
			float64x8_to_uint16x8( _mDouble, _pu16Dst );
		}
		static inline void										double_scast( __m512d _mDouble, int32_t * _pi32Dst ) {
			float64x8_to_int32x8( _mDouble, _pi32Dst );
		}
		static inline void										double_scast( __m512d _mDouble, uint32_t * _pu32Dst ) {
			float64x8_to_uint32x8( _mDouble, _pu32Dst );
		}
		static inline void										double_scast( __m512d _mDouble, int64_t * _pi64Dst ) {
			float64x8_to_int64x8( _mDouble, _pi64Dst );
		}
		static inline void										double_scast( __m512d _mDouble, uint64_t * _pu64Dst ) {
			float64x8_to_uint64x8( _mDouble, _pu64Dst );
		}
		static inline void										double_scast( __m512d _mDouble, nn9::float16 * _pf32Dst ) {
			__m256 mTmp;
			float64x8_to_float32x8( _mDouble, mTmp );
			nn9::float16::Convert8Float32ToFloat16( _pf32Dst, mTmp );
		}
		static inline void										double_scast( __m512d _mDouble, bfloat16_t * _pf32Dst ) {
			__m256 mTmp;
			float64x8_to_float32x8( _mDouble, mTmp );
			bfloat16_t::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pf32Dst), mTmp );
		}
		static inline void										double_scast( __m512d _mDouble, float * _pfDst ) {
			float64x8_to_float32x8( _mDouble, _pfDst );
		}
		static inline void										double_scast( __m512d _mDouble, double * _pdDst ) {
			_mm512_storeu_pd( _pdDst, _mDouble );
		}
		static inline void										double_scast( __m512d _mDouble, bool * _pbDst ) {
			float64x8_to_boolx8( _mDouble, _pbDst );
		}
		static inline void										double_scast( __m512 _mDouble, std::complex<float> * ) {
			throw std::runtime_error( "double_scast: No conversion available for float -> std::complex<float>." );
		}
		static inline void										double_scast( __m512 _mDouble, std::complex<double> * ) {
			throw std::runtime_error( "double_scast: No conversion available for float -> std::complex<double>." );
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		static inline void										double_scast( __m256d _mDouble, int8_t * _pi8Dst ) {
			float64x4_to_int8x4( _mDouble, _pi8Dst );
		}
		static inline void										double_scast( __m256d _mDouble, uint8_t * _pu8Dst ) {
			float64x4_to_uint8x4( _mDouble, _pu8Dst );
		}
		static inline void										double_scast( __m256d _mDouble, int16_t * _pi16Dst ) {
			float64x4_to_int16x4( _mDouble, _pi16Dst );
		}
		static inline void										double_scast( __m256d _mDouble, uint16_t * _pu16Dst ) {
			float64x4_to_uint16x4( _mDouble, _pu16Dst );
		}
		static inline void										double_scast( __m256d _mDouble, int32_t * _pi32Dst ) {
			float64x4_to_int32x4( _mDouble, _pi32Dst );
		}
		static inline void										double_scast( __m256d _mDouble, uint32_t * _pu32Dst ) {
			float64x4_to_uint32x4( _mDouble, _pu32Dst );
		}
		static inline void										double_scast( __m256d _mDouble, int64_t * _pi64Dst ) {
			float64x4_to_int64x4( _mDouble, _pi64Dst );
		}
		static inline void										double_scast( __m256d _mDouble, uint64_t * _pu64Dst ) {
			float64x4_to_uint64x4( _mDouble, _pu64Dst );
		}
		static inline void										double_scast( __m256d _mDouble, nn9::float16 * _pf32Dst ) {
			NN9_ALIGN( 32 )
			float fTmp[4];
			float64x4_to_float32x4( _mDouble, fTmp );
			(*_pf32Dst++) = fTmp[0];
			(*_pf32Dst++) = fTmp[1];
			(*_pf32Dst++) = fTmp[2];
			(*_pf32Dst++) = fTmp[3];
		}
		static inline void										double_scast( __m256d _mDouble, bfloat16_t * _pf32Dst ) {
			__m128 mTmp;
			float64x4_to_float32x4( _mDouble, mTmp );
			bfloat16_t::storeu_fp32_to_bf16( _pf32Dst, mTmp );
		}
		static inline void										double_scast( __m256d _mDouble, float * _pfDst ) {
			float64x4_to_float32x4( _mDouble, _pfDst );
		}
		static inline void										double_scast( __m256d _mDouble, double * _pdDst ) {
			_mm256_storeu_pd( _pdDst, _mDouble );
		}
		static inline void										double_scast( __m256d _mDouble, bool * _pbDst ) {
			float64x4_to_boolx4( _mDouble, _pbDst );
		}
		static inline void										double_scast( __m256 _mDouble, std::complex<float> * ) {
			throw std::runtime_error( "double_scast: No conversion available for double -> std::complex<float>." );
		}
		static inline void										double_scast( __m256 _mDouble, std::complex<double> * ) {
			throw std::runtime_error( "double_scast: No conversion available for double -> std::complex<double>." );
		}
#endif	// #ifdef __AVX2__


#ifdef __AVX512F__
		template <typename _tInType, typename _tOutType>
		static inline void										scast( __m512i _mSrc, _tOutType * _ptDst ) {
			if constexpr ( std::is_same<_tInType, int8_t>::value ) { int8_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint8_t>::value || std::is_same<_tInType, bool>::value ) { uint8_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, int16_t>::value ) { int16_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint16_t>::value ) { uint16_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, int32_t>::value ) { int32_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint32_t>::value ) { uint32_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, int64_t>::value ) { int64_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint64_t>::value ) { uint64_scast( _mSrc, _ptDst ); }

			else { throw std::runtime_error( "scast<__m512i,>: Invalid input type." ); }
		}

		template <typename _tInType, typename _tOutType>
		static inline void										scast( __m512 _mSrc, _tOutType * _ptDst ) {
			if constexpr ( std::is_same<_tInType, float>::value || std::is_same<_tInType, nn9::float16>::value || std::is_same<_tInType, bfloat16_t>::value ) { float_scast( _mSrc, _ptDst ); }

			else { throw std::runtime_error( "scast<__m512,>: Invalid input type." ); }
		}

		template <typename _tInType, typename _tOutType>
		static inline void										scast( __m512d _mSrc, _tOutType * _ptDst ) {
			if constexpr ( std::is_same<_tInType, double>::value ) { double_scast( _mSrc, _ptDst ); }

			else { throw std::runtime_error( "scast<__m512d,>: Invalid input type." ); }
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		template <typename _tInType, typename _tOutType>
		static inline void										scast( __m256i _mSrc, _tOutType * _ptDst ) {
			if constexpr ( std::is_same<_tInType, int8_t>::value ) { int8_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint8_t>::value || std::is_same<_tInType, bool>::value ) { uint8_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, int16_t>::value ) { int16_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint16_t>::value ) { uint16_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, int32_t>::value ) { int32_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint32_t>::value ) { uint32_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, int64_t>::value ) { int64_scast( _mSrc, _ptDst ); }
			else if constexpr ( std::is_same<_tInType, uint64_t>::value ) { uint64_scast( _mSrc, _ptDst ); }

			else { throw std::runtime_error( "scast<__m256i,>: Invalid input type." ); }
		}

		template <typename _tInType, typename _tOutType>
		static inline void										scast( __m256 _mSrc, _tOutType * _ptDst ) {
			if constexpr ( std::is_same<_tInType, float>::value || std::is_same<_tInType, nn9::float16>::value || std::is_same<_tInType, bfloat16_t>::value ) { float_scast( _mSrc, _ptDst ); }

			else { throw std::runtime_error( "scast<__m256,>: Invalid input type." ); }
		}

		template <typename _tInType, typename _tOutType>
		static inline void										scast( __m256d _mSrc, _tOutType * _ptDst ) {
			if constexpr ( std::is_same<_tInType, double>::value ) { double_scast( _mSrc, _ptDst ); }

			else { throw std::runtime_error( "scast<__m256d,>: Invalid input type." ); }
		}
#endif	// #ifdef __AVX2__



#ifdef __AVX512F__
		/**
		 * Multiplies each of the 64 int8 elements in a __m512i by itself.
		 * 
		 * @param _pm512Val the input vector.
		 * @return a __m512i containing the products.
		 */
		static inline __m512i									SquareInt8( __m512i _pm512Val ) {
			__m256i m256Low = _mm512_extracti64x4_epi64( _pm512Val, 0 );
			__m256i m256High = _mm512_extracti64x4_epi64( _pm512Val, 1 );
			__m512i m512Low16 = _mm512_cvtepi8_epi16( m256Low );
			__m512i m512High16 = _mm512_cvtepi8_epi16( m256High );
			__m512i m512MulLow = _mm512_mullo_epi16( m512Low16, m512Low16 );
			__m512i m512MulHigh = _mm512_mullo_epi16( m512High16, m512High16 );
			__m256i m256MulLow0 = _mm512_extracti64x4_epi64( m512MulLow, 0 );
			__m256i m256MulLow1 = _mm512_extracti64x4_epi64( m512MulLow, 1 );
			__m256i m256MulHigh0 = _mm512_extracti64x4_epi64( m512MulHigh, 0 );
			__m256i m256MulHigh1 = _mm512_extracti64x4_epi64( m512MulHigh, 1 );
			__m256i m256LowPacked = _mm256_packs_epi16( m256MulLow0, m256MulLow1 );
			__m256i m256HighPacked = _mm256_packs_epi16( m256MulHigh0, m256MulHigh1 );
			__m512i m512Result = _mm512_inserti64x4( _mm512_castsi256_si512( m256LowPacked ), m256HighPacked, 1 );
			return m512Result;
		}

		/**
		 * Multiplies each of the 64 uint8 elements in a __m512i by itself.
		 * 
		 * @param _pm512Val the input vector.
		 * @return a __m512i containing the products as uint8 (with saturation).
		 */
		static inline __m512i									SquareUint8( __m512i _pm512Val ) {
			__m256i m256Low = _mm512_extracti64x4_epi64( _pm512Val, 0 );
			__m256i m256High = _mm512_extracti64x4_epi64( _pm512Val, 1 );
			__m512i m512Low16 = _mm512_cvtepu8_epi16( m256Low );
			__m512i m512High16 = _mm512_cvtepu8_epi16( m256High );
			__m512i m512MulLow = _mm512_mullo_epi16( m512Low16, m512Low16 );
			__m512i m512MulHigh = _mm512_mullo_epi16( m512High16, m512High16 );
			__m256i m256MulLow0 = _mm512_extracti64x4_epi64( m512MulLow, 0 );
			__m256i m256MulLow1 = _mm512_extracti64x4_epi64( m512MulLow, 1 );
			__m256i m256MulHigh0 = _mm512_extracti64x4_epi64( m512MulHigh, 0 );
			__m256i m256MulHigh1 = _mm512_extracti64x4_epi64( m512MulHigh, 1 );
			__m256i m256LowPacked = _mm256_packus_epi16( m256MulLow0, m256MulLow1 );
			__m256i m256HighPacked = _mm256_packus_epi16( m256MulHigh0, m256MulHigh1 );
			__m512i m512Result = _mm512_inserti64x4( _mm512_castsi256_si512( m256LowPacked ), m256HighPacked, 1 );
			return m512Result;
		}

		/**
		 * Multiplies each of the 32 int16 elements in a __m512i by itself with saturation.
		 * @param _pm512Val the input vector (int16)
		 * @return a __m512i containing the products saturated to int16
		 */
		static inline __m512i									SquareInt16( __m512i _pm512Val ) {
			__m256i m256Low = _mm512_extracti64x4_epi64(_pm512Val, 0);
			__m256i m256High = _mm512_extracti64x4_epi64(_pm512Val, 1);

			__m128i m128Low0 = _mm256_castsi256_si128(m256Low);
			__m128i m128Low1 = _mm256_extracti128_si256(m256Low, 1);
			__m128i m128High0 = _mm256_castsi256_si128(m256High);
			__m128i m128High1 = _mm256_extracti128_si256(m256High, 1);

			__m256i m256ValLo0 = _mm256_cvtepi16_epi32(m128Low0);
			__m256i m256ValLo1 = _mm256_cvtepi16_epi32(m128Low1);
			__m256i m256ValHi0 = _mm256_cvtepi16_epi32(m128High0);
			__m256i m256ValHi1 = _mm256_cvtepi16_epi32(m128High1);

			__m256i m256MulLo0 = _mm256_mullo_epi32(m256ValLo0, m256ValLo0);
			__m256i m256MulLo1 = _mm256_mullo_epi32(m256ValLo1, m256ValLo1);
			__m256i m256MulHi0 = _mm256_mullo_epi32(m256ValHi0, m256ValHi0);
			__m256i m256MulHi1 = _mm256_mullo_epi32(m256ValHi1, m256ValHi1);

			__m128i m128Lo0_0 = _mm256_castsi256_si128(m256MulLo0);
			__m128i m128Lo0_1 = _mm256_extracti128_si256(m256MulLo0, 1);
			__m128i m128Lo1_0 = _mm256_castsi256_si128(m256MulLo1);
			__m128i m128Lo1_1 = _mm256_extracti128_si256(m256MulLo1, 1);
			__m128i m128Hi0_0 = _mm256_castsi256_si128(m256MulHi0);
			__m128i m128Hi0_1 = _mm256_extracti128_si256(m256MulHi0, 1);
			__m128i m128Hi1_0 = _mm256_castsi256_si128(m256MulHi1);
			__m128i m128Hi1_1 = _mm256_extracti128_si256(m256MulHi1, 1);

			__m128i m128ResLoLow = _mm_packs_epi32(m128Lo0_0, m128Lo0_1);
			__m128i m128ResLoHigh = _mm_packs_epi32(m128Lo1_0, m128Lo1_1);
			__m128i m128ResHiLow = _mm_packs_epi32(m128Hi0_0, m128Hi0_1);
			__m128i m128ResHiHigh = _mm_packs_epi32(m128Hi1_0, m128Hi1_1);

			__m256i m256PackedLow = _mm256_setr_m128i(m128ResLoLow, m128ResLoHigh);
			__m256i m256PackedHigh = _mm256_setr_m128i(m128ResHiLow, m128ResHiHigh);
			return _mm512_inserti64x4(_mm512_castsi256_si512(m256PackedLow), m256PackedHigh, 1);
		}

		/**
		 * Multiplies each of the 32 uint16 elements in a __m512i by itself with saturation.
		 * @param _pm512Val the input vector (uint16)
		 * @return a __m512i containing the products saturated to uint16
		 */
		static inline __m512i									SquareUint16( __m512i _pm512Val ) {
			__m256i m256Low = _mm512_extracti64x4_epi64(_pm512Val, 0);
			__m256i m256High = _mm512_extracti64x4_epi64(_pm512Val, 1);

			__m128i m128Low0 = _mm256_castsi256_si128(m256Low);
			__m128i m128Low1 = _mm256_extracti128_si256(m256Low, 1);
			__m128i m128High0 = _mm256_castsi256_si128(m256High);
			__m128i m128High1 = _mm256_extracti128_si256(m256High, 1);

			__m256i m256ValLo0 = _mm256_cvtepu16_epi32(m128Low0);
			__m256i m256ValLo1 = _mm256_cvtepu16_epi32(m128Low1);
			__m256i m256ValHi0 = _mm256_cvtepu16_epi32(m128High0);
			__m256i m256ValHi1 = _mm256_cvtepu16_epi32(m128High1);

			__m256i m256MulLo0 = _mm256_mullo_epi32(m256ValLo0, m256ValLo0);
			__m256i m256MulLo1 = _mm256_mullo_epi32(m256ValLo1, m256ValLo1);
			__m256i m256MulHi0 = _mm256_mullo_epi32(m256ValHi0, m256ValHi0);
			__m256i m256MulHi1 = _mm256_mullo_epi32(m256ValHi1, m256ValHi1);

			__m128i m128Lo0_0 = _mm256_castsi256_si128(m256MulLo0);
			__m128i m128Lo0_1 = _mm256_extracti128_si256(m256MulLo0, 1);
			__m128i m128Lo1_0 = _mm256_castsi256_si128(m256MulLo1);
			__m128i m128Lo1_1 = _mm256_extracti128_si256(m256MulLo1, 1);
			__m128i m128Hi0_0 = _mm256_castsi256_si128(m256MulHi0);
			__m128i m128Hi0_1 = _mm256_extracti128_si256(m256MulHi0, 1);
			__m128i m128Hi1_0 = _mm256_castsi256_si128(m256MulHi1);
			__m128i m128Hi1_1 = _mm256_extracti128_si256(m256MulHi1, 1);

			__m128i m128ResLoLow = _mm_packus_epi32(m128Lo0_0, m128Lo0_1);
			__m128i m128ResLoHigh = _mm_packus_epi32(m128Lo1_0, m128Lo1_1);
			__m128i m128ResHighLow = _mm_packus_epi32(m128Hi0_0, m128Hi0_1);
			__m128i m128ResHighHigh = _mm_packus_epi32(m128Hi1_0, m128Hi1_1);

			__m256i m256PackedLow = _mm256_setr_m128i(m128ResLoLow, m128ResLoHigh);
			__m256i m256PackedHigh = _mm256_setr_m128i(m128ResHighLow, m128ResHighHigh);
			return _mm512_inserti64x4(_mm512_castsi256_si512(m256PackedLow), m256PackedHigh, 1);
		}

		/**
		 * Squares each int32 in a __m512i with saturation.
		 * Any input whose absolute value is greater than 46340 will overflow, so clamp to INT32_MAX.
		 * 
		 * @param _pm512Val input vector (__m512i of int32).
		 * @return __m512i with saturated squares.
		 */
		static inline __m512i									SquareInt32( __m512i _pm512Val ) {
			__m256i m256Low = _mm512_extracti64x4_epi64(_pm512Val, 0);
			__m256i m256High = _mm512_extracti64x4_epi64(_pm512Val, 1);

			__m256i m256AbsLow = _mm256_abs_epi32(m256Low);
			__m256i m256AbsHigh = _mm256_abs_epi32(m256High);
			__m256i m256Threshold = _mm256_set1_epi32(46340);
			__m256i m256OverflowMaskLow = _mm256_cmpgt_epi32(m256AbsLow, m256Threshold);
			__m256i m256OverflowMaskHigh = _mm256_cmpgt_epi32(m256AbsHigh, m256Threshold);
			__m256i m256MulLow = _mm256_mullo_epi32(m256Low, m256Low);
			__m256i m256MulHigh = _mm256_mullo_epi32(m256High, m256High);
			__m256i m256Max = _mm256_set1_epi32(0x7FFFFFFF);
			m256MulLow = _mm256_blendv_epi8(m256MulLow, m256Max, m256OverflowMaskLow);
			m256MulHigh = _mm256_blendv_epi8(m256MulHigh, m256Max, m256OverflowMaskHigh);

			return _mm512_inserti64x4(_mm512_castsi256_si512(m256MulLow), m256MulHigh, 1);
		}

		/**
		 * Squares each uint32 in a __m512i with saturation.
		 * Any input greater than 65535 will overflow, so clamp to UINT32_MAX.
		 * 
		 * @param _pm512Val input vector (__m512i of uint32).
		 * @return __m512i with saturated squares.
		 */
		static inline __m512i									SquareUint32( __m512i _pm512Val ) {
			__m256i m256Low = _mm512_extracti64x4_epi64(_pm512Val, 0);
			__m256i m256High = _mm512_extracti64x4_epi64(_pm512Val, 1);

			__m256i m256Threshold = _mm256_set1_epi32(65535);
			__m256i m256OverflowMaskLow = _mm256_cmpgt_epi32(m256Low, m256Threshold);
			__m256i m256OverflowMaskHigh = _mm256_cmpgt_epi32(m256High, m256Threshold);
			__m256i m256MulLow = _mm256_mullo_epi32(m256Low, m256Low);
			__m256i m256MulHigh = _mm256_mullo_epi32(m256High, m256High);
			__m256i m256Max = _mm256_set1_epi32(-1); /* UINT32_MAX */
			m256MulLow = _mm256_blendv_epi8(m256MulLow, m256Max, m256OverflowMaskLow);
			m256MulHigh = _mm256_blendv_epi8(m256MulHigh, m256Max, m256OverflowMaskHigh);

			return _mm512_inserti64x4(_mm512_castsi256_si512(m256MulLow), m256MulHigh, 1);
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * Multiplies each of the 32 int8 elements in a __m256i by itself.
		 * 
		 * @param _pm256Val the input vector.
		 * @return a __m256i containing the products.
		 */
		static inline __m256i									SquareInt8( __m256i _pm256Val ) {
			__m128i m128Low = _mm256_castsi256_si128( _pm256Val );
			__m128i m128High = _mm256_extracti128_si256( _pm256Val, 1 );
			__m256i m256Low16 = _mm256_cvtepi8_epi16( m128Low );
			__m256i m256High16 = _mm256_cvtepi8_epi16( m128High );
			__m256i m256MulLow = _mm256_mullo_epi16( m256Low16, m256Low16 );
			__m256i m256MulHigh = _mm256_mullo_epi16( m256High16, m256High16 );
			__m128i m128MulLow0 = _mm256_castsi256_si128( m256MulLow );
			__m128i m128MulLow1 = _mm256_extracti128_si256( m256MulLow, 1 );
			__m128i m128MulHigh0 = _mm256_castsi256_si128( m256MulHigh );
			__m128i m128MulHigh1 = _mm256_extracti128_si256( m256MulHigh, 1 );
			__m128i m128LowPacked = _mm_packs_epi16( m128MulLow0, m128MulLow1 );
			__m128i m128HighPacked = _mm_packs_epi16( m128MulHigh0, m128MulHigh1 );
			__m256i m256Result = _mm256_setr_m128i( m128LowPacked, m128HighPacked );
			return m256Result;
		}

		/**
		 * Multiplies each of the 32 uint8 elements in a __m256i by itself.
		 * 
		 * @param _pm256Val the input vector.
		 * @return a __m256i containing the products as uint8 (with saturation).
		 */
		static inline __m256i									SquareUint8( __m256i _pm256Val ) {
			__m128i m128Low = _mm256_castsi256_si128( _pm256Val );
			__m128i m128High = _mm256_extracti128_si256( _pm256Val, 1 );
			__m256i m256Low16 = _mm256_cvtepu8_epi16( m128Low );
			__m256i m256High16 = _mm256_cvtepu8_epi16( m128High );
			__m256i m256MulLow = _mm256_mullo_epi16( m256Low16, m256Low16 );
			__m256i m256MulHigh = _mm256_mullo_epi16( m256High16, m256High16 );
			__m128i m128MulLow0 = _mm256_castsi256_si128( m256MulLow );
			__m128i m128MulLow1 = _mm256_extracti128_si256( m256MulLow, 1 );
			__m128i m128MulHigh0 = _mm256_castsi256_si128( m256MulHigh );
			__m128i m128MulHigh1 = _mm256_extracti128_si256( m256MulHigh, 1 );
			__m128i m128LowPacked = _mm_packus_epi16( m128MulLow0, m128MulLow1 );
			__m128i m128HighPacked = _mm_packus_epi16( m128MulHigh0, m128MulHigh1 );
			return _mm256_setr_m128i( m128LowPacked, m128HighPacked );
		}

		/**
		 * Multiplies each of the 16 int16 elements in a __m256i by itself.
		 * 
		 * @param _pm256Val the input vector.
		 * @return a __m256i containing the products as int16.
		 */
		static inline __m256i									SquareInt16( __m256i _pm256Val ) {
			__m128i m128Low = _mm256_castsi256_si128( _pm256Val );
			__m128i m128High = _mm256_extracti128_si256( _pm256Val, 1 );

			__m256i m256ValLo32 = _mm256_cvtepi16_epi32( m128Low );
			__m256i m256ValHi32 = _mm256_cvtepi16_epi32( m128High );

			__m256i m256MulLo32 = _mm256_mullo_epi32( m256ValLo32, m256ValLo32 );
			__m256i m256MulHi32 = _mm256_mullo_epi32( m256ValHi32, m256ValHi32 );

			__m128i m128LoLo = _mm256_castsi256_si128( m256MulLo32 );
			__m128i m128LoHi = _mm256_extracti128_si256( m256MulLo32, 1 );
			__m128i m128HiLo = _mm256_castsi256_si128( m256MulHi32 );
			__m128i m128HiHi = _mm256_extracti128_si256( m256MulHi32, 1 );

			__m128i m128ResLo = _mm_packs_epi32( m128LoLo, m128LoHi );
			__m128i m128ResHi = _mm_packs_epi32( m128HiLo, m128HiHi );

			return _mm256_setr_m128i( m128ResLo, m128ResHi );
		}

		/**
		 * Multiplies each of the 16 uint16 elements in a __m256i by itself with saturation.
		 * 
		 * @param _pm256Val the input vector (uint16).
		 * @return a __m256i containing the products saturated to uint16.
		 */
		static inline __m256i									SquareUint16( __m256i _pm256Val ) {
			__m128i m128Low = _mm256_castsi256_si128(_pm256Val);
			__m128i m128High = _mm256_extracti128_si256(_pm256Val, 1);

			__m256i m256ValLo32 = _mm256_cvtepu16_epi32(m128Low);
			__m256i m256ValHi32 = _mm256_cvtepu16_epi32(m128High);

			__m256i m256MulLo32 = _mm256_mullo_epi32(m256ValLo32, m256ValLo32);
			__m256i m256MulHi32 = _mm256_mullo_epi32(m256ValHi32, m256ValHi32);

			__m128i m128LoLo = _mm256_castsi256_si128(m256MulLo32);
			__m128i m128LoHi = _mm256_extracti128_si256(m256MulLo32, 1);
			__m128i m128HiLo = _mm256_castsi256_si128(m256MulHi32);
			__m128i m128HiHi = _mm256_extracti128_si256(m256MulHi32, 1);

			__m128i m128ResLo = _mm_packus_epi32(m128LoLo, m128LoHi);
			__m128i m128ResHi = _mm_packus_epi32(m128HiLo, m128HiHi);

			return _mm256_setr_m128i(m128ResLo, m128ResHi);
		}

		/**
		 * Multiplies each of the 8 int32 elements in a __m256i by itself with saturation using packing.
		 * @param _pm256Val the input vector (int32)
		 * @return a __m256i containing the products saturated to int32
		 */
		static inline __m256i									SquareInt32( __m256i _pm256Val ) {
			__m256i m256Abs = _mm256_abs_epi32( _pm256Val );
			__m256i m256Threshold = _mm256_set1_epi32( 46340 );
			__m256i m256OverflowMask = _mm256_cmpgt_epi32( m256Abs, m256Threshold );
			__m256i m256Mul = _mm256_mullo_epi32( _pm256Val, _pm256Val );
			__m256i m256Max = _mm256_set1_epi32( 0x7FFFFFFF );
			return _mm256_blendv_epi8( m256Mul, m256Max, m256OverflowMask );
		}

		/**
		 * Multiplies each of the 8 uint32 elements in a __m256i by itself with saturation using clamping.
		 * 
		 * @param _pm256Val the input vector (uint32).
		 * @return a __m256i containing the products saturated to uint32.
		 */
		static inline __m256i									SquareUint32( __m256i _pm256Val ) {
			__m256i m256Threshold = _mm256_set1_epi32( 65535 );
			__m256i m256OverflowMask = _mm256_cmpgt_epi32( _pm256Val, m256Threshold );
			__m256i m256Mul = _mm256_mullo_epi32( _pm256Val, _pm256Val );
			__m256i m256Max = _mm256_set1_epi32( -1 ); /* UINT32_MAX */
			return _mm256_blendv_epi8( m256Mul, m256Max, m256OverflowMask );
		}

		/**
		 * Squares each int64 in a __m256i with saturation using only AVX2.
		 * Any input whose absolute value is greater than 3037000499 will overflow.
		 * Those results are clamped to INT64_MAX.
		 * 
		 * @param pm256Val input vector (__m256i of int64)
		 * @return __m256i with saturated squares
		 */
		static inline __m256i									SquareInt64( __m256i pm256Val ) {
			__m256i zero = _mm256_setzero_si256();
			__m256i negMask = _mm256_cmpgt_epi64(zero, pm256Val);
			__m256i absVal = _mm256_sub_epi64(_mm256_xor_si256(pm256Val, negMask), negMask);
			__m256i threshold = _mm256_set1_epi64x(3037000499LL);
			__m256i overflowMask = _mm256_cmpgt_epi64(absVal, threshold);
			__m256i maxVal = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);

			// If not overflow, absVal fits in 32 bits, so we can mask and multiply as 32-bit
			__m256i mask32 = _mm256_set1_epi64x(0xFFFFFFFFULL);
			__m256i absVal32 = _mm256_and_si256(absVal, mask32);

			// _mm_mul_epu32 takes lower 32 bits of each 64-bit element and produces a full 64-bit product in that element
			__m128i lo_half = _mm256_castsi256_si128(absVal32);
			__m128i hi_half = _mm256_extracti128_si256(absVal32, 1);
			__m128i mul_lo = _mm_mul_epu32(lo_half, lo_half);
			__m128i mul_hi = _mm_mul_epu32(hi_half, hi_half);
			__m256i mulResult = _mm256_setr_m128i(mul_lo, mul_hi);

			return _mm256_blendv_epi8(mulResult, maxVal, overflowMask);
		}

		/**
		 * Squares each uint64 in a __m256i with saturation using only AVX2.
		 * Any input greater than 4294967295 will overflow.
		 * Those results are clamped to UINT64_MAX.
		 * 
		 * @param pm256Val input vector (__m256i of uint64)
		 * @return __m256i with saturated squares
		 */
		static inline __m256i									SquareUint64( __m256i pm256Val ) {
			__m256i m256Threshold = _mm256_set1_epi64x(4294967295ULL);
			__m256i m256OverflowMask = _mm256_cmpgt_epi64(pm256Val, m256Threshold);
			__m256i m256Max = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL);
			__m256i v_mask = _mm256_set1_epi64x(0xFFFFFFFFULL);
			__m256i v_low = _mm256_and_si256(pm256Val, v_mask);
			__m128i lo_half = _mm256_castsi256_si128(v_low);
			__m128i hi_half = _mm256_extracti128_si256(v_low, 1);
			__m128i mul_lo = _mm_mul_epu32(lo_half, lo_half);
			__m128i mul_hi = _mm_mul_epu32(hi_half, hi_half);
			__m256i m256Mul = _mm256_setr_m128i(mul_lo, mul_hi);
			return _mm256_blendv_epi8(m256Mul, m256Max, m256OverflowMask);
		}
#endif	// #ifdef __AVX2__

	};

}	// namespace nn9

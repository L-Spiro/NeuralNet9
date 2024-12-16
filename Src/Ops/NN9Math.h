/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Math functions.
 */

#pragma once

#include "../Foundation/NN9Math.h"
#include "../Types/NN9BFloat16.h"
#include "../Types/NN9Float16.h"
#include "../Utilities/NN9Utilities.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>


namespace nn9 {

	/**
	 * Class Math
	 * \brief Math functions.
	 *
	 * Description: Math functions.
	 */
	class Math {
	public :

		// == Functions.
		// ===============================
		// Utilities
		// ===============================
		/**
		 * A constexpr function that checks if T is a 64-bit float type.
		 *
		 * \tparam T The type to check.
		 * \return true if T is float, false otherwise.
		 */
		template <typename T>
		static constexpr bool										Is64BitFloat() { return std::is_same<T, double>::value; }

		/**
		 * A constexpr function that checks if T is a 32-bit float type.
		 *
		 * \tparam T The type to check.
		 * \return true if T is float, false otherwise.
		 */
		template <typename T>
		static constexpr bool										Is32BitFloat() { return std::is_same<T, float>::value; }

		/**
		 * A constexpr function that checks if T is a bfloat16_t type.
		 *
		 * \tparam T The type to check.
		 * \return true if T is bfloat16_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool										IsBFloat16() { return std::is_same<T, bfloat16_t>::value; }

		/**
		 * A constexpr function that checks if T is a nn9::float16 type.
		 *
		 * \tparam T The type to check.
		 * \return true if T is nn9::float16, false otherwise.
		 */
		template <typename T>
		static constexpr bool										IsFloat16() { return std::is_same<T, nn9::float16>::value; }

		/**
		 * \brief A constexpr function that checks if T is an unsigned type.
		 *
		 * This relies on std::is_unsigned, which checks if T is an unsigned integral type.
		 * Types like unsigned int, unsigned long, etc., will return true.
		 * Non-integral types or signed integral types will return false.
		 *
		 * \tparam T The type to check.
		 * \return true if T is an unsigned integral type, false otherwise.
		 */
		template <typename T>
		static constexpr bool										IsUnsigned() { return std::is_unsigned<T>::value; }

		/**
		 * Applies the given function to each item in the view.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tFunc The function type.
		 * \param _vValues The input/output view to modify.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tFunc>
		static _tType &												Func( _tType &_vValues, _tFunc _fFunc ) {
			using ValueType = typename _tType::value_type;
#ifdef __AVX512F__
			if constexpr ( IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					// Decode 16 bfloat16_t's at once for super-fast processing.
					bfloat16_t * pSrc = reinterpret_cast<bfloat16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<uint16_t *>(pSrc) );
						_mm512_store_ps( fTmp, mSrc );

						fTmp[0] = float( _fFunc( fTmp[0] ) );
						fTmp[1] = float( _fFunc( fTmp[1] ) );
						fTmp[2] = float( _fFunc( fTmp[2] ) );
						fTmp[3] = float( _fFunc( fTmp[3] ) );
						fTmp[4] = float( _fFunc( fTmp[4] ) );
						fTmp[5] = float( _fFunc( fTmp[5] ) );
						fTmp[6] = float( _fFunc( fTmp[6] ) );
						fTmp[7] = float( _fFunc( fTmp[7] ) );
						fTmp[8] = float( _fFunc( fTmp[8] ) );
						fTmp[9] = float( _fFunc( fTmp[9] ) );
						fTmp[10] = float( _fFunc( fTmp[10] ) );
						fTmp[11] = float( _fFunc( fTmp[11] ) );
						fTmp[12] = float( _fFunc( fTmp[12] ) );
						fTmp[13] = float( _fFunc( fTmp[13] ) );
						fTmp[14] = float( _fFunc( fTmp[14] ) );
						fTmp[15] = float( _fFunc( fTmp[15] ) );

						sSize -= 16;
						__m512 mDst = _mm512_load_ps( fTmp );
						bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pSrc), mDst );
						pSrc += 16;
					}
					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(pSrc)) = _tType::value_type( _fFunc( (*reinterpret_cast<bfloat16_t *>(pSrc)) ) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
			if constexpr ( IsFloat16<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mVal = nn9::float16::Convert16Float16ToFloat32( pSrc );
						_mm512_store_ps( fTmp, mVal );

						fTmp[0] = float( _fFunc( fTmp[0] ) );
						fTmp[1] = float( _fFunc( fTmp[1] ) );
						fTmp[2] = float( _fFunc( fTmp[2] ) );
						fTmp[3] = float( _fFunc( fTmp[3] ) );
						fTmp[4] = float( _fFunc( fTmp[4] ) );
						fTmp[5] = float( _fFunc( fTmp[5] ) );
						fTmp[6] = float( _fFunc( fTmp[6] ) );
						fTmp[7] = float( _fFunc( fTmp[7] ) );
						fTmp[8] = float( _fFunc( fTmp[8] ) );
						fTmp[9] = float( _fFunc( fTmp[9] ) );
						fTmp[10] = float( _fFunc( fTmp[10] ) );
						fTmp[11] = float( _fFunc( fTmp[11] ) );
						fTmp[12] = float( _fFunc( fTmp[12] ) );
						fTmp[13] = float( _fFunc( fTmp[13] ) );
						fTmp[14] = float( _fFunc( fTmp[14] ) );
						fTmp[15] = float( _fFunc( fTmp[15] ) );

						__m512 mDst = _mm512_load_ps( fTmp );
						nn9::float16::Convert16Float32ToFloat16( pSrc, mDst );

						pSrc += 16;
						sSize -= 16;
					}
					while ( sSize ) {
						(*pSrc) = _tType::value_type( _fFunc( (*pSrc) ) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if constexpr ( IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					// Decode 8 bfloat16_t's at once for super-fast processing.
					bfloat16_t * pSrc = reinterpret_cast<bfloat16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<uint16_t *>(pSrc) );
						_mm256_store_ps( fTmp, mSrc );

						fTmp[0] = float( _fFunc( fTmp[0] ) );
						fTmp[1] = float( _fFunc( fTmp[1] ) );
						fTmp[2] = float( _fFunc( fTmp[2] ) );
						fTmp[3] = float( _fFunc( fTmp[3] ) );
						fTmp[4] = float( _fFunc( fTmp[4] ) );
						fTmp[5] = float( _fFunc( fTmp[5] ) );
						fTmp[6] = float( _fFunc( fTmp[6] ) );
						fTmp[7] = float( _fFunc( fTmp[7] ) );

						sSize -= 8;
						__m256 mDst = _mm256_load_ps( fTmp );
						bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pSrc), mDst );
						pSrc += 8;
					}
					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(pSrc)) = _tType::value_type( _fFunc( (*reinterpret_cast<bfloat16_t *>(pSrc)) ) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
			if constexpr ( IsFloat16<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mVal = nn9::float16::Convert8Float16ToFloat32( pSrc );
						_mm256_store_ps( fTmp, mVal );

						fTmp[0] = float( _fFunc( fTmp[0] ) );
						fTmp[1] = float( _fFunc( fTmp[1] ) );
						fTmp[2] = float( _fFunc( fTmp[2] ) );
						fTmp[3] = float( _fFunc( fTmp[3] ) );
						fTmp[4] = float( _fFunc( fTmp[4] ) );
						fTmp[5] = float( _fFunc( fTmp[5] ) );
						fTmp[6] = float( _fFunc( fTmp[6] ) );
						fTmp[7] = float( _fFunc( fTmp[7] ) );

						__m256 mDst = _mm256_load_ps( fTmp );
						nn9::float16::Convert8Float32ToFloat16( pSrc, mDst );

						pSrc += 8;
						sSize -= 8;
					}
					while ( sSize ) {
						(*pSrc) = _tType::value_type( _fFunc( (*pSrc) ) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
#endif	// #ifdef __AVX2__
			for ( auto & aThis : _vValues ) {
				aThis = _tType::value_type( _fFunc( aThis ) );
			}
			return _vValues;
		}

		/**
		 * Applies the given function to each item in the view.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tFunc The function type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \throw If NN9_SAFETY_CHECK, throws if the views are not the same lengths.
		 * \return Returns _vOut.
		 **/
		template <typename _tTypeIn, typename _tTypeOut, typename _tFunc>
		static _tTypeOut &											Func( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tFunc _fFunc ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Func: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
#ifdef __AVX512F__
			if constexpr ( IsBFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					// Decode 16 bfloat16_t's at once for super-fast processing.
					const uint16_t * pSrc = reinterpret_cast<const uint16_t *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( pSrc );
						_mm512_store_ps( fTmp, mSrc );

						if constexpr ( IsBFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );
							fTmp[8] = _fFunc( fTmp[8] );
							fTmp[9] = _fFunc( fTmp[9] );
							fTmp[10] = _fFunc( fTmp[10] );
							fTmp[11] = _fFunc( fTmp[11] );
							fTmp[12] = _fFunc( fTmp[12] );
							fTmp[13] = _fFunc( fTmp[13] );
							fTmp[14] = _fFunc( fTmp[14] );
							fTmp[15] = _fFunc( fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( IsFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );
							fTmp[8] = _fFunc( fTmp[8] );
							fTmp[9] = _fFunc( fTmp[9] );
							fTmp[10] = _fFunc( fTmp[10] );
							fTmp[11] = _fFunc( fTmp[11] );
							fTmp[12] = _fFunc( fTmp[12] );
							fTmp[13] = _fFunc( fTmp[13] );
							fTmp[14] = _fFunc( fTmp[14] );
							fTmp[15] = _fFunc( fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							nn9::float16::Convert16Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							pDst[0] = ValueTypeOut( _fFunc( fTmp[0] ) );
							pDst[1] = ValueTypeOut( _fFunc( fTmp[1] ) );
							pDst[2] = ValueTypeOut( _fFunc( fTmp[2] ) );
							pDst[3] = ValueTypeOut( _fFunc( fTmp[3] ) );
							pDst[4] = ValueTypeOut( _fFunc( fTmp[4] ) );
							pDst[5] = ValueTypeOut( _fFunc( fTmp[5] ) );
							pDst[6] = ValueTypeOut( _fFunc( fTmp[6] ) );
							pDst[7] = ValueTypeOut( _fFunc( fTmp[7] ) );
							pDst[8] = ValueTypeOut( _fFunc( fTmp[8] ) );
							pDst[9] = ValueTypeOut( _fFunc( fTmp[9] ) );
							pDst[10] = ValueTypeOut( _fFunc( fTmp[10] ) );
							pDst[11] = ValueTypeOut( _fFunc( fTmp[11] ) );
							pDst[12] = ValueTypeOut( _fFunc( fTmp[12] ) );
							pDst[13] = ValueTypeOut( _fFunc( fTmp[13] ) );
							pDst[14] = ValueTypeOut( _fFunc( fTmp[14] ) );
							pDst[15] = ValueTypeOut( _fFunc( fTmp[15] ) );
						}

						sSize -= 16;
						pSrc += 16;
						pDst += 16;
					}
					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(pDst++)) = _tTypeIn::value_type( _fFunc( (*reinterpret_cast<const bfloat16_t *>(pSrc++)) ) );
						--sSize;
					}
					return _vOut;
				}
			}
			if constexpr ( IsFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mVal = nn9::float16::Convert16Float16ToFloat32( pSrc );
						_mm512_store_ps( fTmp, mVal );

						if constexpr ( IsBFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );
							fTmp[8] = _fFunc( fTmp[8] );
							fTmp[9] = _fFunc( fTmp[9] );
							fTmp[10] = _fFunc( fTmp[10] );
							fTmp[11] = _fFunc( fTmp[11] );
							fTmp[12] = _fFunc( fTmp[12] );
							fTmp[13] = _fFunc( fTmp[13] );
							fTmp[14] = _fFunc( fTmp[14] );
							fTmp[15] = _fFunc( fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( IsFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );
							fTmp[8] = _fFunc( fTmp[8] );
							fTmp[9] = _fFunc( fTmp[9] );
							fTmp[10] = _fFunc( fTmp[10] );
							fTmp[11] = _fFunc( fTmp[11] );
							fTmp[12] = _fFunc( fTmp[12] );
							fTmp[13] = _fFunc( fTmp[13] );
							fTmp[14] = _fFunc( fTmp[14] );
							fTmp[15] = _fFunc( fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							nn9::float16::Convert16Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							pDst[0] = ValueTypeOut( _fFunc( fTmp[0] ) );
							pDst[1] = ValueTypeOut( _fFunc( fTmp[1] ) );
							pDst[2] = ValueTypeOut( _fFunc( fTmp[2] ) );
							pDst[3] = ValueTypeOut( _fFunc( fTmp[3] ) );
							pDst[4] = ValueTypeOut( _fFunc( fTmp[4] ) );
							pDst[5] = ValueTypeOut( _fFunc( fTmp[5] ) );
							pDst[6] = ValueTypeOut( _fFunc( fTmp[6] ) );
							pDst[7] = ValueTypeOut( _fFunc( fTmp[7] ) );
							pDst[8] = ValueTypeOut( _fFunc( fTmp[8] ) );
							pDst[9] = ValueTypeOut( _fFunc( fTmp[9] ) );
							pDst[10] = ValueTypeOut( _fFunc( fTmp[10] ) );
							pDst[11] = ValueTypeOut( _fFunc( fTmp[11] ) );
							pDst[12] = ValueTypeOut( _fFunc( fTmp[12] ) );
							pDst[13] = ValueTypeOut( _fFunc( fTmp[13] ) );
							pDst[14] = ValueTypeOut( _fFunc( fTmp[14] ) );
							pDst[15] = ValueTypeOut( _fFunc( fTmp[15] ) );
						}

						pSrc += 16;
						sSize -= 16;
					}
					while ( sSize ) {
						(*pDst++) = _tTypeIn::value_type( _fFunc( (*pSrc++) ) );
						--sSize;
					}
					return _vOut;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if constexpr ( IsBFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					// Decode 8 bfloat16_t's at once for super-fast processing.
					const uint16_t * pSrc = reinterpret_cast<const uint16_t *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( pSrc );
						_mm256_store_ps( fTmp, mSrc );

						if constexpr ( IsBFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( IsFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							nn9::float16::Convert8Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							pDst[0] = ValueTypeOut( _fFunc( fTmp[0] ) );
							pDst[1] = ValueTypeOut( _fFunc( fTmp[1] ) );
							pDst[2] = ValueTypeOut( _fFunc( fTmp[2] ) );
							pDst[3] = ValueTypeOut( _fFunc( fTmp[3] ) );
							pDst[4] = ValueTypeOut( _fFunc( fTmp[4] ) );
							pDst[5] = ValueTypeOut( _fFunc( fTmp[5] ) );
							pDst[6] = ValueTypeOut( _fFunc( fTmp[6] ) );
							pDst[7] = ValueTypeOut( _fFunc( fTmp[7] ) );
						}

						sSize -= 8;
						pSrc += 8;
						pDst += 8;
					}
					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(pDst++)) = _tTypeIn::value_type( _fFunc( (*reinterpret_cast<const bfloat16_t *>(pSrc++)) ) );
						--sSize;
					}
					return _vOut;
				}
			}
			if constexpr ( IsFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mVal = nn9::float16::Convert8Float16ToFloat32( pSrc );
						_mm256_store_ps( fTmp, mVal );

						if constexpr ( IsBFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( IsFloat16<ValueTypeOut>() ) {
							fTmp[0] = _fFunc( fTmp[0] );
							fTmp[1] = _fFunc( fTmp[1] );
							fTmp[2] = _fFunc( fTmp[2] );
							fTmp[3] = _fFunc( fTmp[3] );
							fTmp[4] = _fFunc( fTmp[4] );
							fTmp[5] = _fFunc( fTmp[5] );
							fTmp[6] = _fFunc( fTmp[6] );
							fTmp[7] = _fFunc( fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							nn9::float16::Convert8Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							pDst[0] = ValueTypeOut( _fFunc( fTmp[0] ) );
							pDst[1] = ValueTypeOut( _fFunc( fTmp[1] ) );
							pDst[2] = ValueTypeOut( _fFunc( fTmp[2] ) );
							pDst[3] = ValueTypeOut( _fFunc( fTmp[3] ) );
							pDst[4] = ValueTypeOut( _fFunc( fTmp[4] ) );
							pDst[5] = ValueTypeOut( _fFunc( fTmp[5] ) );
							pDst[6] = ValueTypeOut( _fFunc( fTmp[6] ) );
							pDst[7] = ValueTypeOut( _fFunc( fTmp[7] ) );
						}

						pSrc += 8;
						sSize -= 8;
					}
					while ( sSize ) {
						(*pDst++) = _tTypeIn::value_type( _fFunc( (*pSrc++) ) );
						--sSize;
					}
					return _vOut;
				}
			}
#endif	// #ifdef __AVX2__

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				_vOut[i] = _tTypeIn::value_type( _fFunc( _vIn[i] ) );
			}
			return _vOut;
		}

		/**
		 * Applies element-wise abs() to the input.
		 * 
		 * \param _pfInOut The array of int8_t's to abs() in-place.
		 * \param _sSize The total number of int8_t's to which _pfInOut points.
		 **/
		static inline void											Abs_Int8( int8_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 64 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfInOut) );
					_mm512_storeu_si512( _pfInOut, _mm512_abs_epi8( mVal ) );

					_pfInOut += 64;
					_sSize -= 64;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 32 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfInOut) );
					_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfInOut), _mm256_abs_epi8( mVal ) );

					_pfInOut += 32;
					_sSize -= 32;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aVal = (*_pfInOut);
				(*_pfInOut) = static_cast<int8_t>(std::abs( static_cast<int>(aVal) ));
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise abs() to the input.
		 * 
		 * \param _pfInOut The array of int16_t's to abs() in-place.
		 * \param _sSize The total number of int16_t's to which _pfInOut points.
		 **/
		static inline void											Abs_Int16( int16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 32 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfInOut) );
					_mm512_storeu_si512( _pfInOut, _mm512_abs_epi16( mVal ) );

					_pfInOut += 32;
					_sSize -= 32;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 16 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfInOut) );
					_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfInOut), _mm256_abs_epi16( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aVal = (*_pfInOut);
				(*_pfInOut) = static_cast<int16_t>(std::abs( static_cast<int>(aVal) ));
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise abs() to the input.
		 * 
		 * \param _pfInOut The array of int32_t's to abs() in-place.
		 * \param _sSize The total number of int32_t's to which _pfInOut points.
		 **/
		static inline void											Abs_Int32( int32_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfInOut) );
					_mm512_storeu_si512( _pfInOut, _mm512_abs_epi32( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfInOut) );
					_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfInOut), _mm256_abs_epi32( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aVal = (*_pfInOut);
				(*_pfInOut) = static_cast<int32_t>(std::abs( static_cast<int>(aVal) ));
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise abs() to the input.
		 * 
		 * \param _pfInOut The array of int64_t's to abs() in-place.
		 * \param _sSize The total number of int64_t's to which _pfInOut points.
		 **/
		static inline void											Abs_Int64( int64_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfInOut) );
					_mm512_storeu_si512( _pfInOut, _mm512_abs_epi64( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfInOut) );
					_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfInOut), _mm256_abs_epi64( mVal ) );

					_pfInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aVal = (*_pfInOut);
				(*_pfInOut) = aVal < 0 ? -aVal : aVal;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise fabs() to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to fabs() in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Abs_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
			while ( _sSize >= sizeof( uint64_t ) ) {
				(*reinterpret_cast<uint64_t *>(_pfInOut)) &= 0x7FFF7FFF7FFF7FFFULL;
				_sSize -= sizeof( uint64_t ) / sizeof( bfloat16_t );
				_pfInOut += sizeof( uint64_t ) / sizeof( bfloat16_t );
			}

			while ( _sSize ) {
				(*_pfInOut) = std::fabs( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise fabs() to the input.
		 * 
		 * \param _pfInOut The array of float16's to fabs() in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Abs_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
			while ( _sSize >= sizeof( uint64_t ) ) {
				(*reinterpret_cast<uint64_t *>(_pfInOut)) &= 0x7FFF7FFF7FFF7FFFULL;
				_sSize -= sizeof( uint64_t ) / sizeof( nn9::float16 );
				_pfInOut += sizeof( uint64_t ) / sizeof( nn9::float16 );
			}

			while ( _sSize ) {
				(*_pfInOut) = std::fabs( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise fabs() to the input.
		 * 
		 * \param _pfInOut The array of floats to fabs() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Abs_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_abs_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_abs_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::fabs( (*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise fabs() to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to fabs().
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Abs_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_abs_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_abs_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::fabs( static_cast<float>( (*_pfIn++) ) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise fabs() to the input.
		 * 
		 * \param _pdInOut The array of doubles to fabs() in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Abs_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_abs_pd( mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_abs_pd( mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = std::fabs( (*_pdInOut) );
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise fabs() to the input.
		 *
		 * \param _pfIn The array of doubles to fabs().
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Abs_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_abs_pd( mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_abs_pd( mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::fabs( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise abs() to the input.
		 *
		 * \tparam _tTypeOut The output type.
		 * \param _pfIn The array of int8_t's to abs().
		 * \param _pfOut The output array of _tTypeOut's.
		 * \param _sSize The total number of elements to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeOut>
		static inline void											Abs_Int8( const int8_t * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 64 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfIn) );
					mVal = _mm512_abs_epi8( mVal );
					int8_scast( mVal, _pfOut );
					

					_pfIn += 64;
					_pfOut += 64;
					_sSize -= 64;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 32 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfIn) );
					mVal = _mm256_abs_epi8( mVal );
					int8_scast( mVal, _pfOut );


					_pfIn += 32;
					_pfOut += 32;
					_sSize -= 32;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				int8_scast( static_cast<int8_t>(std::abs( static_cast<int>((*_pfIn++)) )), (*_pfOut++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise abs() to the input.
		 *
		 * \tparam _tTypeOut The output type.
		 * \param _pfIn The array of int16_t's to abs().
		 * \param _pfOut The output array of _tTypeOut's.
		 * \param _sSize The total number of elements to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeOut>
		static inline void											Abs_Int16( const int16_t * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 32 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfIn) );
					mVal = _mm512_abs_epi16( mVal );
					int16_scast( mVal, _pfOut );
					

					_pfIn += 32;
					_pfOut += 32;
					_sSize -= 32;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 16 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfIn) );
					mVal = _mm256_abs_epi16( mVal );
					int16_scast( mVal, _pfOut );


					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				int16_scast( static_cast<int16_t>(std::abs( static_cast<int>((*_pfIn++)) )), (*_pfOut++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise abs() to the input.
		 *
		 * \tparam _tTypeOut The output type.
		 * \param _pfIn The array of int32_t's to abs().
		 * \param _pfOut The output array of _tTypeOut's.
		 * \param _sSize The total number of elements to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeOut>
		static inline void											Abs_Int32( const int32_t * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512i mVal = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pfIn) );
					mVal = _mm512_abs_epi32( mVal );
					if constexpr ( std::is_same<_tTypeOut, int8_t>::value || std::is_same<_tTypeOut, uint8_t>::value ) {
						__m512i mZero = _mm512_setzero_si512();
						__m512i mPacked16 = _mm512_packs_epi32( mVal, mZero );
						__m512i mPacked8 = _mm512_packs_epi16( mPacked16, mZero );
						__m256i mLowPacked8 = _mm512_castsi512_si256( mPacked8 );
						__m128i mFirst16Packed8 = _mm256_castsi256_si128( mLowPacked8 );
						_mm_storeu_si128( reinterpret_cast<__m128i *>(_pfOut), mFirst16Packed8 );
					}
					else if constexpr ( std::is_same<_tTypeOut, int32_t>::value || std::is_same<_tTypeOut, uint16_t>::value ) {
						__m512i mZero = _mm512_setzero_si512();
						__m512i mPacked16 = _mm512_packs_epi32( mVal, mZero );

						__m256i mLowPacked16 = _mm512_castsi512_si256( mPacked16 );

						_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfOut), mLowPacked16 );
					}
					else if constexpr ( std::is_same<_tTypeOut, int32_t>::value || std::is_same<_tTypeOut, uint32_t>::value ) {
						_mm512_storeu_si512( _pfOut, mVal );
					}
					else if constexpr ( std::is_same<_tTypeOut, int64_t>::value || std::is_same<_tTypeOut, uint64_t>::value ) {
						__m256i mLower = _mm512_extracti32x8_epi32( mVal, 0 );
						__m256i mUpper = _mm512_extracti32x8_epi32( mVal, 1 );

						__m512i mLower64 = _mm512_cvtepi32_epi64( mLower );
						__m512i mUpper64 = _mm512_cvtepi32_epi64( mUpper );

						_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pfOut), mLower64) ;
						_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pfOut + 8), mUpper64 );
					}
					else if constexpr ( Is32BitFloat<_tTypeOut>() || Is64BitFloat<_tTypeOut>() || IsBFloat16<_tTypeOut>() || IsFloat16<_tTypeOut>() ) {
						__m512 mFloat = _mm512_cvtepi32_ps( mVal );

						if constexpr ( Is32BitFloat<_tTypeOut>() ) {
							_mm512_storeu_ps( reinterpret_cast<float *>(_pfOut), mFloat );
						}
						else if constexpr ( IsFloat16<_tTypeOut>() ) {
							nn9::float16::Convert8Float32ToFloat16( _pfOut, mFloat );
						}
						else if constexpr ( IsBFloat16<_tTypeOut>() ) {
							bfloat16_t::storeu_fp32_to_bf16( _pfOut, mFloat );
						}
						else if constexpr ( Is64BitFloat<_tTypeOut>() ) {
							NN9_ALIGN( 64 )
							float fTmp[16];
							_mm512_storeu_ps( reinterpret_cast<float *>(fTmp), mFloat );
							for ( int i = 0; i < 16; ++i ) {
								_pfOut[i] = static_cast<_tTypeOut>(fTmp[i]);
							}
						}
					}
					else { break; }
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256i mVal = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pfIn) );
					mVal = _mm256_abs_epi32( mVal );
					if constexpr ( std::is_same<_tTypeOut, int8_t>::value || std::is_same<_tTypeOut, uint8_t>::value ) {
						__m256i mZero = _mm256_setzero_si256();
						__m256i mPacked16 = _mm256_packs_epi32( mVal, mZero );
						__m256i mPacked8 = _mm256_packs_epi16( mPacked16, mZero );
						__m128i mLowPacked8_128 = _mm256_castsi256_si128( mPacked8 );
						_mm_storel_epi64( reinterpret_cast<__m128i *>(_pfOut), mLowPacked8_128 );
					}
					else if constexpr ( std::is_same<_tTypeOut, int32_t>::value || std::is_same<_tTypeOut, uint16_t>::value ) {
						__m256i mZero = _mm256_setzero_si256();
						__m256i mPacked16 = _mm256_packs_epi32( mVal, mZero );
						__m128i mLowPacked16_128 = _mm256_castsi256_si128( mPacked16 );
						_mm_storeu_si128( reinterpret_cast<__m128i *>(_pfOut), mLowPacked16_128 );
					}
					else if constexpr ( std::is_same<_tTypeOut, int32_t>::value || std::is_same<_tTypeOut, uint32_t>::value ) {
						_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfOut), mVal );
					}
					else if constexpr ( std::is_same<_tTypeOut, int64_t>::value || std::is_same<_tTypeOut, uint64_t>::value ) {
						__m128i mLower = _mm256_castsi256_si128( mVal );
						__m128i mUpper = _mm256_extracti128_si256( mVal, 1 );

						__m256i mLower64 = _mm256_cvtepi32_epi64( mLower );
						__m256i mUpper64 = _mm256_cvtepi32_epi64( mUpper );

						_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfOut), mLower64 );
						_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pfOut + 4), mUpper64 );
					}
					else if constexpr ( Is32BitFloat<_tTypeOut>() || Is64BitFloat<_tTypeOut>() || IsBFloat16<_tTypeOut>() || IsFloat16<_tTypeOut>() ) {
						__m256 mFloat = _mm256_cvtepi32_ps( mVal );

						if constexpr ( Is32BitFloat<_tTypeOut>() ) {
							_mm256_storeu_ps( reinterpret_cast<float *>(_pfOut), mFloat );
						}
						else if constexpr ( IsFloat16<_tTypeOut>() ) {
							nn9::float16::Convert8Float32ToFloat16( _pfOut, mFloat );
						}
						else if constexpr ( IsBFloat16<_tTypeOut>() ) {
							bfloat16_t::storeu_fp32_to_bf16( _pfOut, mFloat );
						}
						else if constexpr ( Is64BitFloat<_tTypeOut>() ) {
							NN9_ALIGN( 32 )
							float fTmp[8];
							_mm256_storeu_ps( reinterpret_cast<float *>(fTmp), mFloat );
							for ( int i = 0; i < 8; ++i ) {
								_pfOut[i] = static_cast<_tTypeOut>(fTmp[i]);
							}
						}
					}
					else { break; }


					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aVal = (*_pfIn++);
				(*_pfOut++) = static_cast<int32_t>(std::abs( static_cast<int>(aVal) ));
				--_sSize;
			}
		}

		/**
		 * Applies element-wise sqrt() to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to sqrt() in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Sqrt_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_sqrt_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_sqrt_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::sqrt( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise sqrt() to the input.
		 * 
		 * \param _pfInOut The array of float16's to sqrt() in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Sqrt_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_sqrt_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_sqrt_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::sqrt( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise sqrt() to the input.
		 * 
		 * \param _pfInOut The array of floats to sqrt() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Sqrt_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_sqrt_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_sqrt_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::sqrt( (*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise sqrt() to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to sqrt().
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Sqrt_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_sqrt_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_sqrt_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::sqrt( static_cast<float>( (*_pfIn++) ) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise sqrt() to the input.
		 * 
		 * \param _pdInOut The array of doubles to sqrt() in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Sqrt_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_sqrt_pd( mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_sqrt_pd( mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = std::sqrt( (*_pdInOut) );
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise sqrt() to the input.
		 *
		 * \param _pfIn The array of doubles to sqrt().
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Sqrt_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_sqrt_pd( mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_sqrt_pd( mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::sqrt( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise 1/sqrt() to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to 1/sqrt() in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Rsqrt_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( mVal ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_div_ps( _mm256_set1_ps( 1.0f ), _mm256_sqrt_ps( mVal ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = 1.0 / std::sqrt( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise 1/sqrt() to the input.
		 * 
		 * \param _pfInOut The array of float16's to 1/sqrt() in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Rsqrt_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( mVal ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_div_ps( _mm256_set1_ps( 1.0f ), _mm256_sqrt_ps( mVal ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = 1.0 / std::sqrt( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise 1/sqrt() to the input.
		 * 
		 * \param _pfInOut The array of floats to 1/sqrt() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Rsqrt_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( mVal ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_div_ps( _mm256_set1_ps( 1.0f ), _mm256_sqrt_ps( mVal ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = 1.0f / std::sqrt( (*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise 1/sqrt() to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to 1/sqrt().
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Rsqrt_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( mVal ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_div_ps( _mm256_set1_ps( 1.0f ), _mm256_sqrt_ps( mVal ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = 1.0f / std::sqrt( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise 1/sqrt() to the input.
		 * 
		 * \param _pdInOut The array of doubles to 1/sqrt() in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Rsqrt_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_div_pd( _mm512_set1_pd( 1.0 ), _mm512_sqrt_pd( mVal ) ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_div_pd( _mm256_set1_pd( 1.0 ), _mm256_sqrt_pd( mVal ) ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = 1.0 / std::sqrt( (*_pdInOut) );
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise 1/sqrt() to the input.
		 *
		 * \param _pfIn The array of doubles to 1/sqrt().
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Rsqrt_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_div_pd( _mm512_set1_pd( 1.0 ), _mm512_sqrt_pd( mVal ) ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_div_pd( _mm256_set1_pd( 1.0 ), _mm256_sqrt_pd( mVal ) ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = 1.0 / std::sqrt( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*x to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to x*x in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Square_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_mul_ps( mVal, mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_mul_ps( mVal, mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				float fTmp = static_cast<float>(*_pfInOut);
				(*_pfInOut) = fTmp * fTmp;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*x to the input.
		 * 
		 * \param _pfInOut The array of float16's to x*x in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Square_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_mul_ps( mVal, mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_mul_ps( mVal, mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				float fTmp = static_cast<float>(*_pfInOut);
				(*_pfInOut) = fTmp * fTmp;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*x to the input.
		 * 
		 * \param _pfInOut The array of floats to x*x in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Square_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_mul_ps( mVal, mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_mul_ps( mVal, mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				float fTmp = static_cast<float>(*_pfInOut);
				(*_pfInOut) = fTmp * fTmp;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*x to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to x*x.
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Square_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_mul_ps( mVal, mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_mul_ps( mVal, mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				float fTmp = (*_pfIn++);
				(*_pfOut++) = fTmp * fTmp;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*x to the input.
		 * 
		 * \param _pdInOut The array of doubles to x*x in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Square_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_mul_pd( mVal, mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_mul_pd( mVal, mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aTmp = (*_pdInOut);
				(*_pdInOut) = aTmp * aTmp;
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*x to the input.
		 *
		 * \param _pfIn The array of doubles to x*x.
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Square_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_mul_pd( mVal, mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_mul_pd( mVal, mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				auto aTmp = (*_pfIn++);
				(*_pfOut++) = aTmp * aTmp;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise ceil() to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to ceil() in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Ceil_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_ceil_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_ceil_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::ceil( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise ceil() to the input.
		 * 
		 * \param _pfInOut The array of float16's to ceil() in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Ceil_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_ceil_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_ceil_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::ceil( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise ceil() to the input.
		 * 
		 * \param _pfInOut The array of floats to ceil() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Ceil_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_ceil_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_ceil_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::ceil( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise ceil() to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to ceil().
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Ceil_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_ceil_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_ceil_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = _tTypeOut( std::ceil( static_cast<double>( (*_pfIn++) ) ) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise ceil() to the input.
		 * 
		 * \param _pdInOut The array of doubles to ceil() in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Ceil_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_ceil_pd( mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_ceil_pd( mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = std::ceil( (*_pdInOut) );
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise ceil() to the input.
		 *
		 * \param _pfIn The array of doubles to ceil().
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Ceil_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_ceil_pd( mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_ceil_pd( mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::ceil( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise floor() to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to floor() in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Floor_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_floor_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_floor_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::floor( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise floor() to the input.
		 * 
		 * \param _pfInOut The array of float16's to floor() in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Floor_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_floor_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_floor_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::floor( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise floor() to the input.
		 * 
		 * \param _pfInOut The array of floats to floor() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Floor_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_floor_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_floor_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::floor( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise floor() to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to floor().
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Floor_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_floor_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_floor_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = _tTypeOut( std::floor( static_cast<double>( (*_pfIn++) ) ) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise floor() to the input.
		 * 
		 * \param _pdInOut The array of doubles to floor() in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Floor_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_floor_pd( mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_floor_pd( mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = std::floor( (*_pdInOut) );
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise floor() to the input.
		 *
		 * \param _pfIn The array of doubles to floor().
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Floor_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_floor_pd( mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_floor_pd( mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::floor( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise trunc() to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to trunc() in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 **/
		static inline void											Trunc_BFloat16( bfloat16_t * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_trunc_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_trunc_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::trunc( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise trunc() to the input.
		 * 
		 * \param _pfInOut The array of float16's to trunc() in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 **/
		static inline void											Trunc_Float16( nn9::float16 * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_trunc_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_trunc_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::trunc( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise trunc() to the input.
		 * 
		 * \param _pfInOut The array of floats to trunc() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 **/
		static inline void											Trunc_Float( float * _pfInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_trunc_ps( mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_trunc_ps( mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = std::trunc( static_cast<float>(*_pfInOut) );
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise trunc() to the input.
		 *
		 * \tparam _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \tparam _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to trunc().
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats/float16/bfloat16_t's to which _pfIn and _pfOut point.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Trunc_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_trunc_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_trunc_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = _tTypeOut( std::trunc( static_cast<double>( (*_pfIn++) ) ) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise trunc() to the input.
		 * 
		 * \param _pdInOut The array of doubles to trunc() in-place.
		 * \param _sSize The total number of doubles to which _pdInOut points.
		 **/
		static inline void											Trunc_Double( double * _pdInOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_trunc_pd( mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_trunc_pd( mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = std::trunc( (*_pdInOut) );
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise trunc() to the input.
		 *
		 * \param _pfIn The array of doubles to trunc().
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 **/
		static inline void											Trunc_Double( const double * _pfIn, double * _pfOut, size_t _sSize ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_trunc_pd( mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_trunc_pd( mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = std::trunc( (*_pfIn++) );
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x+s to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to x+s in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 * \param _fScalar The scalar to add to the elements in _pfInOut.
		 **/
		static inline void											Add_BFloat16( bfloat16_t * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_add_ps( _mm512_set1_ps( _fScalar ), mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_add_ps( _mm256_set1_ps( _fScalar ), mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) + _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x+s to the input.
		 * 
		 * \param _pfInOut The array of float16's to x+s in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 * \param _fScalar The scalar to add to the elements in _pfInOut.
		 **/
		static inline void											Add_Float16( nn9::float16 * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_add_ps( _mm512_set1_ps( _fScalar ), mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_add_ps( _mm256_set1_ps( _fScalar ), mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) + _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x+s to the input.
		 * 
		 * \param _pfInOut The array of floats to x+s in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to add to the elements in _pfInOut.
		 **/
		static inline void											Add_Float( float * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_add_ps( _mm512_set1_ps( _fScalar ), mVal ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_add_ps( _mm256_set1_ps( _fScalar ), mVal ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) + _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x+s to the input.
		 *
		 * \param _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \param _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to x+s.
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to add to the elements in _pfIn.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Add_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_add_ps( _mm512_set1_ps( _fScalar ), mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_add_ps( _mm256_set1_ps( _fScalar ), mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = static_cast<float>(*_pfIn++) + _fScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x+s to the input.
		 * 
		 * \param _pdInOut The array of floats to x+s in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
		 * \param _dScalar The scalar to add to the elements in _pfInOut.
		 **/
		static inline void											Add_Double( double * _pdInOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_add_pd( _mm512_set1_pd( _dScalar ), mVal ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_add_pd( _mm256_set1_pd( _dScalar ), mVal ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = (*_pdInOut) + _dScalar;
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x+s to the input.
		 *
		 * \param _pfIn The array of doubles to x+s.
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 * \param _dScalar The scalar to add to the elements in _pfIn.
		 **/
		static inline void											Add_Double( const double * _pfIn, double * _pfOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_add_pd( _mm512_set1_pd( _dScalar ), mVal ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_add_pd( _mm256_set1_pd( _dScalar ), mVal ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = (*_pfIn++) + _dScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x-s to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to x-s in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 * \param _fScalar The scalar to sub to the elements in _pfInOut.
		 **/
		static inline void											Sub_BFloat16( bfloat16_t * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_sub_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_sub_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) - _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x-s to the input.
		 * 
		 * \param _pfInOut The array of float16's to x-s in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 * \param _fScalar The scalar to sub to the elements in _pfInOut.
		 **/
		static inline void											Sub_Float16( nn9::float16 * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_sub_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_sub_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) - _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x-s to the input.
		 * 
		 * \param _pfInOut The array of floats to x-s in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to sub to the elements in _pfInOut.
		 **/
		static inline void											Sub_Float( float * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_sub_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_sub_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) - _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x-s to the input.
		 *
		 * \param _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \param _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to x-s.
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to sub to the elements in _pfIn.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Sub_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_sub_ps( mVal, _mm512_set1_ps( _fScalar ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_sub_ps( mVal, _mm256_set1_ps( _fScalar ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = static_cast<float>(*_pfIn++) - _fScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x-s to the input.
		 * 
		 * \param _pdInOut The array of floats to x-s in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
		 * \param _dScalar The scalar to sub to the elements in _pfInOut.
		 **/
		static inline void											Sub_Double( double * _pdInOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_sub_pd( mVal, _mm512_set1_pd( _dScalar ) ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_sub_pd( mVal, _mm256_set1_pd( _dScalar ) ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = (*_pdInOut) - _dScalar;
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x-s to the input.
		 *
		 * \param _pfIn The array of doubles to x-s.
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 * \param _dScalar The scalar to sub to the elements in _pfIn.
		 **/
		static inline void											Sub_Double( const double * _pfIn, double * _pfOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_sub_pd( mVal, _mm512_set1_pd( _dScalar ) ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_sub_pd( mVal, _mm256_set1_pd( _dScalar ) ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = (*_pfIn++) - _dScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*s to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to x*s in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 * \param _fScalar The scalar to mul to the elements in _pfInOut.
		 **/
		static inline void											Mul_BFloat16( bfloat16_t * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_mul_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_mul_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) * _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*s to the input.
		 * 
		 * \param _pfInOut The array of float16's to x*s in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 * \param _fScalar The scalar to mul to the elements in _pfInOut.
		 **/
		static inline void											Mul_Float16( nn9::float16 * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_mul_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_mul_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) * _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*s to the input.
		 * 
		 * \param _pfInOut The array of floats to x*s in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to mul to the elements in _pfInOut.
		 **/
		static inline void											Mul_Float( float * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_mul_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_mul_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) * _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*s to the input.
		 *
		 * \param _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \param _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to x*s.
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to mul to the elements in _pfIn.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Mul_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_mul_ps( mVal, _mm512_set1_ps( _fScalar ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_mul_ps( mVal, _mm256_set1_ps( _fScalar ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = static_cast<float>(*_pfIn++) * _fScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*s to the input.
		 * 
		 * \param _pdInOut The array of floats to x*s in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
		 * \param _dScalar The scalar to mul to the elements in _pfInOut.
		 **/
		static inline void											Mul_Double( double * _pdInOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_mul_pd( mVal, _mm512_set1_pd( _dScalar ) ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_mul_pd( mVal, _mm256_set1_pd( _dScalar ) ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = (*_pdInOut) * _dScalar;
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x*s to the input.
		 *
		 * \param _pfIn The array of doubles to x*s.
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 * \param _dScalar The scalar to mul to the elements in _pfIn.
		 **/
		static inline void											Mul_Double( const double * _pfIn, double * _pfOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_mul_pd( mVal, _mm512_set1_pd( _dScalar ) ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_mul_pd( mVal, _mm256_set1_pd( _dScalar ) ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = (*_pfIn++) * _dScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x/s to the input.
		 * 
		 * \param _pfInOut The array of bfloat16_t's to x/s in-place.
		 * \param _sSize The total number of bfloat16_t's to which _pfInOut points.
		 * \param _fScalar The scalar to div to the elements in _pfInOut.
		 **/
		static inline void											Div_BFloat16( bfloat16_t * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm512_div_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfInOut) );
					bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfInOut), _mm256_div_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) / _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x/s to the input.
		 * 
		 * \param _pfInOut The array of float16's to x/s in-place.
		 * \param _sSize The total number of float16's to which _pfInOut points.
		 * \param _fScalar The scalar to div to the elements in _pfInOut.
		 **/
		static inline void											Div_Float16( nn9::float16 * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = nn9::float16::Convert16Float16ToFloat32( _pfInOut );
					nn9::float16::Convert16Float32ToFloat16( _pfInOut, _mm512_div_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = nn9::float16::Convert8Float16ToFloat32( _pfInOut );
					nn9::float16::Convert8Float32ToFloat16( _pfInOut, _mm256_div_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) / _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x/s to the input.
		 * 
		 * \param _pfInOut The array of floats to x/s in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to div to the elements in _pfInOut.
		 **/
		static inline void											Div_Float( float * _pfInOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal = _mm512_loadu_ps( _pfInOut );
					_mm512_storeu_ps( _pfInOut, _mm512_div_ps( mVal, _mm512_set1_ps( _fScalar ) ) );

					_pfInOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal = _mm256_loadu_ps( _pfInOut );
					_mm256_storeu_ps( _pfInOut, _mm256_div_ps( mVal, _mm256_set1_ps( _fScalar ) ) );

					_pfInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfInOut) = static_cast<float>(*_pfInOut) / _fScalar;
				++_pfInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x/s to the input.
		 *
		 * \param _tTypeIn The input type.  Must be float, bfloat16_t, or float16.
		 * \param _tTypeOut The output type.  Must be float, bfloat16_t, or float16.
		 * \param _pfIn The array of floats/float16/bfloat16_t's to x/s.
		 * \param _pfOut The output array of floats/float16/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
		 * \param _fScalar The scalar to div to the elements in _pfIn.
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static inline void											Div_Float( const _tTypeIn * _pfIn, _tTypeOut * _pfOut, size_t _sSize, float _fScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 16 ) {
					__m512 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert16Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_div_ps( mVal, _mm512_set1_ps( _fScalar ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert16Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm512_storeu_ps( _pfOut, mVal );
					}
					

					_pfIn += 16;
					_pfOut += 16;
					_sSize -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 8 ) {
					__m256 mVal;
					if constexpr ( IsBFloat16<_tTypeIn>() ) {
						mVal = nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pfIn) );
					}
					else if constexpr ( IsFloat16<_tTypeIn>() ) {
						mVal = nn9::float16::Convert8Float16ToFloat32( _pfIn );
					}
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_div_ps( mVal, _mm256_set1_ps( _fScalar ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
					}
					else if constexpr ( IsFloat16<_tTypeOut>() ) {
						nn9::float16::Convert8Float32ToFloat16( _pfOut, mVal );
					}
					else {
						_mm256_storeu_ps( _pfOut, mVal );
					}

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = static_cast<float>(*_pfIn++) / _fScalar;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x/s to the input.
		 * 
		 * \param _pdInOut The array of floats to x/s in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
		 * \param _dScalar The scalar to div to the elements in _pfInOut.
		 **/
		static inline void											Div_Double( double * _pdInOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pdInOut );
					_mm512_storeu_pd( _pdInOut, _mm512_div_pd( mVal, _mm512_set1_pd( _dScalar ) ) );

					_pdInOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pdInOut );
					_mm256_storeu_pd( _pdInOut, _mm256_div_pd( mVal, _mm256_set1_pd( _dScalar ) ) );

					_pdInOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pdInOut) = (*_pdInOut) / _dScalar;
				++_pdInOut;
				--_sSize;
			}
		}

		/**
		 * Applies element-wise x/s to the input.
		 *
		 * \param _pfIn The array of doubles to x/s.
		 * \param _pfOut The output array of doubles.
		 * \param _sSize The total number of doubles to which _pfIn and _pfOut point.
		 * \param _dScalar The scalar to div to the elements in _pfIn.
		 **/
		static inline void											Div_Double( const double * _pfIn, double * _pfOut, size_t _sSize, double _dScalar ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				while ( _sSize >= 8 ) {
					__m512d mVal = _mm512_loadu_pd( _pfIn );
					_mm512_storeu_pd( _pfOut, _mm512_div_pd( mVal, _mm512_set1_pd( _dScalar ) ) );

					_pfIn += 8;
					_pfOut += 8;
					_sSize -= 8;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				while ( _sSize >= 4 ) {
					__m256d mVal = _mm256_loadu_pd( _pfIn );
					_mm256_storeu_pd( _pfOut, _mm256_div_pd( mVal, _mm256_set1_pd( _dScalar ) ) );

					_pfIn += 4;
					_pfOut += 4;
					_sSize -= 4;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sSize ) {
				(*_pfOut++) = (*_pfIn++) / _dScalar;
				--_sSize;
			}
		}


		// ===============================
		// Basic Operations
		// ===============================
		/**
		 * Compute absolute values using pure C++.
		 *
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Abs( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsUnsigned<ValueType>() ) { return _vValues; }

			if constexpr ( IsFloat16<ValueType>() ) {
				Abs_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Abs_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Abs_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Abs_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( std::is_same<ValueType, int8_t>::value ) {
				Abs_Int8( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( std::is_same<ValueType, int16_t>::value ) {
				Abs_Int16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( std::is_same<ValueType, int32_t>::value ) {
				Abs_Int32( &_vValues[0], _vValues.size() );
				return _vValues;
			}


			for ( std::size_t i = 0; i < _vValues.size(); ++i ) {
				_vValues[i] = ValueType( std::fabs( static_cast<double>(_vValues[i]) ) );
			}
			return _vValues;
		}

		/**
		 * Applies Abs() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Abs( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Abs( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise abs().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Abs( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Abs: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Abs_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Abs: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Abs_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}


			if constexpr ( std::is_same<ValueTypeIn, int8_t>::value ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Abs: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Abs_Int8( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( std::is_same<ValueTypeIn, int16_t>::value ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Abs: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Abs_Int16( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( std::is_same<ValueTypeIn, int32_t>::value ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Abs: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Abs_Int32( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}

			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::abs( x ); } );
		}

		/**
		 * Applies Abs() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Abs( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Abs: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Abs( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}


		// ===============================
		// Sin/Cos/Tan
		// ===============================
		/**
		 * Computes element-wise acos().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Acos( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::acos( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Acos() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Acos( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Acos( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise acos().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Acos( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::acos( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Acos() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Acos( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Acos: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Acos( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise asin().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Asin( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::asin( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Asin() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Asin( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Asin( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise asin().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Asin( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::asin( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Asin() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Asin( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Asin: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Asin( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise atan().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Atan( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::atan( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Atan() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Atan( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Atan( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise atan().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Atan( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::atan( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Atan() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Atan( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Atan: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Atan( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise acosh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Acosh( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::acosh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Acosh() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Acosh( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Acosh( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise acosh().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Acosh( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::acosh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Acosh() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Acosh( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Acosh: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Acosh( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise asinh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Asinh( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::asinh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Asinh() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Asinh( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Asinh( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise asinh().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Asinh( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::asinh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Asinh() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Asinh( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Asinh: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Asinh( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise atanh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Atanh( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::atanh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Atanh() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Atanh( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Atanh( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise atanh().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Atanh( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::atanh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Atanh() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Atanh( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Atanh: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Atanh( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise cos().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Cos( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::cos( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Cos() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Cos( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Cos( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise cos().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Cos( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::cos( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Cos() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Cos( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Cos: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Cos( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise cosh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Cosh( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::cosh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Cosh() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Cosh( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Cosh( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise cosh().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Cosh( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::cosh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Cosh() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Cosh( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Cosh: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Cosh( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise sin().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Sin( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::sin( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Sin() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Sin( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Sin( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise sin().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Sin( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::sin( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Sin() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Sin( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sin: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Sin( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise sinh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Sinh( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::sinh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Sinh() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Sinh( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Sinh( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise sinh().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Sinh( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::sinh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Sinh() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Sinh( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sinh: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Sinh( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise tan().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Tan( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::tan( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Tan() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Tan( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Tan( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise tan().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Tan( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::tan( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Tan() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Tan( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Tan: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Tan( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise tanh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Tanh( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::tanh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Tanh() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Tanh( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Tanh( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise tanh().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Tanh( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::tanh( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Tanh() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Tanh( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Tanh: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Tanh( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}


		// ===============================
		// Exponential
		// ===============================
		/**
		 * Computes element-wise square().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Square( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Square_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Square_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Square_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Square_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			return Func<_tType>( _vValues, [](auto x) { return x * x; } );
		}

		/**
		 * Applies Square() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Square( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Square( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise square().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Square( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Square: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Square_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Square: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Square_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x * x; } );
		}

		/**
		 * Applies Square() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Square( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Square: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Square( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise sqrt().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Sqrt( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Sqrt_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Sqrt_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Sqrt_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Sqrt_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			return Func<_tType>( _vValues, [](auto x) { return std::sqrt( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Sqrt() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Sqrt( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Sqrt( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise sqrt().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Sqrt( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Sqrt_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Sqrt_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::sqrt( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Sqrt() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Sqrt( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Sqrt( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise 1/sqrt().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Rsqrt( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Rsqrt_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Rsqrt_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Rsqrt_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Rsqrt_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			return Func<_tType>( _vValues, [](auto x) { return 1.0 / std::sqrt( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Rsqrt() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Rsqrt( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Rsqrt( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise 1.0/sqrt().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Rsqrt( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Rsqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Rsqrt_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Rsqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Rsqrt_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return 1.0 / std::sqrt( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Rsqrt() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Rsqrt( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Rsqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Rsqrt( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise exp().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Exp( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::exp( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Exp() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Exp( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Exp( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise exp().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Exp( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::exp( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Exp() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Exp( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Exp: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Exp( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise expm1().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Expm1( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::expm1( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Expm1() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Expm1( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Expm1( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise expm1().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Expm1( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::expm1( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Expm1() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Expm1( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Expm1: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Expm1( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise log().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Log( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::log( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Log( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Log( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise log().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Log( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::log( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Log( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Log: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Log( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise log2().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Log2( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::log2( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log2() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Log2( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Log2( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise log2().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Log2( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::log2( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log2() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Log2( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Log2: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Log2( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise log10().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Log10( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::log10( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log10() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Log10( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Log10( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise log10().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Log10( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::log10( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log10() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Log10( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Log10: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Log10( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise log1p().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Log1p( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::log1p( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log1p() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Log1p( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Log1p( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise log1p().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Log1p( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::log1p( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Log1p() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Log1p( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Log1p: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Log1p( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}


		// ===============================
		// Rounding
		// ===============================
		/**
		 * Computes element-wise ceil().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Ceil( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Ceil_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Ceil_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Ceil_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Ceil_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			return Func<_tType>( _vValues, [](auto x) { return std::ceil( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Ceil() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Ceil( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Ceil( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise ceil().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Ceil( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Ceil: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Ceil_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Ceil: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Ceil_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::ceil( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Ceil() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Ceil( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Ceil: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Ceil( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise floor().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Floor( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Floor_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Floor_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Floor_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Floor_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			return Func<_tType>( _vValues, [](auto x) { return std::floor( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Floor() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Floor( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Floor( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise floor().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Floor( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Floor: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Floor_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Floor: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Floor_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::floor( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Floor() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Floor( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Floor: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Floor( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise trunc().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Trunc( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Trunc_Float16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Trunc_BFloat16( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Trunc_Float( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Trunc_Double( &_vValues[0], _vValues.size() );
				return _vValues;
			}
			return Func<_tType>( _vValues, [](auto x) { return std::trunc( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Trunc() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Trunc( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Trunc( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise trunc().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Trunc( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Trunc: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Trunc_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Trunc: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Trunc_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::trunc( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Trunc() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Trunc( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Trunc: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Trunc( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise round().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Round( _tType &_vValues ) {
			return Func<_tType>( _vValues, [](auto x) { return std::round( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Round() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Round( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) {
				Round( aThis );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise round().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Round( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::round( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Round() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Round( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Round: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Round( _vIn[i], _vOut[i] );
			}
			return _vOut;
		}


		// ===============================
		// Summation
		// ===============================
		/**
		 * \brief Computes the sum of a sequence of numbers using the Kahan summation algorithm.
		 * 
		 * \param _vValues The vector of numbers to sum.
		 * \return The accurate sum as a double.
		 */
		template <typename _tType>
		static double												KahanSum( _tType &_vValues ) {
			double dSum = 0.0;	// Running total.
			double dC = 0.0;	// Compensation for lost low-order bits.

			for ( const double dValue : _vValues ) {
				double dY = dValue - dC;			// Apply compensation.
				double dT = dSum + dY;				// Temporary sum.
				dC = (dT - dSum) - dY;				// Update compensation.
				dSum = dT;							// Update running total.
			}

			return dSum;
		}

		/**
		 * Applies KahanSum() to an array of inputs.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tOutType The output view/container type.
		 * \param _vValues The input view/container.
		 * \param _vOut The output view/container.
		 * \throw If NN9_SAFETY_CHECK, throws if _vValues and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 **/
		template <typename _tType, typename _tOutType>
		static _tOutType &											KahanSum( const std::vector<_tType> &_vValues, _tOutType &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vValues.size() != _vOut.size() ) { throw std::runtime_error( "Math::KahanSum: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
			for ( size_t i = 0; i < _vValues.size(); ++i ) {
				_vOut[i] = _tOutType::value_type( KahanSum( _vValues[i] ) );
			}
			return _vOut;
		}

		/**
		 * \brief Computes the sum of a sequence of numbers.
		 * 
		 * \param _vValues The vector of numbers to sum.
		 * \return The sum as a double.
		 */
		template <typename _tType>
		static double												Sum( _tType &_vValues ) {
			using ValueType = typename _tType::value_type;

			double dSum = 0.0;
			ValueType * pvtThis = &_vValues[0];
			size_t sSize = _vValues.size();

#ifdef __AVX512F__
			if constexpr ( IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					// Decode 16 bfloat16_t's at once for super-fast processing.
					__m512 mSum = _mm512_setzero_ps();
					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<uint16_t *>(pvtThis) );
						mSum = _mm512_add_ps( mSum, mSrc );

						sSize -= 16;
						pvtThis += 16;
					}
					dSum += _mm512_reduce_add_ps( mSum );
				}
			}
			else if constexpr ( Is32BitFloat<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					constexpr size_t sRegSize = sizeof( __m512 ) / sizeof( float );
					__m512 mSum = _mm512_setzero_ps();
					while ( sSize >= sRegSize ) {
						__m512 mSrc = _mm512_loadu_ps( pvtThis );
						mSum = _mm512_add_ps( mSum, mSrc );

						sSize -= sRegSize;
						pvtThis += sRegSize;
					}
					dSum += _mm512_reduce_add_ps( mSum );
				}
			}
			else if constexpr ( Is64BitFloat<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					constexpr size_t sRegSize = sizeof( __m512d ) / sizeof( double );
					__m512d mSum = _mm512_setzero_pd();
					while ( sSize >= sRegSize ) {
						__m512d mSrc = _mm512_loadu_pd( pvtThis );
						mSum = _mm512_add_pd( mSum, mSrc );

						sSize -= sRegSize;
						pvtThis += sRegSize;
					}
					dSum += _mm512_reduce_add_pd( mSum );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if constexpr ( IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					// Decode 8 bfloat16_t's at once for super-fast processing.
					__m256 mSum = _mm256_setzero_ps();
					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<uint16_t *>(pvtThis) );
						mSum = _mm256_add_ps( mSum, mSrc );

						sSize -= 8;
						pvtThis += 8;
					}
					dSum += Utilities::HorizontalSum( mSum );
				}
			}
			else if constexpr ( Is32BitFloat<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					constexpr size_t sRegSize = sizeof( __m256 ) / sizeof( float );
					__m256 mSum = _mm256_setzero_ps();
					while ( sSize >= sRegSize ) {
						__m256 mSrc = _mm256_loadu_ps( pvtThis );
						mSum = _mm256_add_ps( mSum, mSrc );

						sSize -= sRegSize;
						pvtThis += sRegSize;
					}
					dSum += Utilities::HorizontalSum( mSum );
				}
			}
			else if constexpr ( Is64BitFloat<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					constexpr size_t sRegSize = sizeof( __m256d ) / sizeof( double );
					__m256d mSum = _mm256_setzero_pd();
					while ( sSize >= sRegSize ) {
						__m256d mSrc = _mm256_loadu_pd( pvtThis );
						mSum = _mm256_add_pd( mSum, mSrc );

						sSize -= sRegSize;
						pvtThis += sRegSize;
					}
					dSum += Utilities::HorizontalSum( mSum );
				}
			}
#endif	// #ifdef __AVX2__

			while ( sSize-- ) {
				dSum += double( (*pvtThis++) );
			}

			return dSum;
		}

		/**
		 * Applies Sum() to an array of inputs.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tOutType The output view/container type.
		 * \param _vValues The input view/container.
		 * \param _vOut The output view/container.
		 * \throw If NN9_SAFETY_CHECK, throws if _vValues and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 **/
		template <typename _tType, typename _tOutType>
		static _tOutType &											Sum( const std::vector<_tType> &_vValues, _tOutType &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vValues.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sum: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
			for ( size_t i = 0; i < _vValues.size(); ++i ) {
				_vOut[i] = _tOutType::value_type( Sum( _vValues[i] ) );
			}
			return _vOut;
		}


		// ===============================
		// Scalars
		// ===============================
		/**
		 * Computes element-wise add.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view.
		 * \param _stScalar The scalar to add element-wise.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Add( _tType &_vValues, _tScalarType _stScalar ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Add_Float16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Add_BFloat16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Add_Float( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Add_Double( &_vValues[0], _vValues.size(), static_cast<double>(_stScalar) );
				return _vValues;
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return x + _stScalar; } );
		}

		/**
		 * Applies Add() to an array of inputs.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view to modify.
		 * \param _stScalar The scalar to add element-wise.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Add( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) {
				Add( aThis, _stScalar );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise add.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to add element-wise.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Add( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Add: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Add_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Add: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Add_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return x + _stScalar; } );
		}

		/**
		 * Applies Add() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to add element-wise.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Add( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Add: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Add( _vIn[i], _vOut[i], _stScalar );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise sub.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view.
		 * \param _stScalar The scalar to sub element-wise.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Sub( _tType &_vValues, _tScalarType _stScalar ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Sub_Float16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Sub_BFloat16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Sub_Float( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Sub_Double( &_vValues[0], _vValues.size(), static_cast<double>(_stScalar) );
				return _vValues;
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return x - _stScalar; } );
		}

		/**
		 * Applies Sub() to an array of inputs.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view to modify.
		 * \param _stScalar The scalar to sub element-wise.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Sub( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) {
				Sub( aThis, _stScalar );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise sub.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to sub element-wise.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Sub( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sub: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Sub_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sub: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Sub_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return x - _stScalar; } );
		}

		/**
		 * Applies Sub() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to sub element-wise.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Sub( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sub: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Sub( _vIn[i], _vOut[i], _stScalar );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise mul.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view.
		 * \param _stScalar The scalar to mul element-wise.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Mul( _tType &_vValues, _tScalarType _stScalar ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Mul_Float16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Mul_BFloat16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Mul_Float( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Mul_Double( &_vValues[0], _vValues.size(), static_cast<double>(_stScalar) );
				return _vValues;
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return x * _stScalar; } );
		}

		/**
		 * Applies Mul() to an array of inputs.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view to modify.
		 * \param _stScalar The scalar to mul element-wise.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Mul( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) {
				Mul( aThis, _stScalar );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise mul.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to mul element-wise.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Mul( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Mul: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Mul_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Mul: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Mul_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return x * _stScalar; } );
		}

		/**
		 * Applies Mul() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to mul element-wise.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Mul( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Mul: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Mul( _vIn[i], _vOut[i], _stScalar );
			}
			return _vOut;
		}

		/**
		 * Computes element-wise div.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view.
		 * \param _stScalar The scalar to div element-wise.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Div( _tType &_vValues, _tScalarType _stScalar ) {
			using ValueType = typename _tType::value_type;
			if constexpr ( IsFloat16<ValueType>() ) {
				Div_Float16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( IsBFloat16<ValueType>() ) {
				Div_BFloat16( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is32BitFloat<ValueType>() ) {
				Div_Float( &_vValues[0], _vValues.size(), static_cast<float>(_stScalar) );
				return _vValues;
			}
			if constexpr ( Is64BitFloat<ValueType>() ) {
				Div_Double( &_vValues[0], _vValues.size(), static_cast<double>(_stScalar) );
				return _vValues;
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return x / _stScalar; } );
		}

		/**
		 * Applies Div() to an array of inputs.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vValues The input/output view to modify.
		 * \param _stScalar The scalar to div element-wise.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Div( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) {
				Div( aThis, _stScalar );
			}
			return _vValues;
		}

		/**
		 * Computes element-wise div.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to div element-wise.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Div( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
			if constexpr ( (IsBFloat16<ValueTypeIn>() || Is32BitFloat<ValueTypeIn>()) &&
				(IsFloat16<ValueTypeIn>() || Is32itFloat<ValueTypeIn>()) &&
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Div: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Div_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Div: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Div_Double( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return x / _stScalar; } );
		}

		/**
		 * Applies Div() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tScalarType The scalar type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _stScalar The scalar to div element-wise.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Div( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Div: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Div( _vIn[i], _vOut[i], _stScalar );
			}
			return _vOut;
		}


		// ===============================
		// Element-wise Algebra
		// ===============================

	};

}	// namespace nn9

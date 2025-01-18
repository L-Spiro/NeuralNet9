/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Math functions.
 */

#pragma once

#include "../Foundation/NN9Intrin.h"
#include "../Foundation/NN9Math.h"
#include "../Types/NN9BFloat16.h"
#include "../Types/NN9Float16.h"
#include "../Types/NN9Types.h"
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
			if constexpr ( nn9::Types::IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					// Decode 16 bfloat16_t's at once for super-fast processing.
					bfloat16_t * pSrc = reinterpret_cast<bfloat16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<uint16_t *>(pSrc) );
						_mm512_store_ps( fTmp, mSrc );

						Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
						Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
						Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
						Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
						Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
						Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
						Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
						Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );
						Intrin::scast( _fFunc( fTmp[8] ), fTmp[8] );
						Intrin::scast( _fFunc( fTmp[9] ), fTmp[9] );
						Intrin::scast( _fFunc( fTmp[10] ), fTmp[10] );
						Intrin::scast( _fFunc( fTmp[11] ), fTmp[11] );
						Intrin::scast( _fFunc( fTmp[12] ), fTmp[12] );
						Intrin::scast( _fFunc( fTmp[13] ), fTmp[13] );
						Intrin::scast( _fFunc( fTmp[14] ), fTmp[14] );
						Intrin::scast( _fFunc( fTmp[15] ), fTmp[15] );

						sSize -= 16;
						__m512 mDst = _mm512_load_ps( fTmp );
						bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pSrc), mDst );
						pSrc += 16;
					}
					while ( sSize ) {
						Intrin::scast( _fFunc( (*pSrc) ), (*pSrc) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
			if constexpr ( nn9::Types::IsFloat16<ValueType>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mVal = nn9::float16::Convert16Float16ToFloat32( pSrc );
						_mm512_store_ps( fTmp, mVal );

						Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
						Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
						Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
						Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
						Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
						Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
						Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
						Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );
						Intrin::scast( _fFunc( fTmp[8] ), fTmp[8] );
						Intrin::scast( _fFunc( fTmp[9] ), fTmp[9] );
						Intrin::scast( _fFunc( fTmp[10] ), fTmp[10] );
						Intrin::scast( _fFunc( fTmp[11] ), fTmp[11] );
						Intrin::scast( _fFunc( fTmp[12] ), fTmp[12] );
						Intrin::scast( _fFunc( fTmp[13] ), fTmp[13] );
						Intrin::scast( _fFunc( fTmp[14] ), fTmp[14] );
						Intrin::scast( _fFunc( fTmp[15] ), fTmp[15] );

						__m512 mDst = _mm512_load_ps( fTmp );
						nn9::float16::Convert16Float32ToFloat16( pSrc, mDst );

						pSrc += 16;
						sSize -= 16;
					}
					while ( sSize ) {
						Intrin::scast( _fFunc( (*pSrc) ), (*pSrc) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if constexpr ( nn9::Types::IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					// Decode 8 bfloat16_t's at once for super-fast processing.
					bfloat16_t * pSrc = reinterpret_cast<bfloat16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<uint16_t *>(pSrc) );
						_mm256_store_ps( fTmp, mSrc );

						Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
						Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
						Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
						Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
						Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
						Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
						Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
						Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );

						sSize -= 8;
						__m256 mDst = _mm256_load_ps( fTmp );
						bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pSrc), mDst );
						pSrc += 8;
					}
					while ( sSize ) {
						Intrin::scast( _fFunc( (*pSrc) ), (*pSrc) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
			if constexpr ( nn9::Types::IsFloat16<ValueType>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mVal = nn9::float16::Convert8Float16ToFloat32( pSrc );
						_mm256_store_ps( fTmp, mVal );

						Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
						Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
						Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
						Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
						Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
						Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
						Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
						Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );

						__m256 mDst = _mm256_load_ps( fTmp );
						nn9::float16::Convert8Float32ToFloat16( pSrc, mDst );

						pSrc += 8;
						sSize -= 8;
					}
					while ( sSize ) {
						Intrin::scast( _fFunc( (*pSrc) ), (*pSrc) );
						++pSrc;
						--sSize;
					}
					return _vValues;
				}
			}
#endif	// #ifdef __AVX2__
			for ( auto & aThis : _vValues ) {
				Intrin::scast( _fFunc( aThis ), aThis );
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
			if constexpr ( nn9::Types::IsBFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					// Decode 16 bfloat16_t's at once for super-fast processing.
					const bfloat16_t * pSrc = reinterpret_cast<const bfloat16_t *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( pSrc );
						_mm512_store_ps( fTmp, mSrc );

						if constexpr ( nn9::Types::IsBFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );
							Intrin::scast( _fFunc( fTmp[8] ), fTmp[8] );
							Intrin::scast( _fFunc( fTmp[9] ), fTmp[9] );
							Intrin::scast( _fFunc( fTmp[10] ), fTmp[10] );
							Intrin::scast( _fFunc( fTmp[11] ), fTmp[11] );
							Intrin::scast( _fFunc( fTmp[12] ), fTmp[12] );
							Intrin::scast( _fFunc( fTmp[13] ), fTmp[13] );
							Intrin::scast( _fFunc( fTmp[14] ), fTmp[14] );
							Intrin::scast( _fFunc( fTmp[15] ), fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( nn9::Types::IsFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );
							Intrin::scast( _fFunc( fTmp[8] ), fTmp[8] );
							Intrin::scast( _fFunc( fTmp[9] ), fTmp[9] );
							Intrin::scast( _fFunc( fTmp[10] ), fTmp[10] );
							Intrin::scast( _fFunc( fTmp[11] ), fTmp[11] );
							Intrin::scast( _fFunc( fTmp[12] ), fTmp[12] );
							Intrin::scast( _fFunc( fTmp[13] ), fTmp[13] );
							Intrin::scast( _fFunc( fTmp[14] ), fTmp[14] );
							Intrin::scast( _fFunc( fTmp[15] ), fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							nn9::float16::Convert16Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							Intrin::scast( _fFunc( fTmp[0] ), pDst[0] );
							Intrin::scast( _fFunc( fTmp[1] ), pDst[1] );
							Intrin::scast( _fFunc( fTmp[2] ), pDst[2] );
							Intrin::scast( _fFunc( fTmp[3] ), pDst[3] );
							Intrin::scast( _fFunc( fTmp[4] ), pDst[4] );
							Intrin::scast( _fFunc( fTmp[5] ), pDst[5] );
							Intrin::scast( _fFunc( fTmp[6] ), pDst[6] );
							Intrin::scast( _fFunc( fTmp[7] ), pDst[7] );
							Intrin::scast( _fFunc( fTmp[8] ), pDst[8] );
							Intrin::scast( _fFunc( fTmp[9] ), pDst[9] );
							Intrin::scast( _fFunc( fTmp[10] ), pDst[10] );
							Intrin::scast( _fFunc( fTmp[11] ), pDst[11] );
							Intrin::scast( _fFunc( fTmp[12] ), pDst[12] );
							Intrin::scast( _fFunc( fTmp[13] ), pDst[13] );
							Intrin::scast( _fFunc( fTmp[14] ), pDst[14] );
							Intrin::scast( _fFunc( fTmp[15] ), pDst[15] );
						}

						sSize -= 16;
						pSrc += 16;
						pDst += 16;
					}
					while ( sSize-- ) {
						Intrin::scast( _fFunc( (*pSrc++) ), (*pDst++) );
					}
					return _vOut;
				}
			}
			if constexpr ( nn9::Types::IsFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx512FSupported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					while ( sSize >= 16 ) {
						__m512 mVal = nn9::float16::Convert16Float16ToFloat32( pSrc );
						_mm512_store_ps( fTmp, mVal );

						if constexpr ( nn9::Types::IsBFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );
							Intrin::scast( _fFunc( fTmp[8] ), fTmp[8] );
							Intrin::scast( _fFunc( fTmp[9] ), fTmp[9] );
							Intrin::scast( _fFunc( fTmp[10] ), fTmp[10] );
							Intrin::scast( _fFunc( fTmp[11] ), fTmp[11] );
							Intrin::scast( _fFunc( fTmp[12] ), fTmp[12] );
							Intrin::scast( _fFunc( fTmp[13] ), fTmp[13] );
							Intrin::scast( _fFunc( fTmp[14] ), fTmp[14] );
							Intrin::scast( _fFunc( fTmp[15] ), fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( nn9::Types::IsFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );
							Intrin::scast( _fFunc( fTmp[8] ), fTmp[8] );
							Intrin::scast( _fFunc( fTmp[9] ), fTmp[9] );
							Intrin::scast( _fFunc( fTmp[10] ), fTmp[10] );
							Intrin::scast( _fFunc( fTmp[11] ), fTmp[11] );
							Intrin::scast( _fFunc( fTmp[12] ), fTmp[12] );
							Intrin::scast( _fFunc( fTmp[13] ), fTmp[13] );
							Intrin::scast( _fFunc( fTmp[14] ), fTmp[14] );
							Intrin::scast( _fFunc( fTmp[15] ), fTmp[15] );

							__m512 mDst = _mm512_load_ps( fTmp );
							nn9::float16::Convert16Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							Intrin::scast( _fFunc( fTmp[0] ), pDst[0] );
							Intrin::scast( _fFunc( fTmp[1] ), pDst[1] );
							Intrin::scast( _fFunc( fTmp[2] ), pDst[2] );
							Intrin::scast( _fFunc( fTmp[3] ), pDst[3] );
							Intrin::scast( _fFunc( fTmp[4] ), pDst[4] );
							Intrin::scast( _fFunc( fTmp[5] ), pDst[5] );
							Intrin::scast( _fFunc( fTmp[6] ), pDst[6] );
							Intrin::scast( _fFunc( fTmp[7] ), pDst[7] );
							Intrin::scast( _fFunc( fTmp[8] ), pDst[8] );
							Intrin::scast( _fFunc( fTmp[9] ), pDst[9] );
							Intrin::scast( _fFunc( fTmp[10] ), pDst[10] );
							Intrin::scast( _fFunc( fTmp[11] ), pDst[11] );
							Intrin::scast( _fFunc( fTmp[12] ), pDst[12] );
							Intrin::scast( _fFunc( fTmp[13] ), pDst[13] );
							Intrin::scast( _fFunc( fTmp[14] ), pDst[14] );
							Intrin::scast( _fFunc( fTmp[15] ), pDst[15] );
						}

						pSrc += 16;
						sSize -= 16;
					}
					while ( sSize-- ) {
						Intrin::scast( _fFunc( (*pSrc++) ), (*pDst++) );
					}
					return _vOut;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if constexpr ( nn9::Types::IsBFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					// Decode 8 bfloat16_t's at once for super-fast processing.
					const bfloat16_t * pSrc = reinterpret_cast<const bfloat16_t *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( pSrc );
						_mm256_store_ps( fTmp, mSrc );

						if constexpr ( nn9::Types::IsBFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( nn9::Types::IsFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							nn9::float16::Convert8Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							Intrin::scast( _fFunc( fTmp[0] ), pDst[0] );
							Intrin::scast( _fFunc( fTmp[1] ), pDst[1] );
							Intrin::scast( _fFunc( fTmp[2] ), pDst[2] );
							Intrin::scast( _fFunc( fTmp[3] ), pDst[3] );
							Intrin::scast( _fFunc( fTmp[4] ), pDst[4] );
							Intrin::scast( _fFunc( fTmp[5] ), pDst[5] );
							Intrin::scast( _fFunc( fTmp[6] ), pDst[6] );
							Intrin::scast( _fFunc( fTmp[7] ), pDst[7] );
						}

						sSize -= 8;
						pSrc += 8;
						pDst += 8;
					}
					while ( sSize-- ) {
						Intrin::scast( _fFunc( (*pSrc++) ), (*pDst++) );
					}
					return _vOut;
				}
			}
			if constexpr ( nn9::Types::IsFloat16<ValueTypeIn>() ) {
				if ( Utilities::IsAvx2Supported() ) {
					nn9::float16 * pSrc = reinterpret_cast<nn9::float16 *>(&_vIn[0]);
					ValueTypeOut * pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mVal = nn9::float16::Convert8Float16ToFloat32( pSrc );
						_mm256_store_ps( fTmp, mVal );

						if constexpr ( nn9::Types::IsBFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(pDst), mDst );
						}
						else if constexpr ( nn9::Types::IsFloat16<ValueTypeOut>() ) {
							Intrin::scast( _fFunc( fTmp[0] ), fTmp[0] );
							Intrin::scast( _fFunc( fTmp[1] ), fTmp[1] );
							Intrin::scast( _fFunc( fTmp[2] ), fTmp[2] );
							Intrin::scast( _fFunc( fTmp[3] ), fTmp[3] );
							Intrin::scast( _fFunc( fTmp[4] ), fTmp[4] );
							Intrin::scast( _fFunc( fTmp[5] ), fTmp[5] );
							Intrin::scast( _fFunc( fTmp[6] ), fTmp[6] );
							Intrin::scast( _fFunc( fTmp[7] ), fTmp[7] );

							__m256 mDst = _mm256_load_ps( fTmp );
							nn9::float16::Convert8Float32ToFloat16( reinterpret_cast<nn9::float16 *>(pDst), mDst );
						}
						else {
							Intrin::scast( _fFunc( fTmp[0] ), pDst[0] );
							Intrin::scast( _fFunc( fTmp[1] ), pDst[1] );
							Intrin::scast( _fFunc( fTmp[2] ), pDst[2] );
							Intrin::scast( _fFunc( fTmp[3] ), pDst[3] );
							Intrin::scast( _fFunc( fTmp[4] ), pDst[4] );
							Intrin::scast( _fFunc( fTmp[5] ), pDst[5] );
							Intrin::scast( _fFunc( fTmp[6] ), pDst[6] );
							Intrin::scast( _fFunc( fTmp[7] ), pDst[7] );
						}

						pSrc += 8;
						sSize -= 8;
					}
					while ( sSize-- ) {
						Intrin::scast( _fFunc( (*pSrc++) ), (*pDst++) );
					}
					return _vOut;
				}
			}
#endif	// #ifdef __AVX2__

			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				Intrin::scast( _fFunc( _vIn[i] ), _vOut[i] );
			}
			return _vOut;
		}

#ifdef __AVX512F__
		/**
		 * Applies the given function to each item in the view using AVX-512.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tAvx512Func The AVX-512 function type.
		 * \tparam _tFunc The function type.
		 * \param _vValues The input/output view to modify.
		 * \param _fAvxFunc A pointer to the function to call on each item in the view.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tAvx512Func, typename _tFunc>
		static _tType &												FuncAvx512( _tType &_vValues, _tAvx512Func _fAvxFunc, _tFunc _fFunc ) {
			using ValueType = typename _tType::value_type;
			auto * pvtiIn = &_vValues[0];
			auto sSize = _vValues.size();

			if constexpr ( nn9::Types::SimdInt<ValueType>() ) {
				if constexpr ( nn9::Types::IsInt8<ValueType>() || nn9::Types::IsUint8<ValueType>() ||
					nn9::Types::IsInt16<ValueType>() || nn9::Types::IsUint16<ValueType>() ) {
					if ( !Utilities::IsAvx512BWSupported() ) { goto End; }
				}
				constexpr size_t sRegSize = sizeof( __m512i ) / sizeof( ValueType );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm512_loadu_epi64( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueType>( mReg, pvtiIn );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
				}
				goto End;	// To remove "unused label" warning.
			}
			else if constexpr ( nn9::Types::SimdFloat<ValueType>() ) {
				constexpr size_t sRegSize = sizeof( __m512 ) / sizeof( float );
				while ( sSize >= sRegSize ) {
					__m512 mReg;
					if constexpr ( nn9::Types::IsFloat16<ValueType>() ) {
						mReg = nn9::float16::Convert16Float16ToFloat32( pvtiIn );
					}
					else if constexpr ( nn9::Types::IsBFloat16<ValueType>() ) {
						mReg = nn9::bfloat16::loadu_bf16_to_fp32_16( pvtiIn );
					}
					else if constexpr ( nn9::Types::Is32BitFloat<ValueType>() ) {
						mReg = _mm512_loadu_ps( reinterpret_cast<const float *>(pvtiIn) );
					}
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueType>( mReg, pvtiIn );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
				}
				goto End;	// To remove "unused label" warning.
			}
			else if constexpr ( nn9::Types::SimdDouble<ValueType>() ) {
				constexpr size_t sRegSize = sizeof( __m512d ) / sizeof( ValueType );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm512_loadu_pd( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueType>( mReg, pvtiIn );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
				}
				goto End;	// To remove "unused label" warning.
			}

		End :
			while ( sSize-- ) {
				Intrin::scast( (*pvtiIn), (*pvtiIn) );
				++pvtiIn;
			}
			return _vValues;
		}

		/**
		 * Applies the given function to each item in the view.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tAvx512Func The AVX-512 function type.
		 * \tparam _tFunc The function type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _fAvxFunc A pointer to the function to call on each item in the view.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \throw If NN9_SAFETY_CHECK, throws if the views are not the same lengths.
		 * \return Returns _vOut.
		 **/
		template <typename _tTypeIn, typename _tTypeOut, typename _tAvx512Func, typename _tFunc>
		static _tTypeOut &											FuncAvx512( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tAvx512Func _fAvxFunc, _tFunc _fFunc ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::FuncAvx512: The source and destination must both have the same number of elements." ); }
#endif		// #ifdef NN9_SAFETY_CHECK

			const auto * pvtiIn = &_vIn[0];
			auto * pvtoOut = &_vOut[0];
			auto sSize = _vIn.size();

			if constexpr ( nn9::Types::SimdInt<ValueTypeIn>() ) {
				if constexpr ( nn9::Types::IsInt8<ValueTypeIn>() || nn9::Types::IsUint8<ValueTypeIn>() ||
					nn9::Types::IsInt16<ValueTypeIn>() || nn9::Types::IsUint16<ValueTypeIn>() ) {
					if ( !Utilities::IsAvx512BWSupported() ) { goto End; }
				}
				constexpr size_t sRegSize = sizeof( __m512i ) / sizeof( ValueTypeIn );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm512_loadu_epi8( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueTypeIn>( mReg, pvtoOut );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
					pvtoOut += sRegSize;
				}
				goto End;	// To remove "unused label" warning.
			}
			else if constexpr ( nn9::Types::SimdFloat<ValueTypeIn>() ) {
				constexpr size_t sRegSize = sizeof( __m512 ) / sizeof( float );
				while ( sSize >= sRegSize ) {
					__m512 mReg;
					if constexpr ( nn9::Types::IsFloat16<ValueTypeIn>() ) {
						mReg = nn9::float16::Convert16Float16ToFloat32( pvtiIn );
					}
					else if constexpr ( nn9::Types::IsBFloat16<ValueTypeIn>() ) {
						mReg = nn9::bfloat16::loadu_bf16_to_fp32_16( pvtiIn );
					}
					else if constexpr ( nn9::Types::Is32BitFloat<ValueTypeIn>() ) {
						mReg = _mm512_loadu_ps( reinterpret_cast<const float *>(pvtiIn) );
					}
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueTypeIn>( mReg, pvtoOut );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
					pvtoOut += sRegSize;
				}
				goto End;	// To remove "unused label" warning.
			}
			else if constexpr ( nn9::Types::SimdDouble<ValueTypeIn>() ) {
				constexpr size_t sRegSize = sizeof( __m512d ) / sizeof( ValueTypeIn );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm512_loadu_pd( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueTypeIn>( mReg, pvtoOut );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
					pvtoOut += sRegSize;
				}
				goto End;	// To remove "unused label" warning.
			}

		End :
			while ( sSize-- ) {
				Intrin::scast( (*pvtiIn++), (*pvtoOut++) );
			}
			return _vOut;
		}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * Applies the given function to each item in the view using AVX2.
		 * 
		 * \tparam _tType The view/container type.
		 * \tparam _tAvx2Func The AVX2 function type.
		 * \tparam _tFunc The function type.
		 * \param _vValues The input/output view to modify.
		 * \param _fAvxFunc A pointer to the function to call on each item in the view.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tAvx2Func, typename _tFunc>
		static _tType &												FuncAvx2( _tType &_vValues, _tAvx2Func _fAvxFunc, _tFunc _fFunc ) {
			using ValueType = typename _tType::value_type;
			auto * pvtiIn = &_vValues[0];
			auto sSize = _vValues.size();

			if constexpr ( nn9::Types::SimdInt<ValueType>() ) {
				constexpr size_t sRegSize = sizeof( __m256i ) / sizeof( ValueType );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm256_loadu_epi64( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueType>( mReg, pvtiIn );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
				}
			}
			else if constexpr ( nn9::Types::SimdFloat<ValueType>() ) {
				constexpr size_t sRegSize = sizeof( __m256 ) / sizeof( float );
				while ( sSize >= sRegSize ) {
					__m256 mReg;
					if constexpr ( nn9::Types::IsFloat16<ValueType>() ) {
						mReg = nn9::float16::Convert8Float16ToFloat32( pvtiIn );
					}
					else if constexpr ( nn9::Types::IsBFloat16<ValueType>() ) {
						mReg = nn9::bfloat16::loadu_bf16_to_fp32_8( pvtiIn );
					}
					else if constexpr ( nn9::Types::Is32BitFloat<ValueType>() ) {
						mReg = _mm256_loadu_ps( reinterpret_cast<const float *>(pvtiIn) );
					}
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueType>( mReg, pvtiIn );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
				}
			}
			else if constexpr ( nn9::Types::SimdDouble<ValueType>() ) {
				constexpr size_t sRegSize = sizeof( __m256d ) / sizeof( ValueType );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm256_loadu_pd( reinterpret_cast<const double *>(pvtiIn) );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueType>( mReg, pvtiIn );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
				}
			}

			while ( sSize-- ) {
				Intrin::scast( (*pvtiIn), (*pvtiIn) );
				++pvtiIn;
			}
			return _vValues;
		}

		/**
		 * Applies the given function to each item in the view.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \tparam _tAvx2Func The AVX-2 function type.
		 * \tparam _tFunc The function type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _fAvxFunc A pointer to the function to call on each item in the view.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \throw If NN9_SAFETY_CHECK, throws if the views are not the same lengths.
		 * \return Returns _vOut.
		 **/
		template <typename _tTypeIn, typename _tTypeOut, typename _tAvx2Func, typename _tFunc>
		static _tTypeOut &											FuncAvx2( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tAvx2Func _fAvxFunc, _tFunc _fFunc ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::FuncAvx2: The source and destination must both have the same number of elements." ); }
#endif		// #ifdef NN9_SAFETY_CHECK

			const auto * pvtiIn = &_vIn[0];
			auto * pvtoOut = &_vOut[0];
			auto sSize = _vIn.size();

			if constexpr ( nn9::Types::SimdInt<ValueTypeIn>() ) {
				constexpr size_t sRegSize = sizeof( __m256i ) / sizeof( ValueTypeIn );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm256_loadu_epi8( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueTypeIn>( mReg, pvtoOut );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
					pvtoOut += sRegSize;
				}
			}
			else if constexpr ( nn9::Types::SimdFloat<ValueTypeIn>() ) {
				constexpr size_t sRegSize = sizeof( __m256 ) / sizeof( float );
				while ( sSize >= sRegSize ) {
					__m256 mReg;
					if constexpr ( nn9::Types::IsFloat16<ValueTypeIn>() ) {
						mReg = nn9::float16::Convert8Float16ToFloat32( pvtiIn );
					}
					else if constexpr ( nn9::Types::IsBFloat16<ValueTypeIn>() ) {
						mReg = nn9::bfloat16::loadu_bf16_to_fp32_8( pvtiIn );
					}
					else if constexpr ( nn9::Types::Is32BitFloat<ValueTypeIn>() ) {
						mReg = _mm256_loadu_ps( reinterpret_cast<const float *>(pvtiIn) );
					}
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueTypeIn>( mReg, pvtoOut );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
					pvtoOut += sRegSize;
				}
			}
			else if constexpr ( nn9::Types::SimdDouble<ValueTypeIn>() ) {
				constexpr size_t sRegSize = sizeof( __m256d ) / sizeof( ValueTypeIn );
				while ( sSize >= sRegSize ) {
					auto mReg = _mm256_loadu_pd( pvtiIn );
					mReg = _fAvxFunc( mReg );
					Intrin::scast<ValueTypeIn>( mReg, pvtoOut );
					sSize -= sRegSize;
					pvtiIn += sRegSize;
					pvtoOut += sRegSize;
				}
			}

			while ( sSize-- ) {
				Intrin::scast( (*pvtiIn++), (*pvtoOut++) );
			}
			return _vOut;
		}
#endif	// #ifdef __AVX2__


		// ===============================
		// Trigonometric Functions
		// ===============================
		/**
		 * Computes element-wise cos().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Cos( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// cos( 1 ) = 0.5403023058681397650.
				// cos( 0 ) = 1.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::cos( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Cos( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::cos( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Cos( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// sin( 1 ) = 1.
				// sin( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::sin( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Sin( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::sin( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Sin( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// tan( 1 ) = 1.5574077246549.
				// tan( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::tan( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Tan( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::tan( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Tan( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise acos().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Acos( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// acos( 1 ) = 0.
				// acos( 0 ) = 1.57079632679.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::acos( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Acos( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::acos( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Acos( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// asin( 1 ) = 1.57079632679.
				// asin( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::asin( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Asin( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::asin( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Asin( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// atan( 1 ) = 0.7853981633974.
				// atan( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::atan( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Atan( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::atan( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Atan( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Hyperbolic Functions
		// ===============================
		/**
		 * Computes element-wise cosh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Cosh( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// cosh( 1 ) = 1.54308063481524.
				// cosh( 0 ) = 1.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::cosh( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Cosh( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::cosh( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Cosh( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// sinh( 1 ) = 1.1752011936438.
				// sinh( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::sinh( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Sinh( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::sinh( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Sinh( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// tanh( 1 ) = 0.76159415595576.
				// tanh( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::tanh( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Tanh( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::tanh( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Tanh( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// acosh( 1 ) = 0.
				// acosh( 0 ) = NaN.
				return Func<_tType>( _vValues, [](auto x) { return false; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::acosh( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Acosh( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return false; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::acosh( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Acosh( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// asinh( 1 ) = 0.881373587019543.
				// asinh( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::asinh( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Asinh( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::asinh( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Asinh( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// atanh( 1 ) = inf.
				// atanh( 0 ) = 0.
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via this operation.
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::atanh( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Atanh( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::atanh( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Atanh( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Exponential and Logarithmic Functions
		// ===============================
		/**
		 * Computes element-wise exp().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Exp( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// exp( 1 ) = 2.718281828459.
				// exp( 0 ) = 1.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::exp( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Exp( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::exp( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Exp( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// log( 1 ) = 0.
				// log( 0 ) = -inf.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::log( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Log( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::log( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Log( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// log10( 1 ) = 0.
				// log10( 0 ) = -inf.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::log10( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Log10( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::log10( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Log10( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// log2( 1 ) = 0.
				// log2( 0 ) = -inf.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::log2( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Log2( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::log2( static_cast<double>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Log2( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise exp2().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Exp2( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// exp2( 1 ) = 2.
				// exp2( 0 ) = 1.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::exp2( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Exp2() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Exp2( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Exp2( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise exp2().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Exp2( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<_tTypeIn::value_type>(std::exp2( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Exp2() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Exp2( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Exp2: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Exp2( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// expm1( 1 ) = 1.718281828459.
				// expm1( 0 ) = 0.
				return Func<_tType>( _vValues, [](auto x) { return x; } );
			}
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
			for ( auto & aThis : _vValues ) { Expm1( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Expm1( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise ilogb().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Ilogb( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// ilogb( 1 ) = 0.
				// ilogb( 0 ) = -2147483648.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::ilogb( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Ilogb() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Ilogb( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Ilogb( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise ilogb().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Ilogb( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::ilogb( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Ilogb() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Ilogb( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Ilogb: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Ilogb( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// log1p( 1 ) = 0.693147180559945.
				// log1p( 0 ) = 0.
				return Func<_tType>( _vValues, [](auto x) { return x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::log1p( static_cast<double>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Log1p( aThis ); }
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
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Log1p( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise logb().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Logb( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// logb( 1 ) = 0.
				// logb( 0 ) = -inf.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::logb( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Logb() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Logb( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Logb( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise logb().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Logb( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::logb( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Logb() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Logb( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Logb: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Logb( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Power Functions
		// ===============================
		/**
		 * Computes element-wise x*x.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Square( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via x*x.
			}
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return Intrin::SquareInt8( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return Intrin::SquareUint8( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return Intrin::SquareInt16( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return Intrin::SquareUint16( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return Intrin::SquareInt32( x ); }, [](auto x) { return int64_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return Intrin::SquareUint32( x ); }, [](auto x) { return uint64_t( x ) * x; } );
				}

				if constexpr ( Types::IsInt64<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareInt64( x ); }, [](auto x) {
							constexpr int64_t i64Max = 3037000499ULL;
							int64_t i64Abs = std::abs<int64_t>( x );
							return (x * x * (i64Abs <= i64Max)) | ((i64Abs > i64Max) * INT64_MAX);
						} );
				}
				if constexpr ( Types::IsUint64<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareUint64( x ); }, [](auto x) {
							constexpr uint64_t ui64Max = 4294967295ULL;
							return (x * x * (x <= ui64Max)) | ((x > ui64Max) * UINT64_MAX);
						} );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_mul_ps( x, x ); }, [](auto x) { return x * x; } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_mul_pd( x, x ); }, [](auto x) { return x * x; } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareInt8( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareUint8( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareInt16( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareUint16( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareInt32( x ); }, [](auto x) { return int64_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareUint32( x ); }, [](auto x) { return uint64_t( x ) * x; } );
				}

				if constexpr ( Types::IsInt64<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareInt64( x ); }, [](auto x) {
							constexpr int64_t i64Max = 3037000499ULL;
							int64_t i64Abs = std::abs<int64_t>( x );
							return (x * x * (i64Abs <= i64Max)) | ((i64Abs > i64Max) * INT64_MAX);
						} );
				}
				if constexpr ( Types::IsUint64<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return Intrin::SquareUint64( x ); }, [](auto x) {
							constexpr uint64_t ui64Max = 4294967295ULL;
							return (x * x * (x <= ui64Max)) | ((x > ui64Max) * UINT64_MAX);
						} );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_mul_ps( x, x ); }, [](auto x) { return x * x; } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_mul_pd( x, x ); }, [](auto x) { return x * x; } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt8<Type>() || Types::IsUint8<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return int32_t( x ) * x; } );
			}
			if constexpr ( Types::IsInt16<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return int32_t( x ) * x; } );
			}
			if constexpr ( Types::IsUint16<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return uint32_t( x ) * x; } );
			}
			if constexpr ( Types::IsInt32<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return int64_t( x ) * x; } );
			}
			if constexpr ( Types::IsUint32<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return uint64_t( x ) * x; } );
			}
			if constexpr ( Types::IsInt64<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) {
						constexpr int64_t i64Max = 3037000499ULL;
						int64_t i64Abs = std::abs<int64_t>( x );
						return (x * x * (i64Abs <= i64Max)) | ((i64Abs > i64Max) * INT64_MAX);
					} );
			}
			if constexpr ( Types::IsUint64<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) {
						constexpr uint64_t ui64Max = 4294967295ULL;
						return (x * x * (x <= ui64Max)) | ((x > ui64Max) * UINT64_MAX);
					} );
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
			for ( auto & aThis : _vValues ) { Square( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise x*x.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Square( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt8( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint8( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt16( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint16( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt32( x ); }, [](auto x) { return int64_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint32( x ); }, [](auto x) { return uint64_t( x ) * x; } );
				}

				if constexpr ( Types::IsInt64<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt64( x ); }, [](auto x) {
							constexpr int64_t i64Max = 3037000499ULL;
							int64_t i64Abs = std::abs<int64_t>( x );
							return (x * x * (i64Abs <= i64Max)) | ((i64Abs > i64Max) * INT64_MAX);
						} );
				}
				if constexpr ( Types::IsUint64<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint64( x ); }, [](auto x) {
							constexpr uint64_t ui64Max = 4294967295ULL;
							return (x * x * (x <= ui64Max)) | ((x > ui64Max) * UINT64_MAX);
						} );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_mul_ps( x, x ); }, [](auto x) { return x * x; } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_mul_pd( x, x ); }, [](auto x) { return x * x; } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt8( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint8( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt16( x ); }, [](auto x) { return int32_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint16( x ); }, [](auto x) { return uint32_t( x ) * x; } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt32( x ); }, [](auto x) { return int64_t( x ) * x; } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint32( x ); }, [](auto x) { return uint64_t( x ) * x; } );
				}

				if constexpr ( Types::IsInt64<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareInt64( x ); }, [](auto x) {
							constexpr int64_t i64Max = 3037000499ULL;
							int64_t i64Abs = std::abs<int64_t>( x );
							return (x * x * (i64Abs <= i64Max)) | ((i64Abs > i64Max) * INT64_MAX);
						} );
				}
				if constexpr ( Types::IsUint64<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return Intrin::SquareUint64( x ); }, [](auto x) {
							constexpr uint64_t ui64Max = 4294967295ULL;
							return (x * x * (x <= ui64Max)) | ((x > ui64Max) * UINT64_MAX);
						} );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_mul_ps( x, x ); }, [](auto x) { return x * x; } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_mul_pd( x, x ); }, [](auto x) { return x * x; } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt8<TypeIn>() || Types::IsUint8<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return int32_t( x ) * x; } );
			}
			if constexpr ( Types::IsInt16<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return int32_t( x ) * x; } );
			}
			if constexpr ( Types::IsUint16<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return uint32_t( x ) * x; } );
			}
			if constexpr ( Types::IsInt32<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return int64_t( x ) * x; } );
			}
			if constexpr ( Types::IsUint32<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return uint64_t( x ) * x; } );
			}

			if constexpr ( Types::IsInt64<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
						constexpr int64_t i64Max = 3037000499ULL;
						int64_t i64Abs = std::abs<int64_t>( x );
						return (x * x * (i64Abs <= i64Max)) | ((i64Abs > i64Max) * INT64_MAX);
					} );
			}
			if constexpr ( Types::IsUint64<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
						constexpr uint64_t ui64Max = 4294967295ULL;
						return (x * x * (x <= ui64Max)) | ((x > ui64Max) * UINT64_MAX);
					} );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Square( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				//return Func<_tType>( _vValues, [](auto x) { return x; } );
				return _vValues;	// Assuming well formed bool values, no changes will be made via sqrt().
			}
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 ); m2 = _mm512_sqrt_ps( m2 ); m3 = _mm512_sqrt_ps( m3 );
							return Intrin::float32x64_to_int8x64( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 ); m2 = _mm512_sqrt_ps( m2 ); m3 = _mm512_sqrt_ps( m3 );
							return Intrin::float32x64_to_uint8x64( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_sqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_sqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_sqrt_ps( x ); }, [](auto x) { return static_cast<Type>(std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_sqrt_pd( x ); }, [](auto x) { return static_cast<Type>(std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 ); m2 = _mm256_sqrt_ps( m2 ); m3 = _mm256_sqrt_ps( m3 );
							return Intrin::float32x32_to_int8x32( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 ); m2 = _mm256_sqrt_ps( m2 ); m3 = _mm256_sqrt_ps( m3 );
							return Intrin::float32x32_to_uint8x32( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm256_sqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm256_sqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [](auto x) { return std::sqrt<Type>( x ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_sqrt_ps( x ); }, [](auto x) { return static_cast<Type>(std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_sqrt_pd( x ); }, [](auto x) { return static_cast<Type>(std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return std::sqrt<Type>( x ); } );
			}
			if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return std::sqrt( x ); } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<Type>(std::sqrt( static_cast<float>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Sqrt( aThis ); }
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
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 ); m2 = _mm512_sqrt_ps( m2 ); m3 = _mm512_sqrt_ps( m3 );
							return Intrin::float32x64_to_int8x64( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 ); m2 = _mm512_sqrt_ps( m2 ); m3 = _mm512_sqrt_ps( m3 );
							return Intrin::float32x64_to_uint8x64( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_sqrt_ps( m0 ); m1 = _mm512_sqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_sqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_sqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_sqrt_ps( x ); }, [](auto x) { return static_cast<TypeIn>(std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_sqrt_pd( x ); }, [](auto x) { return static_cast<TypeIn>(std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 ); m2 = _mm256_sqrt_ps( m2 ); m3 = _mm256_sqrt_ps( m3 );
							return Intrin::float32x32_to_int8x32( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 ); m2 = _mm256_sqrt_ps( m2 ); m3 = _mm256_sqrt_ps( m3 );
							return Intrin::float32x32_to_uint8x32( m0, m1, m2, m3 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_sqrt_ps( m0 ); m1 = _mm256_sqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32( m0, m1 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm256_sqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm256_sqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [](auto x) { return std::sqrt<TypeIn>( x ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_sqrt_ps( x ); }, [](auto x) { return static_cast<TypeIn>(std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_sqrt_pd( x ); }, [](auto x) { return static_cast<TypeIn>(std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::sqrt<TypeIn>( x ); } );
			}
			if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::sqrt( x ); } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<TypeIn>(std::sqrt( static_cast<float>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Sqrt( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// rsqrt( 0 ): +inf
				// rsqrt( 1 ): 1
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 ); m2 = _mm512_rsqrt_ps( m2 ); m3 = _mm512_rsqrt_ps( m3 );
							return Intrin::float32x64_to_int8x64_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 ); m2 = _mm512_rsqrt_ps( m2 ); m3 = _mm512_rsqrt_ps( m3 );
							return Intrin::float32x64_to_uint8x64_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_rsqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_rsqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_rsqrt_ps( x ); }, [](auto x) { return static_cast<Type>(1.0f / std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_rsqrt_pd( x ); }, [](auto x) { return static_cast<Type>(1.0 / std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 ); m2 = _mm256_rsqrt_ps( m2 ); m3 = _mm256_rsqrt_ps( m3 );
							return Intrin::float32x32_to_int8x32_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 ); m2 = _mm256_rsqrt_ps( m2 ); m3 = _mm256_rsqrt_ps( m3 );
							return Intrin::float32x32_to_uint8x32_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16_saturated( x, m0 );
							m0 = _mm256_rsqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16_saturated( x, m0 );
							m0 = _mm256_rsqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_rsqrt_ps( x ); }, [](auto x) { return static_cast<Type>(1.0f / std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_rsqrt_pd( x ); }, [](auto x) { return static_cast<Type>(1.0 / std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return 1.0f / std::sqrt<Type>( x ); } );
			}
			if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return 1.0 / std::sqrt( x ); } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<Type>(1.0f / std::sqrt( static_cast<float>(x) )); } );
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
			for ( auto & aThis : _vValues ) { Rsqrt( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise 1/sqrt().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Rsqrt( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 ); m2 = _mm512_rsqrt_ps( m2 ); m3 = _mm512_rsqrt_ps( m3 );
							return Intrin::float32x64_to_int8x64_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 ); m2 = _mm512_rsqrt_ps( m2 ); m3 = _mm512_rsqrt_ps( m3 );
							return Intrin::float32x64_to_uint8x64_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_rsqrt_ps( m0 ); m1 = _mm512_rsqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_rsqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_rsqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_rsqrt_ps( x ); }, [](auto x) { return static_cast<TypeIn>(1.0f / std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_rsqrt_pd( x ); }, [](auto x) { return static_cast<TypeIn>(1.0 / std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 ); m2 = _mm256_rsqrt_ps( m2 ); m3 = _mm256_rsqrt_ps( m3 );
							return Intrin::float32x32_to_int8x32_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 ); m2 = _mm256_rsqrt_ps( m2 ); m3 = _mm256_rsqrt_ps( m3 );
							return Intrin::float32x32_to_uint8x32_saturated( m0, m1, m2, m3 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_rsqrt_ps( m0 ); m1 = _mm256_rsqrt_ps( m1 );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm256_rsqrt_ps( m0 );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm256_rsqrt_ps( m0 );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_rsqrt_ps( x ); }, [](auto x) { return static_cast<TypeIn>(1.0f / std::sqrt( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_rsqrt_pd( x ); }, [](auto x) { return static_cast<TypeIn>(1.0 / std::sqrt( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return 1.0f / std::sqrt<TypeIn>( x ); } );
			}
			if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return 1.0 / std::sqrt( x ); } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<TypeIn>(1.0f / std::sqrt( static_cast<float>(x) )); } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Rsqrt( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise cbrt().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Cbrt( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// cbrt( 1 ) = 1.
				// cbrt( 0 ) = 0.
				return Func<_tType>( _vValues, [](auto x) { return x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::cbrt( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Cbrt() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Cbrt( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Cbrt( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise cbrt().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Cbrt( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::cbrt( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Cbrt() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Cbrt( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Cbrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Cbrt( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Error and Gamma Functions
		// ===============================
		/**
		 * Computes element-wise erf().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Erf( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// erf( 1 ) = 0.84270069.
				// erf( 0 ) = 0.
				return Func<_tType>( _vValues, [](auto x) { return x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::erf( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Erf() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Erf( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Erf( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise erf().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Erf( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::erf( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Erf() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Erf( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Erf: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Erf( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise nn9::Erfinv().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Erfinv( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// nn9::Erfinv( 1 ) = NaN.
				// nn9::Erfinv( 0 ) = 0.
				return Func<_tType>( _vValues, [](auto x) { return x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(nn9::Erfinv( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Erfinv() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Erfinv( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Erfinv( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise nn9::Erfinv().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Erfinv( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return nn9::Erfinv( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Erfinv() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Erfinv( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Erfinv: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Erfinv( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise erfc().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Erfc( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// erfc( 1 ) = 0.15729920705028511.
				// erfc( 0 ) = 1.0.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::erfc( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Erfc() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Erfc( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Erfc( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise erfc().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Erfc( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::erfc( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Erfc() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Erfc( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Erfc: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Erfc( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise lgamma().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Lgamma( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// lgamma( 1 ) = 0.
				// lgamma( 0 ) = +inf.
				return Func<_tType>( _vValues, [](auto x) { return !x; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::lgamma( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Lgamma() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Lgamma( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Lgamma( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise lgamma().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Lgamma( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return !x; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::lgamma( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Lgamma() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Lgamma( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Lgamma: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Lgamma( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise digamma().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Digamma( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// digamma( 1 ) = -0.57721566494246179.
				// digamma( 0 ) = +inf.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(nn9::digamma( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Digamma() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Digamma( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Digamma( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise digamma().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Digamma( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return nn9::digamma( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Digamma() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Digamma( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Digamma: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Digamma( _vIn[i], _vOut[i] ); }
			return _vOut;
		}

		/**
		 * Computes element-wise tgamma().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Tgamma( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() ) {
				// tgamma( 1 ) = 1.0.
				// tgamma( 0 ) = +inf.
				return Func<_tType>( _vValues, [](auto x) { return true; } );
			}
			return Func<_tType>( _vValues, [](auto x) { return static_cast<_tType::value_type>(std::tgamma( static_cast<double>(x) )); } );
		}

		/**
		 * Applies Tgamma() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType>
		static std::vector<_tType> &								Tgamma( std::vector<_tType> &_vValues ) {
			for ( auto & aThis : _vValues ) { Tgamma( aThis ); }
			return _vValues;
		}

		/**
		 * Computes element-wise tgamma().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											Tgamma( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			if constexpr ( Types::IsBool<_tTypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return true; } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::tgamma( static_cast<double>(x) ); } );
		}

		/**
		 * Applies Tgamma() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut>
		static std::vector<_tTypeOut> &								Tgamma( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Tgamma: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Tgamma( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Rounding and Remainder Functions
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
			using Type = typename _tType::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_ceil_ps( x ); }, [](auto x) { return static_cast<Type>(std::ceil( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_ceil_pd( x ); }, [](auto x) { return static_cast<Type>(std::ceil( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_ceil_ps( x ); }, [](auto x) { return static_cast<Type>(std::ceil( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_ceil_pd( x ); }, [](auto x) { return static_cast<Type>(std::ceil( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			return _vValues;
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
			for ( auto & aThis : _vValues ) { Ceil( aThis ); }
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
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_ceil_ps( x ); }, [](auto x) { return static_cast<TypeIn>(std::ceil( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_ceil_pd( x ); }, [](auto x) { return static_cast<TypeIn>(std::ceil( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_ceil_ps( x ); }, [](auto x) { return static_cast<TypeIn>(std::ceil( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_ceil_pd( x ); }, [](auto x) { return static_cast<TypeIn>(std::ceil( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Ceil( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_floor_ps( x ); }, [](auto x) { return static_cast<Type>(std::floor( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_floor_pd( x ); }, [](auto x) { return static_cast<Type>(std::floor( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_floor_ps( x ); }, [](auto x) { return static_cast<Type>(std::floor( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_floor_pd( x ); }, [](auto x) { return static_cast<Type>(std::floor( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			return _vValues;
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
			for ( auto & aThis : _vValues ) { Floor( aThis ); }
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
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_floor_ps( x ); }, [](auto x) { return static_cast<TypeIn>(std::floor( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_floor_pd( x ); }, [](auto x) { return static_cast<TypeIn>(std::floor( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_floor_ps( x ); }, [](auto x) { return static_cast<TypeIn>(std::floor( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_floor_pd( x ); }, [](auto x) { return static_cast<TypeIn>(std::floor( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Floor( _vIn[i], _vOut[i] ); }
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
			using Type = typename _tType::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_roundscale_ps( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<Type>(std::round( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_roundscale_pd( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<Type>(std::round( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_round_ps( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<Type>(std::round( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_round_pd( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<Type>(std::round( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			return _vValues;
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
			for ( auto & aThis : _vValues ) { Round( aThis ); }
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
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_roundscale_ps( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<TypeIn>(std::round( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_roundscale_pd( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<TypeIn>(std::round( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_round_ps( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<TypeIn>(std::round( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_round_pd( x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ); }, [](auto x) { return static_cast<TypeIn>(std::round( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Round( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Other Functions
		// ===============================
		/**
		 * Computes element-wise abs().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Abs( _tType &_vValues ) {
			using Type = typename _tType::value_type;
			if constexpr ( Types::IsBool<Type>() || Types::IsUnsigned<Type>() ) {
				return _vValues;
			}
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_abs_epi8( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_abs_epi16( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_abs_epi32( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_abs_ps( x ); }, [](auto x) { return static_cast<float>(std::abs( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) { return _mm512_abs_pd( x ); }, [](auto x) { return static_cast<double>(std::abs( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_abs_epi8( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_abs_epi16( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_abs_epi32( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_abs_ps( x ); }, [](auto x) { return static_cast<float>(std::abs( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) { return _mm256_abs_pd( x ); }, [](auto x) { return static_cast<double>(std::abs( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt64<Type>() || Types::IsInt32<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return std::abs<double>( x ); } );
			}
			else if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return std::abs<double>( x ); } );
			}
			else if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return std::abs<double>( x ); } );
			}
			else {
				return Func<_tType>( _vValues, [](auto x) { return static_cast<float>(std::abs( static_cast<float>(x) )); } );
			}
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
			for ( auto & aThis : _vValues ) { Abs( aThis ); }
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
			using TypeIn = typename _tTypeIn::value_type;
			if constexpr ( Types::IsBool<TypeIn>() || Types::IsUnsigned<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x; } );
			}
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_abs_epi8( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_abs_epi16( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_abs_epi32( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_abs_ps( x ); }, [](auto x) { return static_cast<float>(std::abs( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm512_abs_pd( x ); }, [](auto x) { return static_cast<double>(std::abs( x )); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_abs_epi8( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_abs_epi16( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_abs_epi32( x );
						}, [](auto x) { return std::abs<double>( x ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_abs_ps( x ); }, [](auto x) { return static_cast<float>(std::abs( static_cast<float>(x) )); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return _mm256_abs_pd( x ); }, [](auto x) { return static_cast<double>(std::abs( x )); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt64<TypeIn>() || Types::IsInt32<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::abs<double>( x ); } );
			}
			else if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::abs<double>( x ); } );
			}
			else if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return std::abs<double>( x ); } );
			}
			else {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<float>(std::abs( static_cast<float>(x) )); } );
			}
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

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Abs( _vIn[i], _vOut[i] ); }
			return _vOut;
		}


		// ===============================
		// Arithmetic Functions
		// ===============================
		/**
		 * Computes element-wise x+s.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Add( _tType &_vValues, _tScalarType _stScalar ) {
			using Type = typename _tType::value_type;
			
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_adds_epi8( x, _mm512_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_adds_epu8( x, _mm512_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_adds_epi16( x, _mm512_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return _mm512_adds_epi16( x, _mm512_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return Intrin::_mm512_adds_epi32( x, _mm512_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [](auto x) {
							return Intrin::_mm512_adds_epu32( x, _mm512_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_add_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [](auto x) { return x + static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_add_pd( x, _mm512_set1_pd( static_cast<Type>(_stScalar) ) ); }, [](auto x) { return x + static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_adds_epi8( x, _mm256_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_adds_epu8( x, _mm256_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_adds_epi16( x, _mm256_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return _mm256_adds_epi16( x, _mm256_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return Intrin::_mm256_adds_epi32( x, _mm256_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [](auto x) {
							return Intrin::_mm256_adds_epu32( x, _mm256_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_add_ps( x, _mm256_set1_ps( static_cast<float>(_stScalar) ) ); }, [](auto x) { return x + static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_add_pd( x, _mm256_set1_pd( static_cast<Type>(_stScalar) ) ); }, [](auto x) { return x + static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsBool<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return static_cast<int16_t>( x ) + static_cast<int16_t>(_stScalar); } );
			}
			if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return nn9::adds( x, static_cast<Type>(_stScalar) ); } );
			}
			if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [](auto x) { return x + static_cast<Type>(_stScalar); } );
			}
			return Func<_tType>( _vValues, [](auto x) { return x + static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Add() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Add( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) { Add( aThis, _stScalar ); }
			return _vValues;
		}

		/**
		 * Computes element-wise x+s.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Add( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_adds_epi8( x, _mm512_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_adds_epu8( x, _mm512_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_adds_epi16( x, _mm512_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm512_adds_epi16( x, _mm512_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return Intrin::_mm512_adds_epi32( x, _mm512_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return Intrin::_mm512_adds_epu32( x, _mm512_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_add_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [](auto x) { return x + static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_add_pd( x, _mm512_set1_pd( static_cast<TypeIn>(_stScalar) ) ); }, [](auto x) { return x + static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_adds_epi8( x, _mm256_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_adds_epu8( x, _mm256_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_adds_epi16( x, _mm256_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return _mm256_adds_epi16( x, _mm256_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return Intrin::_mm256_adds_epi32( x, _mm256_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) {
							return Intrin::_mm256_adds_epu32( x, _mm256_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_add_ps( x, _mm256_set1_ps( static_cast<float>(_stScalar) ) ); }, [](auto x) { return x + static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_add_pd( x, _mm256_set1_pd( static_cast<TypeIn>(_stScalar) ) ); }, [](auto x) { return x + static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsBool<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return static_cast<int16_t>( x ) + static_cast<int16_t>(_stScalar); } );
			}
			if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return nn9::adds( x, static_cast<TypeIn>(_stScalar) ); } );
			}
			if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x + static_cast<TypeIn>(_stScalar); } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [](auto x) { return x + static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Add() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Add( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Add: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Add( _vIn[i], _vOut[i], _stScalar ); }
			return _vOut;
		}

		/**
		 * Computes element-wise x-s.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Sub( _tType &_vValues, _tScalarType _stScalar ) {
			using Type = typename _tType::value_type;
			
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							return _mm512_subs_epi8( x, _mm512_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							return _mm512_subs_epu8( x, _mm512_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							return _mm512_subs_epi16( x, _mm512_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							return _mm512_subs_epu16( x, _mm512_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							return Intrin::_mm512_subs_epi32( x, _mm512_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							return Intrin::_mm512_subs_epu32( x, _mm512_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_sub_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x + static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_sub_pd( x, _mm512_set1_pd( static_cast<Type>(_stScalar) ) ); }, [_stScalar](auto x) { return x + static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							return _mm256_subs_epi8( x, _mm256_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							return _mm256_subs_epu8( x, _mm256_set1_epi8( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							return _mm256_subs_epi16( x, _mm256_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							return _mm256_subs_epi16( x, _mm256_set1_epi16( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							return Intrin::_mm256_subs_epi32( x, _mm256_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							return Intrin::_mm256_subs_epu32( x, _mm256_set1_epi32( static_cast<Type>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_sub_ps( x, _mm256_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x + static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_sub_pd( x, _mm256_set1_pd( static_cast<Type>(_stScalar) ) ); }, [_stScalar](auto x) { return x + static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsBool<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<int16_t>( x ) - static_cast<int16_t>(_stScalar); } );
			}
			if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return nn9::subs( x, static_cast<Type>(_stScalar) ); } );
			}
			if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return x - static_cast<Type>(_stScalar); } );
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return x - static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Sub() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Sub( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) { Sub( aThis, _stScalar ); }
			return _vValues;
		}

		/**
		 * Computes element-wise x-s.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Sub( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm512_subs_epi8( x, _mm512_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm512_subs_epu8( x, _mm512_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm512_subs_epi16( x, _mm512_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm512_subs_epi16( x, _mm512_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return Intrin::_mm512_subs_epi32( x, _mm512_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return Intrin::_mm512_subs_epu32( x, _mm512_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_sub_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x - static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_sub_pd( x, _mm512_set1_pd( static_cast<TypeIn>(_stScalar) ) ); }, [_stScalar](auto x) { return x - static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm256_subs_epi8( x, _mm256_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm256_subs_epu8( x, _mm256_set1_epi8( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm256_subs_epi16( x, _mm256_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return _mm256_subs_epi16( x, _mm256_set1_epi16( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return Intrin::_mm256_subs_epi32( x, _mm256_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							return Intrin::_mm256_subs_epu32( x, _mm256_set1_epi32( static_cast<TypeIn>(_stScalar) ) );
						}, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_sub_ps( x, _mm256_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x - static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_sub_pd( x, _mm256_set1_pd( static_cast<TypeIn>(_stScalar) ) ); }, [_stScalar](auto x) { return x - static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsBool<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<int16_t>( x ) - static_cast<int16_t>(_stScalar); } );
			}
			if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return nn9::subs( x, static_cast<TypeIn>(_stScalar) ); } );
			}
			if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return x - static_cast<TypeIn>(_stScalar); } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return x - static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Sub() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Sub( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sub: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Sub( _vIn[i], _vOut[i], _stScalar ); }
			return _vOut;
		}

		/**
		 * Computes element-wise x*s().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Mul( _tType &_vValues, _tScalarType _stScalar ) {
			using Type = typename _tType::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_int8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_uint8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_mul_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_mul_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16_saturated( x, m0 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16_saturated( x, m0 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_mul_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_mul_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
			}
			if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Mul() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Mul( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) { Mul( aThis, _stScalar ); }
			return _vValues;
		}

		/**
		 * Computes element-wise x*s().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Mul( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_int8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_uint8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_mul_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_mul_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_mul_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_mul_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_mul_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm256_mul_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_mul_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_mul_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x * static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
			}
			if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<double>(x) * static_cast<double>(_stScalar); } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<float>(x) * static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Mul() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Mul( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Mul: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Mul( _vIn[i], _vOut[i], _stScalar ); }
			return _vOut;
		}

		/**
		 * Computes element-wise x/s().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vValues The input/output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType, typename _tScalarType>
		static _tType &												Div( _tType &_vValues, _tScalarType _stScalar ) {
			using Type = typename _tType::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_int8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_uint8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_div_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx512<_tType>( _vValues, [_stScalar](auto x) { return _mm512_div_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16_saturated( x, m0 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16_saturated( x, m0 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<Type>() || Types::IsBFloat16<Type>() || Types::Is32BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_div_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<Type>() ) {
					return FuncAvx2<_tType>( _vValues, [_stScalar](auto x) { return _mm256_div_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<Type>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
			}
			if constexpr ( Types::Is64BitFloat<Type>() ) {
				return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
			}
			return Func<_tType>( _vValues, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Div() to an array of inputs.
		 * 
		 * \param _tType The view/container type.
		 * \param _vValues The input/output view to modify.
		 * \return Returns _vValues.
		 **/
		template <typename _tType, typename _tScalarType>
		static std::vector<_tType> &								Div( std::vector<_tType> &_vValues, _tScalarType _stScalar ) {
			for ( auto & aThis : _vValues ) { Div( aThis, _stScalar ); }
			return _vValues;
		}

		/**
		 * Computes element-wise x/s().
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static _tTypeOut &											Div( const _tTypeIn &_vIn, _tTypeOut &_vOut, _tScalarType _stScalar ) {
			using TypeIn = typename _tTypeIn::value_type;
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::int8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_int8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1, m2, m3;
							Intrin::uint8x64_to_float32x64( x, m0, m1, m2, m3 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm512_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm512_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x64_to_uint8x64_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm512_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m512 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm512_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_div_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx512<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm512_div_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				if constexpr ( Types::IsInt8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::int8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint8<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1, m2, m3;
							Intrin::uint8x32_to_float32x32( x, m0, m1, m2, m3 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m2 = _mm256_div_ps( m2, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m3 = _mm256_div_ps( m3, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint8x32_saturated( m0, m1, m2, m3 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::int16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_int16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsUint16<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0, m1;
							Intrin::uint16x32_to_float32x32( x, m0, m1 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); m1 = _mm256_div_ps( m1, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x32_to_uint16x32_saturated( m0, m1 );
						}, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::IsInt32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0;
							Intrin::int32x16_to_float32x16( x, m0 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_int32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}
				if constexpr ( Types::IsUint32<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) {
							__m256 m0;
							Intrin::uint32x16_to_float32x16( x, m0 );
							m0 = _mm256_div_ps( m0, _mm512_set1_ps( static_cast<float>(_stScalar) ) );
							return Intrin::float32x16_to_uint32x16_saturated( m0 );
						}, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
				}

				if constexpr ( Types::IsFloat16<TypeIn>() || Types::IsBFloat16<TypeIn>() || Types::Is32BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_div_ps( x, _mm512_set1_ps( static_cast<float>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<float>(_stScalar); } );
				}
				if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
					return FuncAvx2<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return _mm256_div_pd( x, _mm512_set1_pd( static_cast<double>(_stScalar) ) ); }, [_stScalar](auto x) { return x / static_cast<TypeIn>(_stScalar); } );
				}
			}
#endif	// #ifdef __AVX2__
			if constexpr ( Types::IsInt<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
			}
			if constexpr ( Types::Is64BitFloat<TypeIn>() ) {
				return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<double>(x) / static_cast<double>(_stScalar); } );
			}
			return Func<_tTypeIn, _tTypeOut>( _vIn, _vOut, [_stScalar](auto x) { return static_cast<float>(x) / static_cast<float>(_stScalar); } );
		}

		/**
		 * Applies Div() to an array of inputs and outputs.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \throw If NN9_SAFETY_CHECK, throws if _vIn and _vOut are not the same lengths.
		 * \return Returns _vOut.
		 */
		template <typename _tTypeIn, typename _tTypeOut, typename _tScalarType>
		static std::vector<_tTypeOut> &								Div( const std::vector<_tTypeIn> &_vIn, std::vector<_tTypeOut> &_vOut, _tScalarType _stScalar ) {
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Div: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK

			for ( size_t i = 0; i < _vIn.size(); ++i ) { Div( _vIn[i], _vOut[i], _stScalar ); }
			return _vOut;
		}
	};

}	// namespace nn9

/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Math functions.
 */

#pragma once

#include "../Types/NN9BFloat16.h"
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
					bfloat16_t * _pSrc = reinterpret_cast<bfloat16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 64 )
					float fTmp[16];
					// Alignment.
					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<uint16_t *>(_pSrc) );
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
						bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pSrc), mDst );
						_pSrc += 16;
					}

					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(_pSrc)) = _tType::value_type( _fFunc( (*reinterpret_cast<bfloat16_t *>(_pSrc)) ) );
						++_pSrc;
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
					bfloat16_t * _pSrc = reinterpret_cast<bfloat16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 32 )
					float fTmp[8];

					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<uint16_t *>(_pSrc) );
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
						bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pSrc), mDst );
						_pSrc += 8;
					}
					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(_pSrc)) = _tType::value_type( _fFunc( (*reinterpret_cast<bfloat16_t *>(_pSrc)) ) );
						++_pSrc;
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
					const uint16_t * _pSrc = reinterpret_cast<const uint16_t *>(&_vIn[0]);
					ValueTypeOut * _pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 64 )
					float fTmp[16];

					// Alignment.
					while ( sSize >= 16 ) {
						__m512 mSrc = bfloat16::loadu_bf16_to_fp32_16( _pSrc );
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
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), mDst );
						}
						else {
							_pDst[0] = ValueTypeOut( _fFunc( fTmp[0] ) );
							_pDst[1] = ValueTypeOut( _fFunc( fTmp[1] ) );
							_pDst[2] = ValueTypeOut( _fFunc( fTmp[2] ) );
							_pDst[3] = ValueTypeOut( _fFunc( fTmp[3] ) );
							_pDst[4] = ValueTypeOut( _fFunc( fTmp[4] ) );
							_pDst[5] = ValueTypeOut( _fFunc( fTmp[5] ) );
							_pDst[6] = ValueTypeOut( _fFunc( fTmp[6] ) );
							_pDst[7] = ValueTypeOut( _fFunc( fTmp[7] ) );
							_pDst[8] = ValueTypeOut( _fFunc( fTmp[8] ) );
							_pDst[9] = ValueTypeOut( _fFunc( fTmp[9] ) );
							_pDst[10] = ValueTypeOut( _fFunc( fTmp[10] ) );
							_pDst[11] = ValueTypeOut( _fFunc( fTmp[11] ) );
							_pDst[12] = ValueTypeOut( _fFunc( fTmp[12] ) );
							_pDst[13] = ValueTypeOut( _fFunc( fTmp[13] ) );
							_pDst[14] = ValueTypeOut( _fFunc( fTmp[14] ) );
							_pDst[15] = ValueTypeOut( _fFunc( fTmp[15] ) );
						}

						sSize -= 16;
						_pSrc += 16;
						_pDst += 16;
					}

					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(_pDst)) = _tTypeIn::value_type( _fFunc( (*reinterpret_cast<const bfloat16_t *>(_pSrc)) ) );
						++_pSrc;
						++_pDst;
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
					const uint16_t * _pSrc = reinterpret_cast<const uint16_t *>(&_vIn[0]);
					ValueTypeOut * _pDst = reinterpret_cast<ValueTypeOut *>(&_vOut[0]);
					size_t sSize = _vIn.size();
					NN9_ALIGN( 32 )
					float fTmp[8];
					
					// Alignment.
					while ( sSize >= 8 ) {
						__m256 mSrc = bfloat16::loadu_bf16_to_fp32_8( _pSrc );
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
							bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), mDst );
						}
						else {
							_pDst[0] = ValueTypeOut( _fFunc( fTmp[0] ) );
							_pDst[1] = ValueTypeOut( _fFunc( fTmp[1] ) );
							_pDst[2] = ValueTypeOut( _fFunc( fTmp[2] ) );
							_pDst[3] = ValueTypeOut( _fFunc( fTmp[3] ) );
							_pDst[4] = ValueTypeOut( _fFunc( fTmp[4] ) );
							_pDst[5] = ValueTypeOut( _fFunc( fTmp[5] ) );
							_pDst[6] = ValueTypeOut( _fFunc( fTmp[6] ) );
							_pDst[7] = ValueTypeOut( _fFunc( fTmp[7] ) );
						}

						sSize -= 8;
						_pSrc += 8;
						_pDst += 8;
					}

					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(_pDst)) = _tTypeIn::value_type( _fFunc( (*reinterpret_cast<const bfloat16_t *>(_pSrc)) ) );
						++_pSrc;
						++_pDst;
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
		 * Applies element-wise sqrt() to the input.
		 * 
		 * \param _pfInOut The array of floats to sqrt() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _tTypeIn The input type.  Must be float or bfloat16_t.
		 * \param _tTypeOut The output type.  Must be float or bfloat16_t.
		 * \param _pfIn The array of floats/bfloat16_t's to sqrt().
		 * \param _pfOut The output array of floats/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_sqrt_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_sqrt_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
		 * \param _pdInOut The array of floats to sqrt() in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
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
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _pfInOut The array of floats to 1/sqrt() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _tTypeIn The input type.  Must be float or bfloat16_t.
		 * \param _tTypeOut The output type.  Must be float or bfloat16_t.
		 * \param _pfIn The array of floats/bfloat16_t's to 1/sqrt().
		 * \param _pfOut The output array of floats/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( mVal ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_div_ps( _mm256_set1_ps( 1.0f ), _mm256_sqrt_ps( mVal ) );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
		 * \param _pdInOut The array of floats to 1/sqrt() in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
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
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _pfInOut The array of floats to x*x in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _tTypeIn The input type.  Must be float or bfloat16_t.
		 * \param _tTypeOut The output type.  Must be float or bfloat16_t.
		 * \param _pfIn The array of floats/bfloat16_t's to x*x.
		 * \param _pfOut The output array of floats/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_mul_ps( mVal, mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_mul_ps( mVal, mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
		 * \param _pdInOut The array of floats to x*x in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
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
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _pfInOut The array of floats to ceil() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _tTypeIn The input type.  Must be float or bfloat16_t.
		 * \param _tTypeOut The output type.  Must be float or bfloat16_t.
		 * \param _pfIn The array of floats/bfloat16_t's to ceil().
		 * \param _pfOut The output array of floats/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_ceil_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_ceil_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
		 * \param _pdInOut The array of floats to ceil() in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
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
		 * Applies element-wise floor() to the input.
		 * 
		 * \param _pfInOut The array of floats to floor() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _tTypeIn The input type.  Must be float or bfloat16_t.
		 * \param _tTypeOut The output type.  Must be float or bfloat16_t.
		 * \param _pfIn The array of floats/bfloat16_t's to floor().
		 * \param _pfOut The output array of floats/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_floor_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_floor_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
		 * \param _pdInOut The array of floats to floor() in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
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
		 * Applies element-wise trunc() to the input.
		 * 
		 * \param _pfInOut The array of floats to trunc() in-place.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
		 * \param _tTypeIn The input type.  Must be float or bfloat16_t.
		 * \param _tTypeOut The output type.  Must be float or bfloat16_t.
		 * \param _pfIn The array of floats/bfloat16_t's to trunc().
		 * \param _pfOut The output array of floats/bfloat16_t's.
		 * \param _sSize The total number of floats to which _pfInOut points.
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
					else {
						mVal = _mm512_loadu_ps( _pfIn );
					}
					mVal = _mm512_trunc_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
					else {
						mVal = _mm256_loadu_ps( _pfIn );
					}
					mVal = _mm256_trunc_ps( mVal );
					if constexpr ( IsBFloat16<_tTypeOut>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pfOut), mVal );
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
		 * \param _pdInOut The array of floats to trunc() in-place.
		 * \param _sSize The total number of floats to which _pdInOut points.
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


			if constexpr ( sizeof( _tType::value_type ) == 1 ) {
				size_t I = 0;
				uint8_t * pui8Data = reinterpret_cast<uint8_t *>(&_vValues[0]);
				if ( _vValues.size() >= sizeof( uint64_t ) ) {
					const size_t sTotal = _vValues.size() - sizeof( uint64_t );
					
					while ( I <= sTotal ) {
						(*reinterpret_cast<uint64_t *>(&pui8Data[I])) &= 0x7F7F7F7F7F7F7F7FULL;
						I += sizeof( uint64_t );
					}
				}
				if ( _vValues.size() >= sizeof( uint32_t ) ) {
					const size_t sTotal = _vValues.size() - sizeof( uint32_t );

					while ( I <= sTotal ) {
						(*reinterpret_cast<uint32_t *>(&pui8Data[I])) &= 0x7F7F7F7F;
						I += sizeof( uint32_t );
					}
				}
				if ( _vValues.size() >= sizeof( uint16_t ) ) {
					const size_t sTotal = _vValues.size() - sizeof( uint16_t );
					
					while ( I <= sTotal ) {
						(*reinterpret_cast<uint16_t *>(&pui8Data[I])) &= 0x7F7F;
						I += sizeof( uint16_t );
					}
				}
				const size_t sTotal = _vValues.size();
				while ( I < sTotal ) {
					pui8Data[I] &= 0x7F;
				}
				return _vValues;
			}


			if constexpr ( sizeof( _tType::value_type ) == 2 ) {
				size_t I = 0;
				uint16_t * pui16Data = reinterpret_cast<uint16_t *>(&_vValues[0]);
				if ( _vValues.size() >= sizeof( uint64_t ) / 2 ) {
					const size_t sTotal = _vValues.size() - sizeof( uint64_t ) / 2;
					
					while ( I <= sTotal ) {
						(*reinterpret_cast<uint64_t *>(&pui16Data[I])) &= 0x7FFF7FFF7FFF7FFFULL;
						I += sizeof( uint64_t ) / 2;
					}
				}
				if ( _vValues.size() >= sizeof( uint32_t ) / 2 ) {
					const size_t sTotal = _vValues.size() - sizeof( uint32_t ) / 2;

					while ( I <= sTotal ) {
						(*reinterpret_cast<uint32_t *>(&pui16Data[I])) &= 0x7FFF7FFF;
						I += sizeof( uint32_t ) / 2;
					}
				}
				const size_t sTotal = _vValues.size();
				while ( I < sTotal ) {
					pui16Data[I] &= 0x7FFF;
					++I;
				}
				return _vValues;
			}


			if constexpr ( sizeof( _tType::value_type ) == 4 ) {
				size_t I = 0;
				uint32_t * pui32Data = reinterpret_cast<uint32_t *>(&_vValues[0]);
				if ( _vValues.size() >= sizeof( uint64_t ) / 4 ) {
					const size_t sTotal = _vValues.size() - sizeof( uint64_t ) / 4;
					
					while ( I <= sTotal ) {
						(*reinterpret_cast<uint64_t *>(&pui32Data[I])) &= 0x7FFFFFFF7FFFFFFFULL;
						I += sizeof( uint64_t ) / 4;
					}
				}
				const size_t sTotal = _vValues.size();
				while ( I < sTotal ) {
					pui32Data[I] &= 0x7FFFFFFF;
					++I;
				}
				return _vValues;
			}


			if constexpr ( sizeof( _tType::value_type ) == 8 ) {
				size_t I = 0;
				uint64_t * pui64Data = reinterpret_cast<uint64_t *>(&_vValues[0]);
				const size_t sTotal = _vValues.size();
				while ( I < sTotal ) {
					pui64Data[I] &= 0x7FFFFFFF7FFFFFFFULL;
					++I;
				}
				return _vValues;
			}


			for ( std::size_t i = 0; i < _vValues.size(); ++i ) {
				_vValues[i] = _tType::value_type( std::fabs( _vValues[i] ) );
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
		// Square/Sqrt/Rsqrt
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
				(IsBFloat16<ValueTypeOut>() || Is32BitFloat<ValueTypeOut>()) ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sqrt: Input and outputs must have the same number of elements." ); }
#endif	// #ifdef NN9_SAFETY_CHECK
				Square_Float( &_vIn[0], &_vOut[0], _vIn.size() );
				return _vOut;
			}
			if constexpr ( Is64BitFloat<ValueTypeIn>() && Is64BitFloat<ValueTypeOut>() ) {
#ifdef NN9_SAFETY_CHECK
				if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Sqrt: Input and outputs must have the same number of elements." ); }
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
		 * \param _tType The view/container type.
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
		 * \param _tType The view/container type.
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
	};

}	// namespace nn9

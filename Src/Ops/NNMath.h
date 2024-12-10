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
					uint16_t * _pBF16 = reinterpret_cast<uint16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 64 )
					float fTmp[16];
					while ( sSize >= 16 ) {
						__m512 mFloats = bfloat16::loadu_bf16_to_fp32_16( _pBF16 );
						_mm512_store_ps( fTmp, mFloats );

						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[0] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[1] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[2] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[3] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[4] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[5] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[6] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[7] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[8] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[9] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[10] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[11] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[12] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[13] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[14] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[15] ) );

						sSize -= 16;
					}

					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(_pBF16)) = _tType::value_type( _fFunc( (*reinterpret_cast<bfloat16_t *>(_pBF16)) ) );
						++_pBF16;
						--sSize;
					}
					return _vValues;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if constexpr ( IsBFloat16<ValueType>() ) {
				if ( Utilities::IsAvx2FSupported() ) {
					// Decode 16 bfloat16_t's at once for super-fast processing.
					uint16_t * _pBF16 = reinterpret_cast<uint16_t *>(&_vValues[0]);
					size_t sSize = _vValues.size();
					NN9_ALIGN( 32 )
					float fTmp[8];
					while ( sSize >= 8 ) {
						__m512 mFloats = bfloat16::loadu_bf16_to_fp32_8( _pBF16 );
						_mm512_store_ps( fTmp, mFloats );

						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[0] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[1] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[2] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[3] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[4] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[5] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[6] ) );
						(*reinterpret_cast<bfloat16_t *>(_pBF16++)) = _tType::value_type( _fFunc( fTmp[7] ) );

						sSize -= 8;
					}

					while ( sSize ) {
						(*reinterpret_cast<bfloat16_t *>(_pBF16)) = _tType::value_type( _fFunc( (*reinterpret_cast<bfloat16_t *>(_pBF16)) ) );
						++_pBF16;
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
		 * \tparam _tType The view/container type.
		 * \tparam _tFunc The function type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \param _fFunc A pointer to the function to call on each item in the view.
		 * \throw If _DEBUG, throws if the views are not the same lengths.
		 * \return Returns _vOut.
		 **/
		template <typename _tType, typename _tFunc>
		static _tType &												Func( const _tType &_vIn, _tType &_vOut, _tFunc _fFunc ) {
#if _DEBUG
			if ( _vIn.size() != _vOut.size() ) { throw std::runtime_error( "Math::Func: Input and outputs must have the same number of elements." ); }
#endif	// #if _DEBUG
			for ( size_t i = 0; i < _vIn.size(); ++i ) {
				_vOut[i] = _tType::value_type( _fFunc( _vIn[i] ) );
			}
			return _vOut;
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
		 * Computes element-wise acos().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Acos( const _tType &_vIn, _tType &_vOut ) {
			return Func<_tType>( _vIn, _vOut, [](auto x) { return std::acos( static_cast<double>(x) ); } );
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
		 * Computes element-wise asin().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Asin( const _tType &_vIn, _tType &_vOut ) {
			return Func<_tType>( _vIn, _vOut, [](auto x) { return std::asin( static_cast<double>(x) ); } );
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
		 * Computes element-wise atan().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Atan( const _tType &_vIn, _tType &_vOut ) {
			return Func<_tType>( _vIn, _vOut, [](auto x) { return std::atan( static_cast<double>(x) ); } );
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
		 * Computes element-wise acosh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Acosh( const _tType &_vIn, _tType &_vOut ) {
			return Func<_tType>( _vIn, _vOut, [](auto x) { return std::acosh( static_cast<double>(x) ); } );
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
		 * Computes element-wise asinh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Asinh( const _tType &_vIn, _tType &_vOut ) {
			return Func<_tType>( _vIn, _vOut, [](auto x) { return std::asinh( static_cast<double>(x) ); } );
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
		 * Computes element-wise atanh().
		 * 
		 * \tparam _tType The view/container type.
		 * \param _vIn The input view.
		 * \param _vOut The output view.
		 * \return Returns _vValues.
		 */
		template <typename _tType>
		static _tType &												Atanh( const _tType &_vIn, _tType &_vOut ) {
			return Func<_tType>( _vIn, _vOut, [](auto x) { return std::atanh( static_cast<double>(x) ); } );
		}

	};

}	// namespace nn9

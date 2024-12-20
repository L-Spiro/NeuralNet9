/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Provides definitions for the base weight types.
 */

#pragma once

#include "NN9BFloat16.h"
#include "NN9Float16.h"

#include <complex>


namespace nn9 {

	/** The types of supported weights. */
	enum NN9_TYPE {
		NN9_T_BFLOAT16,
		NN9_T_FLOAT16,
		NN9_T_FLOAT,
		NN9_T_DOUBLE,

		NN9_T_UINT8,
		NN9_T_UINT16,
		NN9_T_UINT32,
		NN9_T_UINT64,

		NN9_T_INT8,
		NN9_T_INT16,
		NN9_T_INT32,
		NN9_T_INT64,

		NN9_T_BOOL,

		NN9_T_COMPLEX64,
		NN9_T_COMPLEX128,

		NN9_T_QINT8,
		NN9_T_QINT16,
		NN9_T_QINT32,

		NN9_T_QUINT8,

		NN9_T_OTHER,
	};


	/**
	 * Class Types
	 * \brief Provides functionality related to types.
	 *
	 * Description: Provides functionality related to types.
	 */
	class Types {
	public :
		// == Functions.
		/**
		 * A constexpr function that checks if T is a 64-bit float type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is float, false otherwise.
		 */
		template <typename T>
		static constexpr bool											Is64BitFloat() { return std::is_same<T, double>::value; }

		/**
		 * A constexpr function that checks if T is a 32-bit float type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is float, false otherwise.
		 */
		template <typename T>
		static constexpr bool											Is32BitFloat() { return std::is_same<T, float>::value; }

		/**
		 * A constexpr function that checks if T is a bfloat16_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is bfloat16_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsBFloat16() { return std::is_same<T, bfloat16_t>::value; }

		/**
		 * A constexpr function that checks if T is a nn9::float16 type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is nn9::float16, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsFloat16() { return std::is_same<T, nn9::float16>::value; }

		/**
		 * A constexpr function that checks if T is a int8_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is int8_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsInt8() { return std::is_same<T, int8_t>::value; }

		/**
		 * A constexpr function that checks if T is a uint8_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is uint8_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsUint8() { return std::is_same<T, uint8_t>::value; }

		/**
		 * A constexpr function that checks if T is a int16_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is int16_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsInt16() { return std::is_same<T, int16_t>::value; }

		/**
		 * A constexpr function that checks if T is a uint16_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is uint16_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsUint16() { return std::is_same<T, uint16_t>::value; }

		/**
		 * A constexpr function that checks if T is a int32_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is int32_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsInt32() { return std::is_same<T, int32_t>::value; }

		/**
		 * A constexpr function that checks if T is a uint32_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is uint32_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsUint32() { return std::is_same<T, uint32_t>::value; }

		/**
		 * A constexpr function that checks if T is a int64_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is int64_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsInt64() { return std::is_same<T, int64_t>::value; }

		/**
		 * A constexpr function that checks if T is a uint64_t type.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is uint64_t, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsUint64() { return std::is_same<T, uint64_t>::value; }

		/**
		 * A constexpr function that checks if T is a type that is suitable for an integer SIMD register (__m512i/__m256i).
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is an integer type, false otherwise.
		 */
		template <typename T>
		static constexpr bool											SimdInt() { return std::is_integral<T>::value; }

		/**
		 * A constexpr function that checks if T is a type that is suitable for a float SIMD register (__m512/__m256).
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is float type suitable for a 32-bit foating-point SIMD register, false otherwise.
		 */
		template <typename T>
		static constexpr bool											SimdFloat() { return IsFloat16<T>() || IsBFloat16<T>() || Is32BitFloat<T>(); }

		/**
		 * A constexpr function that checks if T is a type that is suitable for a double SIMD register (__m512d/__m256d).
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is float type suitable for a 64-bit foating-point SIMD register, false otherwise.
		 */
		template <typename T>
		static constexpr bool											SimdDouble() { return IsFloat16<T>() || IsBFloat16<T>() || Is32BitFloat<T>(); }

		/**
		 * \brief A constexpr function that checks if T is an unsigned type.
		 *
		 * This relies on std::is_unsigned, which checks if T is an unsigned integral type.
		 * Types like unsigned int, unsigned long, etc., will return Returns true.
		 * Non-integral types or signed integral types will return false.
		 *
		 * \tparam T The type to check.
		 * \return Returns true if T is an unsigned integral type, false otherwise.
		 */
		template <typename T>
		static constexpr bool											IsUnsigned() { return std::is_unsigned<T>::value; }

		/**
		 * Gets the size of a known type.
		 * 
		 * \param _tType The type whose size is to be obtained.
		 * \return Returns the size of the given type or 0.
		 **/
		static inline constexpr size_t									SizeOf( NN9_TYPE _tType );

		/**
		 * Gets the NN9_TYPE based off the actual C++ type.
		 * 
		 * \tparam _tType The type to convert into an NN9_TYPE value.
		 * \throw Throws if the given type is not recognized.
		 * \return Returns the NN9_TYPE value corresponding to the given native C++ type.
		 **/
		template <typename _tType>
		static inline constexpr NN9_TYPE								ToType();
	};


	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// DEFINITIONS
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// == Functions.
	/**
	 * Gets the size of a known type.
	 * 
	 * \param _tType The type whose size is to be obtained.
	 * \return Returns the size of the given type or 0.
	 **/
	inline constexpr size_t Types::SizeOf( NN9_TYPE _tType ) {
		switch ( _tType ) {
			case NN9_T_BFLOAT16 : { return sizeof( bfloat16_t ); }
			case NN9_T_FLOAT16 : { return sizeof( nn9::float16 ); }
			case NN9_T_FLOAT : { return sizeof( float ); }
			case NN9_T_DOUBLE : { return sizeof( double ); }

			case NN9_T_UINT8 : { return sizeof( uint8_t ); }
			case NN9_T_UINT16 : { return sizeof( uint16_t ); }
			case NN9_T_UINT32 : { return sizeof( uint32_t ); }
			case NN9_T_UINT64 : { return sizeof( uint64_t ); }

			case NN9_T_INT8 : { return sizeof( uint8_t ); }
			case NN9_T_INT16 : { return sizeof( int16_t ); }
			case NN9_T_INT32 : { return sizeof( int32_t ); }
			case NN9_T_INT64 : { return sizeof( int64_t ); }

			case NN9_T_BOOL : { return sizeof( bool ); }

			case NN9_T_COMPLEX64 : { return sizeof( std::complex<float> ); }
			case NN9_T_COMPLEX128 : { return sizeof( std::complex<double> ); }

			case NN9_T_QINT8 : { return sizeof( uint8_t ); }
			case NN9_T_QINT16 : { return sizeof( int16_t ); }
			case NN9_T_QINT32 : { return sizeof( int32_t ); }

			case NN9_T_QUINT8 : { return sizeof( uint8_t ); }
		}
		return 0;
	}

	/**
	 * Gets the NN9_TYPE based off the actual C++ type.
	 * 
	 * \tparam _tType The type to convert into an NN9_TYPE value.
	 * \throw Throws if the given type is not recognized.
	 * \return Returns the NN9_TYPE value corresponding to the given native C++ type.
	 **/
	template <typename _tType>
	inline constexpr NN9_TYPE Types::ToType() {
		if constexpr ( std::is_same<_tType, bfloat16_t>::value ) { return NN9_T_BFLOAT16; }
		if constexpr ( std::is_same<_tType, nn9::float16>::value ) { return NN9_T_FLOAT16; }
		if constexpr ( std::is_same<_tType, float>::value ) { return NN9_T_FLOAT; }
		if constexpr ( std::is_same<_tType, double>::value ) { return NN9_T_DOUBLE; }

		if constexpr ( std::is_same<_tType, uint8_t>::value ) { return NN9_T_UINT8; }
		if constexpr ( std::is_same<_tType, uint16_t>::value ) { return NN9_T_UINT16; }
		if constexpr ( std::is_same<_tType, uint32_t>::value ) { return NN9_T_UINT32; }
		if constexpr ( std::is_same<_tType, uint64_t>::value ) { return NN9_T_UINT64; }

		if constexpr ( std::is_same<_tType, int8_t>::value ) { return NN9_T_INT8; }
		if constexpr ( std::is_same<_tType, int16_t>::value ) { return NN9_T_INT16; }
		if constexpr ( std::is_same<_tType, int32_t>::value ) { return NN9_T_INT32; }
		if constexpr ( std::is_same<_tType, int64_t>::value ) { return NN9_T_INT64; }

		if constexpr ( std::is_same<_tType, bool>::value ) { return NN9_T_BOOL; }

		if constexpr ( std::is_same<_tType, std::complex<float>>::value ) { return NN9_T_COMPLEX64; }
		if constexpr ( std::is_same<_tType, std::complex<double>>::value ) { return NN9_T_COMPLEX128; }

		throw std::runtime_error( "Types::ToType: Unrecognized type." );
	}

}	// namespace nn9

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

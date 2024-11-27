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


		NN9_T_OTHER,
	};

	/** Layer types. */
	enum NN9_LAYER_TYPE {
		NN9_LT_INVALID,
		NN9_LT_INPUT,
		NN9_LT_HIDDEN,
		NN9_LT_POOL,
	};


	/**
	 * Class CTypes
	 * \brief Provides functionality related to types.
	 *
	 * Description: Provides functionality related to types.
	 */
	class CTypes {
	public :
		// == Functions.
		/**
		 * Gets the size of a known type.
		 * 
		 * \param _tType The type whose size is to be obtained.
		 * \return Returns the size of the given type or 0.
		 **/
		static inline size_t									SizeOf( NN9_TYPE _tType );
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
	inline size_t CTypes::SizeOf( NN9_TYPE _tType ) {
		switch ( _tType ) {
			case NN9_T_BFLOAT16 : { return sizeof( bfloat16_t ); }
			case NN9_T_FLOAT16 : { return sizeof( nn9::float16 ); }
			case NN9_T_FLOAT : { return sizeof( float ); }
			case NN9_T_DOUBLE : { return sizeof( double ); }

			case NN9_T_UINT8 : { return sizeof( uint8_t ); }
			case NN9_T_UINT16 : { return sizeof( uint16_t ); }
			case NN9_T_UINT32 : { return sizeof( uint32_t ); }
			case NN9_T_UINT64 : { return sizeof( uint64_t ); }
		}
		return 0;
	}

}	// namespace nn9

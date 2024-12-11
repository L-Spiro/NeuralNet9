/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A software implementation of bfloat16.  Can be seamlessly swapped out for hardware-supported bfloat16.
 */

#include "NN9BFloat16.h"
#include "NN9Float16.h"


namespace nn9 {

	bfloat16::bfloat16( float16 _fVal ) {
		// Truncate the float to 16-bit by discarding the lower 16 bits.
		float fValue = float( _fVal );
		// Benchmark against (1000000*5000) values.
		// Hi:	7.06102
		// Lo:	6.86468
		// Av:	6.885046666666666
		struct s {
			uint16_t						ui16Low;
			uint16_t						ui16High;
		};
		m_u16Value = (*reinterpret_cast<s *>(&fValue)).ui16High;
	}

}	// namespace nn9

/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A software implementation of float16.
 */

#include "NN9Float16.h"
#include "NN9BFloat16.h"


namespace nn9 {

	float16::float16( bfloat16 _bfval ) :
		m_u16Value( FloatToUint16( float( _bfval ) ) ) {
	}

}	// namespace nn9

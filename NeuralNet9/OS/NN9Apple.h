/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: macOS/iOS/tvOS macros and header.
 */

#pragma once

#ifdef __APPLE__

#include <stdexcept>

#define LSN_APPLE

#if 0
inline uint64_t __emulu( unsigned int a, unsigned int b ) {
	return static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
}
#endif


#endif  // #ifdef __APPLE__

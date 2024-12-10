/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Cross-platform math functions.
 */

#pragma once

#include "NN9Macros.h"

#include <cstdint>
#ifdef __GNUC__
#include <math.h>
#endif	// #ifdef __GNUC__
#include <stdexcept>


namespace nn9 {



}	// namespace nn9


// ===============================
// sincos/sincosf
// ===============================
extern "C" {
#if defined( _MSC_VER )
#if defined( _WIN64 )
extern void										sincos( double _dAngle, double * _pdSin, double * _pdCos );
extern void										sincosf( float _fAngle, float * _pfSin, float * _pfCos );
#else
// 32 bit implementation in inline assembly.
inline void										sincos( double _dAngle, double * _pdSin, double * _pdCos ) {
	double dSin, dCos;
	__asm {
		fld QWORD PTR[_dAngle]
		fsincos
		fstp QWORD PTR[dCos]
		fstp QWORD PTR[dSin]
		fwait
	}
	(*_pdSin) = dSin;
	(*_pdCos) = dCos;
}
inline void										sincosf( float _fAngle, float * _pfSin, float * _pfCos ) {
	float fSinT, fCosT;
	__asm {
		fld DWORD PTR[_fAngle]					// Load the 32-bit float into the FPU stack.
		fsincos									// Compute cosine and sine.
		fstp DWORD PTR[fCosT]					// Store the cosine value.
		fstp DWORD PTR[fSinT]					// Store the sine value.
		fwait									// Wait for the FPU to finish.
	}
	(*_pfSin) = fSinT;
	(*_pfCos) = fCosT;
}
#endif	// #if defined( _WIN64 )
#elif defined( __GNUC__ )
	#ifndef sincos
		#define sincos		                    __sincos
	#endif	// #ifndef sincos
	#ifndef sincos
		#define sincosf		                    __sincosf
	#endif	// #ifndef sincos
#else
#endif	// #if defined( _MSC_VER )
}	// extern "C"


// ===============================
// 128-Bit Div/Mul
// ===============================
#if defined( _MSC_VER )
    #ifdef _WIN64
        #pragma intrinsic( _udiv128 )
    #else
        inline uint64_t                         _udiv128( uint64_t _ui64High, uint64_t _ui64Low, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
            if ( _ui64Divisor == 0 ) {
		        throw std::overflow_error( "_udiv128: Division by zero is not allowed." );
	        }

	        if ( _ui64High >= _ui64Divisor ) {
		        throw std::overflow_error( "_udiv128: The division would overflow the 64-bit quotient." );
	        }

	        if ( _ui64High == 0 ) {
		        if ( _pui64Remainder ) { (*_pui64Remainder) = _ui64Low % _ui64Divisor; }
		        return _ui64Low / _ui64Divisor;
	        }

	        uint64_t ui64Q = 0;
	        uint64_t ui64R = _ui64High;

	        for ( int I = 63; I >= 0; --I ) {
		        ui64R = (ui64R << 1) | ((_ui64Low >> I) & 1);

		        if ( ui64R >= _ui64Divisor ) {
			        ui64R -= _ui64Divisor;
			        ui64Q |= 1ULL << I;
		        }
	        }

	        if ( _pui64Remainder ) { (*_pui64Remainder) = ui64R; }
	        return ui64Q;
        }
    #endif  // #ifdef _WIN64
#elif defined( __x86_64__ ) || defined( _M_X64 )
	#include <immintrin.h>

	// Implementation using x86_64 assembly for GCC and Clang
	inline uint64_t                             _udiv128( uint64_t _ui64High, uint64_t _ui64Low, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
		uint64_t ui64Quot, ui64Rem;

		asm(
			"divq %4"
			: "=a"(ui64Quot), "=d"(ui64Rem)
			: "a"(_ui64Low), "d"(_ui64High), "r"(_ui64Divisor)
		);

		if ( _pui64Remainder ) {
			(*_pui64Remainder) = ui64Rem;
		}
		return ui64Quot;
	}
#elif defined( __SIZEOF_INT128__ )
	// Implementation for compilers that support __uint128_t (e.g., GCC, Clang)
	inline uint64_t                             _udiv128( uint64_t _ui64High, uint64_t _ui64Low, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
		if ( _ui64Divisor == 0 ) {
			throw std::overflow_error( "_udiv128: Division by zero is not allowed." );
		}

		if ( _ui64High >= _ui64Divisor ) {
			throw std::overflow_error( "_udiv128: The division would overflow the 64-bit quotient." );
		}

		if ( _ui64High == 0 ) {
			if ( _pui64Remainder ) { (*_pui64Remainder) = _ui64Low % _ui64Divisor; }
			return _ui64Low / _ui64Divisor;
		}
	
		// Combine the high and low parts into a single __uint128_t value.
		__uint128_t ui128Dividend = static_cast<__uint128_t>(_ui64High) << 64 | _ui64Low;
	
		(*_pui64Remainder) = static_cast<uint64_t>(ui128Dividend % _ui64Divisor);
		return static_cast<uint64_t>(ui128Dividend / _ui64Divisor);
	}
#endif  // #if defined( _MSC_VER )


#if defined( _MSC_VER )
    #ifndef _WIN64
        // Mercilessly ripped from: https://stackoverflow.com/a/46924301
        // Still need to find somewhere from which to mercilessly rip a _udiv128() implementation.  MERCILESSLY.
        #include <cstdint>
        #include <intrin.h>
        //#include <winnt.h>
        inline uint64_t NN9_FASTCALL            _umul128( uint64_t _ui64Multiplier, uint64_t _ui64Multiplicand,
            uint64_t * _pui64ProductHi ) {
            // _ui64Multiplier   = ab = a * 2^32 + b
            // _ui64Multiplicand = cd = c * 2^32 + d
            // ab * cd = a * c * 2^64 + (a * d + b * c) * 2^32 + b * d
            uint64_t ui64A = _ui64Multiplier >> 32;
            uint64_t ui64B = static_cast<uint32_t>(_ui64Multiplier);
            uint64_t ui64C = _ui64Multiplicand >> 32;
            uint64_t ui64D = static_cast<uint32_t>(_ui64Multiplicand);

            uint64_t ui64Ad = __emulu( static_cast<unsigned int>(ui64A), static_cast<unsigned int>(ui64D) );
            uint64_t ui64Bd = __emulu( static_cast<unsigned int>(ui64B), static_cast<unsigned int>(ui64D) );

            uint64_t ui64Abdc = ui64Ad + __emulu( static_cast<unsigned int>(ui64B), static_cast<unsigned int>(ui64C) );
            uint64_t ui64AbdcCarry = (ui64Abdc < ui64Ad);

            // _ui64Multiplier * _ui64Multiplicand = _pui64ProductHi * 2^64 + ui64ProductLo
            uint64_t ui64ProductLo = ui64Bd + (ui64Abdc << 32);
            uint64_t ui64ProductLoCarry = (ui64ProductLo < ui64Bd);
            (*_pui64ProductHi) = __emulu( static_cast<unsigned int>(ui64A), static_cast<unsigned int>(ui64C) ) + (ui64Abdc >> 32) + (ui64AbdcCarry << 32) + ui64ProductLoCarry;

            return ui64ProductLo;
        }
    #else
        #pragma intrinsic( _udiv128 )
    #endif  // #ifndef _WIN64

#elif defined( __GNUC__ )
	inline uint64_t                             _umul128( uint64_t _ui64Multiplier, uint64_t _ui64Multiplicand,
		uint64_t * _pui64ProductHi ) {
		__uint128_t ui128Tmp = static_cast<__uint128_t>(_ui64Multiplier) * static_cast<__uint128_t>(_ui64Multiplicand);
		(*_pui64ProductHi) = static_cast<uint64_t>(ui128Tmp >> 64);
		return static_cast<uint64_t>(ui128Tmp);
	}
#endif  // #if defined( _MSC_VER )

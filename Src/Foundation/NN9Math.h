/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Cross-platform math functions.
 */

#pragma once

#include "NN9Macros.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif	// #ifdef __AVX2__


namespace nn9 {

	/**
	 * \brief Inverse of the standard normal CDF ("probit") using Peter J. Acklam's polynomial/rational approximation.
	 *
	 * Valid for 0 < p < 1. For p <= 0 or p >= 1, this function will return ±∞ or NaN, 
	 * though in practice we only call it with 0 < p < 1.
	 *
	 * Reference: Peter J. Acklam, "An algorithm for computing the inverse normal cumulative 
	 * distribution function," 2010, <http://home.online.no/~pjacklam/notes/invnorm/>
	 *
	 * \param _dP Probability in (0, 1).
	 * \return double The quantile value z satisfying Φ(z) = _dP.
	 */
	inline double								AcklamInverseNormal( double _dP ) {
		// Coefficients in rational approximations.
		// For 0 < _dP < 0.5.
		static const double dA[6] = {
			-3.969683028665376e+01,
			 2.209460984245205e+02,
			-2.759285104469687e+02,
			 1.383577518672690e+02,
			-3.066479806614716e+01,
			 2.506628277459239e+00
		};
		static const double dB[5] = {
			-5.447609879822406e+01,
			 1.615858368580409e+02,
			-1.556989798598866e+02,
			 6.680131188771972e+01,
			-1.328068155288572e+01
		};
		// For 0.5 <= _dP < 1.
		static const double dC[6] = {
			-7.784894002430293e-03,
			-3.223964580411365e-01,
			-2.400758277161838e+00,
			-2.549732539343734e+00,
			 4.374664141464968e+00,
			 2.938163982698783e+00
		};
		static const double dD[4] = {
			 7.784695709041462e-03,
			 3.224671290700398e-01,
			 2.445134137142996e+00,
			 3.754408661907416e+00
		};

		// Constants.
		constexpr double dPlow  = 0.02425;
		constexpr double dPHigh = 1.0 - dPlow;

		// For symmetrical reasons, we handle the central region differently from the tails
		double dQ, dR, dX;

		if ( _dP < 0 || _dP > 1 ) {
			// Domain error.
			return std::numeric_limits<double>::quiet_NaN();
		}
		else if ( _dP == 0 ) { return -std::numeric_limits<double>::infinity(); }
		else if ( _dP == 1 ) { return  std::numeric_limits<double>::infinity(); }
		else if ( _dP < dPlow ) {
			// Left tail.
			dQ = std::sqrt( -2.0 * std::log( _dP ) );
			dX = (((((dC[0] * dQ + dC[1]) * dQ + dC[2]) * dQ + dC[3]) * dQ + dC[4]) * dQ + dC[5]) /
				 ((((dD[0] * dQ + dD[1]) * dQ + dD[2]) * dQ + dD[3]) * dQ + 1.0);
			if ( _dP < 0.5 ) { dX = -dX; }
		}
		else if ( _dP > dPHigh ) {
			// Right tail.
			dQ = std::sqrt( -2.0 * std::log( 1.0 - _dP ) );
			dX = (((((dC[0] * dQ + dC[1]) * dQ + dC[2]) * dQ + dC[3]) * dQ + dC[4]) * dQ + dC[5]) /
				 ((((dD[0] * dQ + dD[1]) * dQ + dD[2]) * dQ + dD[3]) * dQ + 1.0);
			if ( _dP > 0.5 ) { dX =  dX; }
			else             { dX = -dX; }
		}
		else {
			// Central region.
			dQ = _dP - 0.5;
			dR = dQ * dQ;
			dX = ((((((dA[0] * dR + dA[1]) * dR + dA[2]) * dR + dA[3]) * dR + dA[4]) * dR + dA[5]) * dQ) /
				 (((((dB[0] * dR + dB[1]) * dR + dB[2]) * dR + dB[3]) * dR + dB[4]) * dR + 1.0);
		}

		return dX;
	}

	/**
	 * \brief Computes the inverse error function Erfinv(x) using Acklam's method + standard normal.
	 *
	 * Erfinv(x) = Φ⁻¹( (x+1)/2 ) / √2,  for -1 < x < 1.
	 *
	 * - Returns NaN if |x| >= 1.
	 * - For x = 0, result is 0 exactly.
	 *
	 * \param _dX Input value, must satisfy -1 < _dX < 1 for a finite result.
	 * \return double Erfinv(_dX).
	 */
	inline double								Erfinv( double _dX ) {
		// Domain check.
		if ( std::fabs(_dX) >= 1.0 ) { return std::numeric_limits<double>::quiet_NaN(); }
		if ( _dX == 0.0 ) { return 0.0; }

		bool bNeg = _dX < 0.0;

		// Transform _dX to a probability for the normal CDF.
		// dP in (0,1).
		double dP = 0.5 * (_dX + 1.0);

		// Inverse standard normal of dP.
		double dZ = AcklamInverseNormal( dP );

		// Scale by 1/sqrt(2).
		constexpr double dInvSqrt2 = 0.70710678118654757273731092936941422522068023681640625; // 1/sqrt(2)
		double dRes = std::fabs( dZ * dInvSqrt2 );
		return bNeg ? -dRes : dRes;
	}

	/**
	 * Function to compute cotangent.
	 * 
	 * \param _dX The input.
	 * \return The cotangent of _dX.
	 **/
	static inline double						cot( double _dX ) {
		return 1.0 / std::tan( _dX );
	}

	/**
	 * \brief Computes the Digamma function (\f$\psi(x)\f$).
	 *
	 * The function implements:
	 * - Reflection formula (\f$\psi(1 - x) - \psi(x) = \pi \cot(\pi x)\f$) 
	 *   for negative values.
	 * - Recurrence relation (\f$\psi(x+1) = \psi(x) + 1/x\f$) 
	 *   to shift small values of x to a larger range.
	 * - Asymptotic expansion for \f$x \ge 10\f$:
	 *   \f[
	 *   \psi(x) \approx \ln(x) - \frac{1}{2x} - \frac{1}{12x^2} + 
	 *   \frac{1}{120x^4} - \frac{1}{252x^6} + \cdots
	 *   \f]
	 *
	 * \param _dX The input value for which Digamma is to be calculated.
	 * \return The computed Digamma value, \f$\psi(dX)\f$.
	 */
	extern double								digamma( double _dX );

	/**
	 * \brief Computes P(a, x) = the lower regularized incomplete gamma function,
	 *        using a domain split and standard series/continued-fraction expansions.
	 *
	 *        P(a, x) = gamma(a, x)/Gamma(a).
	 *
	 * \param _dA > 0
	 * \param _dX >= 0
	 * \return P(a, x)
	 */
	extern double								LowerRegGamma( double _dA, double _dX );

	/**
	 * \brief Series expansion for the lower regularized incomplete gamma, used if x < a + 1.
	 *
	 * \param _dA > 0
	 * \param _dX >= 0
	 * \param _dLogGammaA = ln(Gamma(a))
	 * \return P(a,x)
	 */
	extern double								SeriesP( double _dA, double _dX, double _dLogGammaA );

	/**
	 * \brief Continued-fraction approach for the *upper* regularized incomplete gamma,
	 *        used if x >= a + 1 to get Q(a, x).  Then P(a, x) = 1 - Q(a, x).
	 *
	 * \param _dA > 0
	 * \param _dX >= 0
	 * \param _dLogGammaA = ln(Gamma(a))
	 * \return Q(a,x)
	 */
	extern double								ContFracQ( double _dA, double _dX, double _dLogGammaA );

	/**
	 * \brief Computes the **upper regularized** incomplete gamma function:
	 * 
	 * \f[
	 *   Q(a, x) = \frac{\Gamma(a, x)}{\Gamma(a)} = 1 - P(a, x).
	 * \f]
	 *
	 * - For \f$x < a + 1\f$, we compute \f$P(a,x)\) and return \f$1 - P(a,x)\).
	 * - For \f$x \ge a + 1\f$, we directly compute \f$Q(a,x)\) via the continued fraction.
	 *
	 * \param _dA Shape parameter (must be > 0).
	 * \param _dX Nonnegative real argument (usually \f$x \ge 0\f$).
	 * \return Q(a, x) in the range [0,1].
	 */
	extern double								igammac( double _dA, double _dX );

	/**
	 * \brief Performs round-half-to-even (banker's rounding) on a float.
	 *
	 * \param _fVal The value to round.
	 * \return float Returns the rounded values.
	 */
	static inline float							RoundToEven( float _fVal ) {
		float fFloor = std::floor( _fVal );
		float fDiff = _fVal - fFloor;
		if ( fDiff > 0.5f || (fDiff == 0.5f && std::fmod( fFloor, 2.0f ) != 0.0f) ) {
			return fFloor + 1.0f;
		}
		return fFloor;
	}

	/**
	 * \brief Performs round-half-to-even (banker's rounding) on a double.
	 *
	 * \param _dVal The value to round.
	 * \return double Returns the rounded values.
	 */
	static inline double						RoundToEven( double _dVal ) {
		double dFloor = std::floor( _dVal );
		double dDiff = _dVal - dFloor;
		if ( dDiff > 0.5 || (dDiff == 0.5 && std::fmod( dFloor, 2.0 ) != 0.0) ) {
			return dFloor + 1.0;
		}
		return dFloor;
	}


	/**
	 * \brief Performs saturated addition for signed 64-bit integers.
	 * 
	 * This function adds two signed 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to INT64_MAX. If it results in an underflow, the result is
	 * saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int64_t).
	 * \param _i64B Second operand (int64_t).
	 * \return int64_t The saturated addition result.
	 */
	static inline int64_t						adds( int64_t _i64A, int64_t _i64B ) {
		int64_t iSum = _i64A + _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B > 0 && iSum < 0 ) {
			return std::numeric_limits<int64_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B < 0 && iSum > 0 ) {
			return std::numeric_limits<int64_t>::min();
		}

		return iSum;
	}

	/**
	 * \brief Performs saturated addition for unsigned 64-bit integers.
	 * 
	 * This function adds two unsigned 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to UINT64_MAX.
	 * 
	 * \param _u64A First operand (uint64_t).
	 * \param _u64B Second operand (uint64_t).
	 * \return uint64_t The saturated addition result.
	 */
	static inline uint64_t						adds( uint64_t _u64A, uint64_t _u64B ) {
		uint64_t uSum = _u64A + _u64B;

		// Check for overflow.
		if ( uSum < _u64A ) { return std::numeric_limits<uint64_t>::max(); }

		return uSum;
	}

	/**
	 * \brief Performs saturated subtraction for signed 64-bit integers.
	 * 
	 * This function subtracts the second signed 64-bit integer from the first. If the subtraction
	 * results in an overflow, the result is saturated to INT64_MAX. If it results in an underflow,
	 * the result is saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int64_t).
	 * \param _i64B Second operand (int64_t).
	 * \return int64_t The saturated subtraction result.
	 */
	static inline int64_t						subs( int64_t _i64A, int64_t _i64B ) {
		int64_t iDiff = _i64A - _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B < 0 && iDiff < 0 ) {
			return std::numeric_limits<int64_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B > 0 && iDiff > 0 ) {
			return std::numeric_limits<int64_t>::min();
		}

		return iDiff;
	}

	/**
	 * \brief Performs saturated subtraction for unsigned 64-bit integers.
	 * 
	 * This function subtracts the second unsigned 64-bit integer from the first. If the subtraction
	 * results in an underflow (i.e., if the second operand is greater than the first), the result
	 * is saturated to 0.
	 * 
	 * \param _u64A First operand (uint64_t).
	 * \param _u64B Second operand (uint64_t).
	 * \return uint64_t The saturated subtraction result.
	 */
	static inline uint64_t						subs( uint64_t _u64A, uint64_t _u64B ) {
		// Check for underflow
		if ( _u64A < _u64B ) { return 0; }

		return _u64A - _u64B;
	}

	/**
	 * \brief Performs saturated addition for signed 64-bit integers.
	 * 
	 * This function adds two signed 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to INT64_MAX. If it results in an underflow, the result is
	 * saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int32_t).
	 * \param _i64B Second operand (int32_t).
	 * \return int32_t The saturated addition result.
	 */
	static inline int32_t						adds( int32_t _i64A, int32_t _i64B ) {
		int32_t iSum = _i64A + _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B > 0 && iSum < 0 ) {
			return std::numeric_limits<int32_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B < 0 && iSum > 0 ) {
			return std::numeric_limits<int32_t>::min();
		}

		return iSum;
	}

	/**
	 * \brief Performs saturated addition for unsigned 64-bit integers.
	 * 
	 * This function adds two unsigned 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to UINT64_MAX.
	 * 
	 * \param _u64A First operand (uint32_t).
	 * \param _u64B Second operand (uint32_t).
	 * \return uint32_t The saturated addition result.
	 */
	static inline uint32_t						adds( uint32_t _u64A, uint32_t _u64B ) {
		uint32_t uSum = _u64A + _u64B;

		// Check for overflow.
		if ( uSum < _u64A ) { return std::numeric_limits<uint32_t>::max(); }

		return uSum;
	}

	/**
	 * \brief Performs saturated subtraction for signed 64-bit integers.
	 * 
	 * This function subtracts the second signed 64-bit integer from the first. If the subtraction
	 * results in an overflow, the result is saturated to INT64_MAX. If it results in an underflow,
	 * the result is saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int32_t).
	 * \param _i64B Second operand (int32_t).
	 * \return int32_t The saturated subtraction result.
	 */
	static inline int32_t						subs( int32_t _i64A, int32_t _i64B ) {
		int32_t iDiff = _i64A - _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B < 0 && iDiff < 0 ) {
			return std::numeric_limits<int32_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B > 0 && iDiff > 0 ) {
			return std::numeric_limits<int32_t>::min();
		}

		return iDiff;
	}

	/**
	 * \brief Performs saturated subtraction for unsigned 64-bit integers.
	 * 
	 * This function subtracts the second unsigned 64-bit integer from the first. If the subtraction
	 * results in an underflow (i.e., if the second operand is greater than the first), the result
	 * is saturated to 0.
	 * 
	 * \param _u64A First operand (uint32_t).
	 * \param _u64B Second operand (uint32_t).
	 * \return uint32_t The saturated subtraction result.
	 */
	static inline uint32_t						subs( uint32_t _u64A, uint32_t _u64B ) {
		// Check for underflow
		if ( _u64A < _u64B ) { return 0; }

		return _u64A - _u64B;
	}

	/**
	 * \brief Performs saturated addition for signed 64-bit integers.
	 * 
	 * This function adds two signed 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to INT64_MAX. If it results in an underflow, the result is
	 * saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int16_t).
	 * \param _i64B Second operand (int16_t).
	 * \return int16_t The saturated addition result.
	 */
	static inline int16_t						adds( int16_t _i64A, int16_t _i64B ) {
		int16_t iSum = _i64A + _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B > 0 && iSum < 0 ) {
			return std::numeric_limits<int16_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B < 0 && iSum > 0 ) {
			return std::numeric_limits<int16_t>::min();
		}

		return iSum;
	}

	/**
	 * \brief Performs saturated addition for unsigned 64-bit integers.
	 * 
	 * This function adds two unsigned 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to UINT64_MAX.
	 * 
	 * \param _u64A First operand (uint16_t).
	 * \param _u64B Second operand (uint16_t).
	 * \return uint16_t The saturated addition result.
	 */
	static inline uint16_t						adds( uint16_t _u64A, uint16_t _u64B ) {
		uint16_t uSum = _u64A + _u64B;

		// Check for overflow.
		if ( uSum < _u64A ) { return std::numeric_limits<uint16_t>::max(); }

		return uSum;
	}

	/**
	 * \brief Performs saturated subtraction for signed 64-bit integers.
	 * 
	 * This function subtracts the second signed 64-bit integer from the first. If the subtraction
	 * results in an overflow, the result is saturated to INT64_MAX. If it results in an underflow,
	 * the result is saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int16_t).
	 * \param _i64B Second operand (int16_t).
	 * \return int16_t The saturated subtraction result.
	 */
	static inline int16_t						subs( int16_t _i64A, int16_t _i64B ) {
		int16_t iDiff = _i64A - _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B < 0 && iDiff < 0 ) {
			return std::numeric_limits<int16_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B > 0 && iDiff > 0 ) {
			return std::numeric_limits<int16_t>::min();
		}

		return iDiff;
	}

	/**
	 * \brief Performs saturated subtraction for unsigned 64-bit integers.
	 * 
	 * This function subtracts the second unsigned 64-bit integer from the first. If the subtraction
	 * results in an underflow (i.e., if the second operand is greater than the first), the result
	 * is saturated to 0.
	 * 
	 * \param _u64A First operand (uint16_t).
	 * \param _u64B Second operand (uint16_t).
	 * \return uint16_t The saturated subtraction result.
	 */
	static inline uint16_t						subs( uint16_t _u64A, uint16_t _u64B ) {
		// Check for underflow
		if ( _u64A < _u64B ) { return 0; }

		return _u64A - _u64B;
	}

	/**
	 * \brief Performs saturated addition for signed 64-bit integers.
	 * 
	 * This function adds two signed 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to INT64_MAX. If it results in an underflow, the result is
	 * saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int8_t).
	 * \param _i64B Second operand (int8_t).
	 * \return int8_t The saturated addition result.
	 */
	static inline int8_t						adds( int8_t _i64A, int8_t _i64B ) {
		int8_t iSum = _i64A + _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B > 0 && iSum < 0 ) {
			return std::numeric_limits<int8_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B < 0 && iSum > 0 ) {
			return std::numeric_limits<int8_t>::min();
		}

		return iSum;
	}

	/**
	 * \brief Performs saturated addition for unsigned 64-bit integers.
	 * 
	 * This function adds two unsigned 64-bit integers. If the addition results in an overflow,
	 * the result is saturated to UINT64_MAX.
	 * 
	 * \param _u64A First operand (uint8_t).
	 * \param _u64B Second operand (uint8_t).
	 * \return uint8_t The saturated addition result.
	 */
	static inline uint8_t						adds( uint8_t _u64A, uint8_t _u64B ) {
		uint8_t uSum = _u64A + _u64B;

		// Check for overflow.
		if ( uSum < _u64A ) { return std::numeric_limits<uint8_t>::max(); }

		return uSum;
	}

	/**
	 * \brief Performs saturated subtraction for signed 64-bit integers.
	 * 
	 * This function subtracts the second signed 64-bit integer from the first. If the subtraction
	 * results in an overflow, the result is saturated to INT64_MAX. If it results in an underflow,
	 * the result is saturated to INT64_MIN.
	 * 
	 * \param _i64A First operand (int8_t).
	 * \param _i64B Second operand (int8_t).
	 * \return int8_t The saturated subtraction result.
	 */
	static inline int8_t						subs( int8_t _i64A, int8_t _i64B ) {
		int8_t iDiff = _i64A - _i64B;

		// Check for positive overflow.
		if ( _i64A > 0 && _i64B < 0 && iDiff < 0 ) {
			return std::numeric_limits<int8_t>::max();
		}

		// Check for negative overflow.
		if ( _i64A < 0 && _i64B > 0 && iDiff > 0 ) {
			return std::numeric_limits<int8_t>::min();
		}

		return iDiff;
	}

	/**
	 * \brief Performs saturated subtraction for unsigned 64-bit integers.
	 * 
	 * This function subtracts the second unsigned 64-bit integer from the first. If the subtraction
	 * results in an underflow (i.e., if the second operand is greater than the first), the result
	 * is saturated to 0.
	 * 
	 * \param _u64A First operand (uint8_t).
	 * \param _u64B Second operand (uint8_t).
	 * \return uint8_t The saturated subtraction result.
	 */
	static inline uint8_t						subs( uint8_t _u64A, uint8_t _u64B ) {
		// Check for underflow
		if ( _u64A < _u64B ) { return 0; }

		return _u64A - _u64B;
	}

	/**
	 * \brief Performs saturated multiplication of two int64_t values.
	 *        If the result overflows/underflows, it saturates to the maximum/minimum int64_t value.
	 * \param _i64A The first int64_t operand.
	 * \param _i64B The second int64_t operand.
	 * \return The saturated product of _i64A and _i64B.
	 */
	static inline int64_t						muls( int64_t _i64A, int64_t _i64B ) {
		if ( _i64A == 0 || _i64B == 0 ) {
			return 0;
		}

		if ( _i64A > 0 && _i64B > 0 ) {
			if ( _i64A > std::numeric_limits<int64_t>::max() / _i64B ) { return std::numeric_limits<int64_t>::max(); }
		}
		else if ( _i64A < 0 && _i64B < 0 ) {
			if ( _i64A < std::numeric_limits<int64_t>::max() / _i64B ) { return std::numeric_limits<int64_t>::max(); }
		}
		else if ( _i64A < 0 && _i64B > 0 ) {
			if ( _i64A < std::numeric_limits<int64_t>::min() / _i64B ) { return std::numeric_limits<int64_t>::min(); }
		}
		else if ( _i64A > 0 && _i64B < 0 ) {
			if ( _i64B < std::numeric_limits<int64_t>::min() / _i64A ) { return std::numeric_limits<int64_t>::min(); }
		}
		return _i64A * _i64B;
	}

	/**
	 * \brief Performs saturated multiplication of two uint64_t values.
	 *        If the result overflows, it saturates to the maximum uint64_t value.
	 * \param _ui64A The first uint64_t operand.
	 * \param _ui64B The second uint64_t operand.
	 * \return The saturated product of _ui64A and _ui64B.
	 */
	static inline uint64_t						muls( uint64_t _ui64A, uint64_t _ui64B ) {
		if ( _ui64A == 0 || _ui64B == 0 ) { return 0; }

		if ( _ui64A > std::numeric_limits<uint64_t>::max() / _ui64B ) { return std::numeric_limits<uint64_t>::max(); }
		return _ui64A * _ui64B;
	}

	/**
	 * \brief Performs saturated multiplication of two int32_t values.
	 *        If the result overflows/underflows, it saturates to the maximum/minimum int32_t value.
	 * \param _i32A The first int32_t operand.
	 * \param _i32B The second int32_t operand.
	 * \return The saturated product of _i32A and _i32B.
	 */
	static inline int32_t						muls( int32_t _i32A, int32_t _i32B ) {
		if ( _i32A == 0 || _i32B == 0 ) {
			return 0;
		}

		if ( _i32A > 0 && _i32B > 0 ) {
			if ( _i32A > std::numeric_limits<int32_t>::max() / _i32B ) { return std::numeric_limits<int32_t>::max(); }
		}
		else if ( _i32A < 0 && _i32B < 0 ) {
			if ( _i32A < std::numeric_limits<int32_t>::max() / _i32B ) { return std::numeric_limits<int32_t>::max(); }
		}
		else if ( _i32A < 0 && _i32B > 0 ) {
			if ( _i32A < std::numeric_limits<int32_t>::min() / _i32B ) { return std::numeric_limits<int32_t>::min(); }
		}
		else if ( _i32A > 0 && _i32B < 0 ) {
			if ( _i32B < std::numeric_limits<int32_t>::min() / _i32A ) { return std::numeric_limits<int32_t>::min(); }
		}
		return _i32A * _i32B;
	}

	/**
	 * \brief Performs saturated multiplication of two uint32_t values.
	 *        If the result overflows, it saturates to the maximum uint32_t value.
	 * \param _ui32A The first uint32_t operand.
	 * \param _ui32B The second uint32_t operand.
	 * \return The saturated product of _ui32A and _ui32B.
	 */
	static inline uint32_t						muls( uint32_t _ui32A, uint32_t _ui32B ) {
		if ( _ui32A == 0 || _ui32B == 0 ) { return 0; }

		if ( _ui32A > std::numeric_limits<uint32_t>::max() / _ui32B ) { return std::numeric_limits<uint32_t>::max(); }
		return _ui32A * _ui32B;
	}

	/**
	 * \brief Performs saturated multiplication of two int16_t values.
	 *        If the result overflows/underflows, it saturates to the maximum/minimum int16_t value.
	 * \param _i16A The first int16_t operand.
	 * \param _i16B The second int16_t operand.
	 * \return The saturated product of _i16A and _i16B.
	 */
	static inline int16_t						muls( int16_t _i16A, int16_t _i16B ) {
		if ( _i16A == 0 || _i16B == 0 ) {
			return 0;
		}

		if ( _i16A > 0 && _i16B > 0 ) {
			if ( _i16A > std::numeric_limits<int16_t>::max() / _i16B ) { return std::numeric_limits<int16_t>::max(); }
		}
		else if ( _i16A < 0 && _i16B < 0 ) {
			if ( _i16A < std::numeric_limits<int16_t>::max() / _i16B ) { return std::numeric_limits<int16_t>::max(); }
		}
		else if ( _i16A < 0 && _i16B > 0 ) {
			if ( _i16A < std::numeric_limits<int16_t>::min() / _i16B ) { return std::numeric_limits<int16_t>::min(); }
		}
		else if ( _i16A > 0 && _i16B < 0 ) {
			if ( _i16B < std::numeric_limits<int16_t>::min() / _i16A ) { return std::numeric_limits<int16_t>::min(); }
		}
		return _i16A * _i16B;
	}

	/**
	 * \brief Performs saturated multiplication of two uint16_t values.
	 *        If the result overflows, it saturates to the maximum uint16_t value.
	 * \param _ui16A The first uint16_t operand.
	 * \param _ui16B The second uint16_t operand.
	 * \return The saturated product of _ui16A and _ui16B.
	 */
	static inline uint16_t						muls( uint16_t _ui16A, uint16_t _ui16B ) {
		if ( _ui16A == 0 || _ui16B == 0 ) { return 0; }

		if ( _ui16A > std::numeric_limits<uint16_t>::max() / _ui16B ) { return std::numeric_limits<uint16_t>::max(); }
		return _ui16A * _ui16B;
	}

	/**
	 * \brief Performs saturated multiplication of two int8_t values.
	 *        If the result overflows/underflows, it saturates to the maximum/minimum int8_t value.
	 * \param _i8A The first int8_t operand.
	 * \param _i8B The second int8_t operand.
	 * \return The saturated product of _i8A and _i8B.
	 */
	static inline int8_t						muls( int8_t _i8A, int8_t _i8B ) {
		if ( _i8A == 0 || _i8B == 0 ) {
			return 0;
		}

		if ( _i8A > 0 && _i8B > 0 ) {
			if ( _i8A > std::numeric_limits<int8_t>::max() / _i8B ) { return std::numeric_limits<int8_t>::max(); }
		}
		else if ( _i8A < 0 && _i8B < 0 ) {
			if ( _i8A < std::numeric_limits<int8_t>::max() / _i8B ) { return std::numeric_limits<int8_t>::max(); }
		}
		else if ( _i8A < 0 && _i8B > 0 ) {
			if ( _i8A < std::numeric_limits<int8_t>::min() / _i8B ) { return std::numeric_limits<int8_t>::min(); }
		}
		else if ( _i8A > 0 && _i8B < 0 ) {
			if ( _i8B < std::numeric_limits<int8_t>::min() / _i8A ) { return std::numeric_limits<int8_t>::min(); }
		}
		return _i8A * _i8B;
	}

	/**
	 * \brief Performs saturated multiplication of two uint8_t values.
	 *        If the result overflows, it saturates to the maximum uint8_t value.
	 * \param _ui8A The first uint8_t operand.
	 * \param _ui8B The second uint8_t operand.
	 * \return The saturated product of _ui8A and _ui8B.
	 */
	static inline uint8_t						muls( uint8_t _ui8A, uint8_t _ui8B ) {
		if ( _ui8A == 0 || _ui8B == 0 ) { return 0; }

		if ( _ui8A > std::numeric_limits<uint8_t>::max() / _ui8B ) { return std::numeric_limits<uint8_t>::max(); }
		return _ui8A * _ui8B;
	}

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

#ifdef __AVX2__
	inline __m256								_mm256_abs_ps( __m256 _mX ) {
		// Create a mask that clears the sign bit: 0x7FFFFFFF.
		__m256i mMask = _mm256_set1_epi32( 0x7FFFFFFF );

		// Reinterpret the mask as __m256 for bitwise AND with floats.
		__m256 mMaskPs = _mm256_castsi256_ps( mMask );

		// Perform bitwise AND to clear the sign bit in all elements.
		return _mm256_and_ps( _mX, mMaskPs );
	}

	inline __m256d								_mm256_abs_pd( __m256d _mX ) {
		// Create a mask that clears the sign bit: 0x7FFFFFFFFFFFFFFF.
		__m256i mMask = _mm256_set1_epi64x( 0x7FFFFFFFFFFFFFFF );

		// Reinterpret the mask as __m256d for bitwise AND with doubles.
		__m256d mMaskPd = _mm256_castsi256_pd( mMask );

		// Perform bitwise AND to clear the sign bit in all elements.
		return _mm256_and_pd( _mX, mMaskPd );
	}

#endif	// #ifdef __AVX2__

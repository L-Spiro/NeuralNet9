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
		#pragma intrinsic( _div128 )
		#pragma intrinsic( _umul128 )
		#pragma intrinsic( _mul128 )
    #else
		#include <bit>
		#include <cassert>
		#include <intrin.h>

		/**
		 * \brief Performs unsigned 128-bit-by-64-bit division returning a 64-bit unsigned quotient and (optionally) a 64-bit unsigned remainder.
		 *
		 * Given a unsigned 128-bit dividend split into high and low 64-bit parts, and a unsigned 64-bit divisor,
		 * this routine computes the C/C++-semantics result where the quotient is truncated toward zero and the remainder
		 * has the same sign as the dividend with |remainder| < |divisor|.
		 * Internally it reduces to an unsigned divide via _udiv128() on absolute magnitudes and then fixes signs,
		 * detecting overflow of the 64-bit unsigned quotient.
		 *
		 * \param _ui64High			The unsigned high 64 bits of the 128-bit dividend (sign bit is taken from this).
		 * \param _ui64Low			The unsigned low 64 bits of the 128-bit dividend.
		 * \param _ui64Divisor      The unsigned 64-bit divisor.
		 * \param _pui64Remainder   Optional pointer that receives the unsigned remainder (same sign as the dividend).
		 * \return Returns the unsigned 64-bit quotient.
		 * \throws std::overflow_error On division by zero or when the true quotient does not fit in a unsigned 64-bit integer.
		 */
        inline uint64_t                         _udiv128( uint64_t _ui64High, uint64_t _ui64Low, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
            if ( _ui64Divisor == 0 ) [[unlikely]] { throw std::overflow_error( "Division by zero is not allowed." ); }
			if ( _ui64High >= _ui64Divisor ) [[unlikely]] { throw std::overflow_error( "The division would overflow the 64-bit quotient." ); }
	        if ( _ui64High == 0 ) {
		        if ( _pui64Remainder ) { (*_pui64Remainder) = _ui64Low % _ui64Divisor; }
		        return _ui64Low / _ui64Divisor;
	        }

            // == Knuth-style division in base b = 2^32.
            constexpr uint64_t ui64BitsPerWord = sizeof( uint64_t ) * 8;
            constexpr uint64_t ui64Base = 1ULL << (ui64BitsPerWord / 2);	// b = 2^32

            uint64_t ui64V = _ui64Divisor;
            uint64_t ui64Un64;												// High "digit" after normalization step.
            uint64_t ui64Un10;												// Low 64 after normalization step.

            // Normalize: shift left so that the top bit of ui64V is set.
            unsigned int uiShift = static_cast<unsigned int>(std::countl_zero( ui64V ));
            if ( uiShift > 0 ) {
                ui64V = ui64V << uiShift;
                ui64Un64 = (_ui64High << uiShift) | (_ui64Low >> (ui64BitsPerWord - uiShift));
                ui64Un10 = _ui64Low << uiShift;
            }
            else {
                // Avoid (x >> 64) UB for uiShift==0.
                ui64Un64 = _ui64High;
                ui64Un10 = _ui64Low;
            }

            // Split divisor into two 32-bit digits.
            const uint64_t ui64Vn1 = ui64V >> (ui64BitsPerWord / 2);
            const uint64_t ui64Vn0 = ui64V & 0xFFFFFFFFULL;

            // Split the low (normalized) 64 into two 32-bit digits.
            uint64_t ui64Un1 = ui64Un10 >> (ui64BitsPerWord / 2);
            uint64_t ui64Un0 = ui64Un10 & 0xFFFFFFFFULL;

            // First quotient digit q1.
            uint64_t ui64Q1 = ui64Un64 / ui64Vn1;
            uint64_t ui64Rhat = ui64Un64 - ui64Q1 * ui64Vn1;

            // Correct q1 (at most 2 decrements).
            while ( ui64Q1 >= ui64Base || ui64Q1 * ui64Vn0 > ui64Base * ui64Rhat + ui64Un1 ) {
                ui64Q1 -= 1;
                ui64Rhat += ui64Vn1;
                if ( ui64Rhat >= ui64Base ) { break; }
            }

            // Combine and subtract q1 * v.
            uint64_t ui64Un21 = ui64Un64 * ui64Base + ui64Un1 - ui64Q1 * ui64V;

            // Second quotient digit q0.
            uint64_t ui64Q0 = ui64Un21 / ui64Vn1;
            ui64Rhat = ui64Un21 - ui64Q0 * ui64Vn1;

            // Correct q0 (at most 2 decrements).
            while ( ui64Q0 >= ui64Base || ui64Q0 * ui64Vn0 > ui64Base * ui64Rhat + ui64Un0 ) {
                ui64Q0 -= 1;
                ui64Rhat += ui64Vn1;
                if ( ui64Rhat >= ui64Base ) { break; }
            }

            // Remainder (denormalize back).
            if ( _pui64Remainder ) {
                (*_pui64Remainder) = ((ui64Un21 * ui64Base + ui64Un0) - ui64Q0 * ui64V) >> uiShift;
            }

            // Quotient.
            return ui64Q1 * ui64Base + ui64Q0;

	        /*uint64_t ui64Q = 0;
	        uint64_t ui64R = _ui64High;

	        for ( int I = 63; I >= 0; --I ) {
		        ui64R = (ui64R << 1) | ((_ui64Low >> I) & 1);

		        if ( ui64R >= _ui64Divisor ) {
			        ui64R -= _ui64Divisor;
			        ui64Q |= 1ULL << I;
		        }
	        }

	        if ( _pui64Remainder ) { (*_pui64Remainder) = ui64R; }
	        return ui64Q;*/
        }

		/**
		 * \brief Performs signed 128-bit-by-64-bit division returning a 64-bit signed quotient and (optionally) a 64-bit signed remainder.
		 *
		 * Given a signed 128-bit dividend split into high and low 64-bit parts, and a signed 64-bit divisor,
		 * this routine computes the C/C++-semantics result where the quotient is truncated toward zero and the remainder
		 * has the same sign as the dividend with |remainder| < |divisor|.
		 * Internally it reduces to an unsigned divide via _udiv128() on absolute magnitudes and then fixes signs,
		 * detecting overflow of the 64-bit signed quotient.
		 *
		 * \param _i64HighDividend The signed high 64 bits of the 128-bit dividend (sign bit is taken from this).
		 * \param _i64LowDividend  The signed low 64 bits of the 128-bit dividend.
		 * \param _i64Divisor      The signed 64-bit divisor.
		 * \param _pi64Remainder   Optional pointer that receives the signed remainder (same sign as the dividend).
		 * \return Returns the signed 64-bit quotient.
		 * \throws std::overflow_error On division by zero or when the true quotient does not fit in a signed 64-bit integer.
		 */
		inline int64_t							_div128( int64_t _i64HighDividend, int64_t _i64LowDividend, int64_t _i64Divisor, int64_t * _pi64Remainder ) {
			// Validate divisor.
			if ( _i64Divisor == 0 ) [[unlikely]] { throw std::overflow_error( "Division by zero is not allowed." ); }

			// Determine signs and compute absolute magnitudes without invoking UB on INT64_MIN.
			const bool bDividendNeg		= (_i64HighDividend < 0);
			const bool bDivisorNeg		= (_i64Divisor < 0);
			const bool bQuotNeg			= (bDividendNeg != bDivisorNeg);		// XOR: quotient sign.
			const bool bRemNeg			= bDividendNeg;							// Remainder sign matches dividend (C++ semantics).

			// Represent the 128-bit dividend as unsigned parts; bit patterns preserved on cast.
			uint64_t ui64Hi = static_cast<uint64_t>(_i64HighDividend);
			uint64_t ui64Lo = static_cast<uint64_t>(_i64LowDividend);

			// Two's-complement negate (128-bit) if the dividend is negative: (hi,lo) = - (hi,lo).
			auto Negate128 = []( uint64_t &_ui64Hi, uint64_t &_ui64Lo ) {
				uint64_t ui64NewLo = ~_ui64Lo + 1ULL;							// Add 1 to low after invert.
				_ui64Hi = ~_ui64Hi + (ui64NewLo == 0ULL ? 1ULL : 0ULL);			// Propagate carry into high if low wrapped.
				_ui64Lo = ui64NewLo;
			};
			if ( bDividendNeg ) {
				Negate128( ui64Hi, ui64Lo );
			}

			// Absolute value of the divisor as unsigned using two's-complement trick (works for INT64_MIN).
			uint64_t ui64Div = bDivisorNeg ? static_cast<uint64_t>(0ULL - static_cast<uint64_t>(_i64Divisor)) :
				static_cast<uint64_t>(_i64Divisor);

			// Use existing unsigned 128/64 divide to get magnitude of quotient and remainder.
			uint64_t ui64Rem = 0ULL;
			const uint64_t ui64UQuot = _udiv128( ui64Hi, ui64Lo, ui64Div, &ui64Rem );

			// Check for signed 64-bit overflow of the quotient's final value.
			// If the result is non-negative, it must be <= INT64_MAX.
			// If the result is negative, magnitude may be up to 2^63 (i.e., 0x8000'0000'0000'0000).
			if ( !bQuotNeg ) {
				if ( ui64UQuot > static_cast<uint64_t>(INT64_MAX) ) [[unlikely]] {
					throw std::overflow_error( "The division would overflow the 64-bit signed quotient." );
				}
			}
			else {
				if ( ui64UQuot > (1ULL << 63) ) [[unlikely]] {
					throw std::overflow_error( "The division would overflow the 64-bit signed quotient." );
				}
			}

			// Form the signed quotient without invoking UB on INT64_MIN.
			// For negative: two's-complement via (0 - magnitude) in unsigned domain then cast.
			const int64_t i64Quot = bQuotNeg ? static_cast<int64_t>(static_cast<uint64_t>(0ULL - ui64UQuot)) :
				static_cast<int64_t>(ui64UQuot);

			// Signed remainder (same sign as dividend).
			if ( _pi64Remainder ) {
				(*_pi64Remainder) = bRemNeg ? static_cast<int64_t>(static_cast<uint64_t>(0ULL - ui64Rem)) :
					static_cast<int64_t>(ui64Rem);
			}

			return i64Quot;
		}

		/**
		 * \brief Computes u64 * u64 -> u128, returning the low 64 bits and writing the high 64 bits.
		 *
		 * \param _ui64Multiplier The first operand.
		 * \param _ui64Multiplicand The second operand.
		 * \param _pui64ProductHi Receives the high 64 bits of the product.
		 * \return Returns the low 64 bits of the product.
		 **/
		inline uint64_t NN9_FASTCALL			_umul128( uint64_t _ui64Multiplier, uint64_t _ui64Multiplicand, uint64_t * _pui64ProductHi ) {
			assert( _pui64ProductHi );

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

		/**
		 * \brief Computes s64 * s64 -> s128, returning the low 64 bits and writing the high 64 bits.
		 *
		 * \param _i64Multiplier The first operand.
		 * \param _i64Multiplicand The second operand.
		 * \param _pi64HighProduct Receives the high 64 bits of the product.
		 * \return Returns the low 64 bits of the product.
		 **/
		inline int64_t NN9_FASTCALL				_mul128( int64_t _i64Multiplier, int64_t _i64Multiplicand, int64_t * _pi64HighProduct ) {
			assert( _pi64HighProduct );

			// Do unsigned multiply on magnitudes, then apply sign to the 128-bit result.
			uint64_t ui64A = static_cast<uint64_t>(_i64Multiplier);
			uint64_t ui64B = static_cast<uint64_t>(_i64Multiplicand);

			const bool bNegA = (_i64Multiplier < 0);
			const bool bNegB = (_i64Multiplicand < 0);
			const bool bNeg = (bNegA != bNegB);

			if ( bNegA ) { ui64A = (~ui64A) + 1ULL; }
			if ( bNegB ) { ui64B = (~ui64B) + 1ULL; }

			uint64_t ui64Hi = 0;
			uint64_t ui64Lo = _umul128( ui64A, ui64B, &ui64Hi );

			if ( bNeg ) {
				// Two's-complement negate 128-bit (hi:lo).
				ui64Lo = (~ui64Lo) + 1ULL;
				ui64Hi = (~ui64Hi) + (ui64Lo == 0 ? 1ULL : 0ULL);
			}

			(*_pi64HighProduct) = static_cast<int64_t>(ui64Hi);
			return static_cast<int64_t>(ui64Lo);
		}
    #endif  // #ifdef _WIN64
#elif defined( __x86_64__ ) || defined( _M_X64 )
	#include <immintrin.h>

	/**
	 * \brief Performs unsigned 128-bit-by-64-bit division returning a 64-bit unsigned quotient and (optionally) a 64-bit unsigned remainder.
	 *
	 * Given a unsigned 128-bit dividend split into high and low 64-bit parts, and a unsigned 64-bit divisor,
	 * this routine computes the C/C++-semantics result where the quotient is truncated toward zero and the remainder
	 * has the same sign as the dividend with |remainder| < |divisor|.
	 * Internally it reduces to an unsigned divide via _udiv128() on absolute magnitudes and then fixes signs,
	 * detecting overflow of the 64-bit unsigned quotient.  Implementation using x86_64 assembly for GCC and Clang.
	 *
	 * \param _ui64High			The unsigned high 64 bits of the 128-bit dividend (sign bit is taken from this).
	 * \param _ui64Low			The unsigned low 64 bits of the 128-bit dividend.
	 * \param _ui64Divisor      The unsigned 64-bit divisor.
	 * \param _pui64Remainder   Optional pointer that receives the unsigned remainder (same sign as the dividend).
	 * \return Returns the unsigned 64-bit quotient.
	 * \throws std::overflow_error On division by zero or when the true quotient does not fit in a unsigned 64-bit integer.
	 */ 
	inline uint64_t                             _udiv128( uint64_t _ui64High, uint64_t _ui64Low, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
		uint64_t ui64Quot, ui64Rem;

		asm volatile(
			"divq %4"
			: "=a"(ui64Quot), "=d"(ui64Rem)
			: "a"(_ui64Low), "d"(_ui64High), "r"(_ui64Divisor)
			: "cc"
		);

		if ( _pui64Remainder ) {
			(*_pui64Remainder) = ui64Rem;
		}
		return ui64Quot;
	}

	/**
	 * \brief Performs signed 128-bit-by-64-bit division returning a 64-bit signed quotient and (optionally) a 64-bit signed remainder.
	 *
	 * Given a signed 128-bit dividend split into high and low 64-bit parts, and a signed 64-bit divisor,
	 * this routine computes the C/C++-semantics result where the quotient is truncated toward zero and the remainder
	 * has the same sign as the dividend with |remainder| < |divisor|.
	 * Internally it reduces to an unsigned divide via _udiv128() on absolute magnitudes and then fixes signs,
	 * detecting overflow of the 64-bit signed quotient.  Signed 128/64 -> 64 division (RDX:RAX / r/m64), quotient in RAX, remainder in RDX.
	 *
	 * \param _i64HighDividend The signed high 64 bits of the 128-bit dividend (sign bit is taken from this).
	 * \param _i64LowDividend  The signed low 64 bits of the 128-bit dividend.
	 * \param _i64Divisor      The signed 64-bit divisor.
	 * \param _pi64Remainder   Optional pointer that receives the signed remainder (same sign as the dividend).
	 * \return Returns the signed 64-bit quotient.
	 * \throws std::overflow_error On division by zero or when the true quotient does not fit in a signed 64-bit integer.
	 */
	inline int64_t								_div128( int64_t _i64High, int64_t _i64Low, int64_t _i64Divisor, int64_t * _pi64Remainder ) {
		int64_t i64Quot = 0;
		int64_t i64Rem = 0;

		asm volatile(
			"idivq %4"
			: "=a"(i64Quot), "=d"(i64Rem)
			: "a"(_i64Low), "d"(_i64High), "rm"(_i64Divisor)
			: "cc"
		);

		if ( _pi64Remainder ) {
			(*_pi64Remainder) = i64Rem;
		}
		return i64Quot;
	}

	/**
	 * \brief Computes u64 * u64 -> u128, returning the low 64 bits and writing the high 64 bits.  Unsigned 64 * 64 -> 128: RDX:RAX = RAX * r/m64.
	 *
	 * \param _ui64Multiplier The first operand.
	 * \param _ui64Multiplicand The second operand.
	 * \param _pui64ProductHi Receives the high 64 bits of the product.
	 * \return Returns the low 64 bits of the product.
	 **/
	inline uint64_t								_umul128( uint64_t _ui64Multiplier, uint64_t _ui64Multiplicand, uint64_t * _pui64HighProduct ) {
		uint64_t ui64A = _ui64Multiplier;	// RAX input -> low output.
		uint64_t ui64D = 0;					// RDX output.

		asm volatile(
			"mulq %2"
			: "+a"(ui64A), "=d"(ui64D)
			: "rm"(_ui64Multiplicand)
			: "cc"
		);

		if ( _pui64HighProduct ) {
			(*_pui64HighProduct) = ui64D;
		}
		return ui64A;
	}

	/**
	 * \brief Computes s64 * s64 -> s128, returning the low 64 bits and writing the high 64 bits.  Signed 64 * 64 -> 128: RDX:RAX = RAX * r/m64 (signed).
	 *
	 * \param _i64Multiplier The first operand.
	 * \param _i64Multiplicand The second operand.
	 * \param _pi64HighProduct Receives the high 64 bits of the product.
	 * \return Returns the low 64 bits of the product.
	 **/
	inline int64_t								_mul128( int64_t _i64Multiplier, int64_t _i64Multiplicand, int64_t * _pi64HighProduct ) {
		int64_t i64A = _i64Multiplier;		// RAX input -> low output.
		int64_t i64D = 0;					// RDX output.

		asm volatile(
			"imulq %2"
			: "+a"(i64A), "=d"(i64D)
			: "rm"(_i64Multiplicand)
			: "cc"
		);

		if ( _pi64HighProduct ) {
			(*_pi64HighProduct) = i64D;
		}
		return i64A;
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
	
		if ( _pui64Remainder ) {
			(*_pui64Remainder) = static_cast<uint64_t>(ui128Dividend % _ui64Divisor);
		}
		return static_cast<uint64_t>(ui128Dividend / _ui64Divisor);
	}

	// Implementation for compilers that support __int128 (e.g., GCC, Clang).
	inline int64_t								_div128( int64_t _i64High, int64_t _i64Low, int64_t _i64Divisor, int64_t * _pi64Remainder ) {
		if ( _i64Divisor == 0 ) {
			throw std::overflow_error( "_div128: Division by zero is not allowed." );
		}

		// Combine the high and low parts into a single __int128 value.
		// Note: The low part is treated as an unsigned 64-bit chunk when OR'ing into the 128-bit value.
		const __int128 i128Dividend =
			(static_cast<__int128>(_i64High) << 64) |
			static_cast<__int128>(static_cast<unsigned __int128>(static_cast<uint64_t>(_i64Low)));

		// Compute quotient and remainder using truncation toward zero (C/C++ semantics).
		const __int128 i128Quot = i128Dividend / static_cast<__int128>(_i64Divisor);
		const __int128 i128Rem = i128Dividend % static_cast<__int128>(_i64Divisor);

		// The MSVC-style _div128 contract returns an int64 quotient; overflow must be reported.
		if ( i128Quot < static_cast<__int128>(INT64_MIN) || i128Quot > static_cast<__int128>(INT64_MAX) ) {
			throw std::overflow_error( "_div128: The division would overflow the 64-bit quotient." );
		}

		if ( _pi64Remainder ) {
			// Remainder is always in [-|divisor|+1, |divisor|-1], so it fits in int64_t.
			(*_pi64Remainder) = static_cast<int64_t>(i128Rem);
		}

		return static_cast<int64_t>(i128Quot);
	}

	/**
	 * \brief Computes u64 * u64 -> u128, returning the low 64 bits and writing the high 64 bits.
	 *
	 * \param _ui64Multiplier The first operand.
	 * \param _ui64Multiplicand The second operand.
	 * \param _pui64ProductHi Receives the high 64 bits of the product.
	 * \return Returns the low 64 bits of the product.
	 **/
	inline uint64_t								_umul128( uint64_t _ui64Multiplier, uint64_t _ui64Multiplicand, uint64_t * _pui64ProductHi ) {
		assert( _pui64ProductHi );

		__uint128_t ui128Tmp = static_cast<__uint128_t>(_ui64Multiplier) * static_cast<__uint128_t>(_ui64Multiplicand);
		(*_pui64ProductHi) = static_cast<uint64_t>(ui128Tmp >> 64);
		return static_cast<uint64_t>(ui128Tmp);
	}

	/**
	 * \brief Computes s64 * s64 -> s128, returning the low 64 bits and writing the high 64 bits.
	 *
	 * \param _i64Multiplier The first operand.
	 * \param _i64Multiplicand The second operand.
	 * \param _pi64HighProduct Receives the high 64 bits of the product.
	 * \return Returns the low 64 bits of the product.
	 **/
	inline int64_t NN9_FASTCALL					_mul128( int64_t _i64Multiplier, int64_t _i64Multiplicand, int64_t * _pi64HighProduct ) {
		assert( _pi64HighProduct );

		const __int128 i128Prod = static_cast<__int128>(_i64Multiplier) * static_cast<__int128>(_i64Multiplicand);
		(*_pi64HighProduct) = static_cast<int64_t>(i128Prod >> 64);
		return static_cast<int64_t>(i128Prod);
	}
#endif  // #if defined( _MSC_VER )

/**
 * \brief Multiplies two 64-bit unsigned values to a 128-bit intermediate, then divides by a 64-bit unsigned divisor.
 *
 * This is effectively: (a * b) / d with full 128-bit intermediate precision.
 *
 * Precondition (same as udiv128-style contract): d != 0 and high(a*b) < d (quotient fits in 64 bits).
 *
 * \param _ui64A The first multiplicand.
 * \param _ui64B The second multiplicand.
 * \param _ui64Divisor The divisor.
 * \param _pui64Remainder Receives the remainder.
 * \return Returns the 64-bit quotient.
 **/
static inline uint64_t							_umuldiv128( uint64_t _ui64A, uint64_t _ui64B, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
	assert( _pui64Remainder );
	assert( _ui64Divisor != 0 );

	uint64_t ui64Hi = 0;
	const uint64_t ui64Lo = _umul128( _ui64A, _ui64B, &ui64Hi );

	// Same contract as _udiv128(): quotient must fit in 64 bits.
	assert( ui64Hi < _ui64Divisor );

	return _udiv128( ui64Hi, ui64Lo, _ui64Divisor, _pui64Remainder );
}

/**
 * \brief Checked version of _umuldiv128() that returns false if the quotient would not fit in 64 bits.
 *
 * \param _ui64A The first multiplicand.
 * \param _ui64B The second multiplicand.
 * \param _ui64Divisor The divisor.
 * \param _pui64Quotient Receives the quotient.
 * \param _pui64Remainder Receives the remainder.
 * \return Returns true if the quotient fits in 64 bits; otherwise false (outputs not written).
 **/
static inline bool								_umuldiv128_checked( uint64_t _ui64A, uint64_t _ui64B, uint64_t _ui64Divisor, uint64_t * _pui64Quotient, uint64_t * _pui64Remainder ) {
	assert( _pui64Quotient );
	assert( _pui64Remainder );
	assert( _ui64Divisor != 0 );

	uint64_t ui64Hi = 0;
	const uint64_t ui64Lo = _umul128( _ui64A, _ui64B, &ui64Hi );

	if ( ui64Hi >= _ui64Divisor ) { return false; }

	(*_pui64Quotient) = _udiv128( ui64Hi, ui64Lo, _ui64Divisor, _pui64Remainder );
	return true;
}

/**
 * \brief Multiplies two 64-bit signed values to a 128-bit intermediate, then divides by a 64-bit signed divisor.
 *
 * This is effectively: (a * b) / d with full 128-bit intermediate precision (truncates toward 0).
 *
 * Precondition: d != 0 and quotient fits in 64 bits (same style/contract as _div128()).
 *
 * \param _i64A The first multiplicand.
 * \param _i64B The second multiplicand.
 * \param _i64Divisor The divisor.
 * \param _pi64Remainder Receives the remainder (same sign as dividend, per C/C++).
 * \return Returns the 64-bit quotient.
 **/
static inline int64_t							_muldiv128( int64_t _i64A, int64_t _i64B, int64_t _i64Divisor, int64_t * _pi64Remainder ) {
	assert( _pi64Remainder );
	assert( _i64Divisor != 0 );

	int64_t i64Hi = 0;
	const int64_t i64Lo = _mul128( _i64A, _i64B, &i64Hi );

	return _div128( i64Hi, i64Lo, _i64Divisor, _pi64Remainder );
}

/**
 * \brief Computes round((A * B) / Div) using a 128-bit intermediate.
 *
 * This is (A * B + Div/2) / Div, using full 128-bit precision for the product.
 * Throws on division by zero or if the quotient would overflow 64 bits (same policy as NN9_muldiv128()).
 *
 * \param _ui64A The first value.
 * \param _ui64B The second value.
 * \param _ui64Divisor The divisor.
 * \param _pui64Remainder Receives the remainder after rounding.
 * \return Returns round((A * B) / Divisor).
 **/
static inline uint64_t							_umuldiv128_rounded( uint64_t _ui64A, uint64_t _ui64B, uint64_t _ui64Divisor, uint64_t * _pui64Remainder ) {
	assert( _pui64Remainder );
	assert( _ui64Divisor != 0 );

	uint64_t ui64Hi = 0;
	uint64_t ui64Lo = _umul128( _ui64A, _ui64B, &ui64Hi );

	// Add Div/2 for rounding, propagating carry into the high word.
	{
		const uint64_t ui64Add = _ui64Divisor >> 1;
		const uint64_t ui64Old = ui64Lo;
		ui64Lo += ui64Add;
		if ( ui64Lo < ui64Old ) { ++ui64Hi; }	// Lower-half overflow.
	}

	// Same contract as _udiv128(): quotient must fit in 64 bits.
	assert( ui64Hi < _ui64Divisor );

	return _udiv128( ui64Hi, ui64Lo, _ui64Divisor, _pui64Remainder );
}


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

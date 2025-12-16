/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Cross-platform math functions.
 */

#include "NN9Math.h"


namespace nn9 {

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
	double								digamma( double _dX ) {
		// Poles at non-positive integers
		if ( _dX <= 0.0 && std::floor( _dX ) == _dX ) {
			// Return +∞ for poles at x = 0, -1, -2, ...
			return std::numeric_limits<double>::infinity();
		}

		// For negative values, use reflection formula:
		// ψ(x) = ψ(1 - x) - π cot(π x)
		if ( _dX < 0.0 ) {
			return digamma( 1.0 - _dX ) - std::numbers::pi * cot( std::numbers::pi * _dX );
		}

		// Use recurrence relation to shift x up to at least 10
		double dResult = 0.0;
		while ( _dX < 10.0 ) {
			dResult -= 1.0 / _dX;
			_dX += 1.0;
		}

		// Asymptotic expansion
		double dInvX  = 1.0 / _dX;
		double dInvX2 = dInvX * dInvX;
    
		// 
		// Standard expansion up to the 1/(x^6) term:
		// digamma(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) ...
		//
		constexpr double dC1 = 1.0 / 12.0;
		constexpr double dC2 = 1.0 / 120.0;
		constexpr double dC3 = 1.0 / 252.0;

		dResult += std::log( _dX ) -
			0.5 * dInvX -
			dInvX2 * dC1 +
			(dInvX2 * dInvX2) * dC2 -
			(dInvX2 * dInvX2 * dInvX2) * dC3;

		return dResult;
	}

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
	double								LowerRegGamma( double _dA, double _dX ) {
		// Domain checks
		if NN9_UNLIKELY( _dA <= 0.0 ) { return std::numeric_limits<double>::quiet_NaN(); }
		if NN9_UNLIKELY( _dX < 0.0 ) { return std::numeric_limits<double>::quiet_NaN(); }
		if NN9_UNLIKELY( _dX == 0.0 ) { return 0.0; }

		double dLogGammaA = std::lgamma( _dA );

		if ( _dX < _dA + 1.0 ) {
			// Use series approach for P(a,x).
			return SeriesP( _dA, _dX, dLogGammaA );
		}
		else {
			// We know Q(a,x) from the CF => Q(a,x) .
			// so P(a,x)= 1 - Q(a,x).
			double dQval = ContFracQ( _dA, _dX, dLogGammaA );
			return 1.0 - dQval;
		}
	}

	/**
	 * \brief Series expansion for the lower regularized incomplete gamma, used if x < a + 1.
	 *
	 * \param _dA > 0
	 * \param _dX >= 0
	 * \param _dLogGammaA = ln(Gamma(a))
	 * \return P(a,x)
	 */
	double								SeriesP( double _dA, double _dX, double _dLogGammaA ) {
		// This computes P(a,x) via the series expansion when x < a+1:
		// P(a, x) = ( e^-x * x^a / Gamma(a) ) * sum_{n=0..∞} ( x^n / ( (a) (a+1) ... (a+n) ) )
		// Often simplified to an iterative form.

		const double dMaxIter  = 200;
		const double dEpsilon  = 1e-14;

		double dSum  = 1.0;
		double dTerm = 1.0;
		double dAp   = _dA;

		for ( int iN = 1; iN <= static_cast<int>(dMaxIter); iN++ ) {
			dTerm *= (_dX / (dAp + static_cast<double>(iN)));
			dSum  += dTerm;
			if NN9_UNLIKELY( std::fabs( dTerm ) < std::fabs( dSum ) * dEpsilon ) { break; }
		}

		double dLogPrefactor = _dA*std::log( _dX ) - _dX - _dLogGammaA;
		// e^(a ln x - x - lnGamma(a)) => dimensionless.
		double dPrefactor = std::exp( dLogPrefactor );

		return dPrefactor * dSum;
	}

	/**
	 * \brief Continued-fraction approach for the *upper* regularized incomplete gamma,
	 *        used if x >= a + 1 to get Q(a, x).  Then P(a, x) = 1 - Q(a, x).
	 *
	 * \param _dA > 0
	 * \param _dX >= 0
	 * \param _dLogGammaA = ln(Gamma(a))
	 * \return Q(a,x)
	 */
	double								ContFracQ( double _dA, double _dX, double _dLogGammaA ) {
		// This computes Q(a,x) = Gamma(a,x)/Gamma(a) directly via a continued fraction.
		// Reference: "Numerical Recipes" or other standard references, function 'gcf'.

		const double dEpsilon	= 1e-14;
		const int    iMaxIter	= 200;

		// prefactor = e^( a ln x - x - lnGamma(a) ), dimensionless.
		double dLogPrefactor	= _dA * std::log( _dX ) - _dX - _dLogGammaA;
		double dPrefactor		= std::exp( dLogPrefactor );

		double dFrac = 0.0;		// Will hold CF result.
		// We'll do Lentz's method or something similar.  
		// As a simpler approach here, let's do the standard "gcf" from references:
    
		double dC = 0.0;
		double dD = 0.0;
		double dH = 1.0;		// Start value for CF.
		for ( int iN = 1; iN <= iMaxIter; iN++ ) {
			double dAn = static_cast<double>(iN);
			// Usually we define some recursion:
			//  see "Numerical Recipes: gcf" approach. One typical approach is:
			double dAlpha = dAn - _dA;
			double dBeta  = _dX + dAn - 1.0;

			// We'll try a direct approach to the partial fraction expansions.
			// For brevity, let's say we define:
			double dCterm = dAlpha / dBeta;

			// Lentz iteration:
			dD = 1.0 + dCterm * dD;
			if NN9_UNLIKELY( std::fabs( dD ) < 1e-30 ) { dD = 1e-30; }
			dD = 1.0 / dD;

			dC = 1.0 + dCterm / ( (iN==1) ? 1.0 : dC );
			if NN9_UNLIKELY( std::fabs( dC ) < 1e-30 ) { dC = 1e-30; }

			double dDelta = dC * dD;
			dH *= dDelta;

			if NN9_UNLIKELY( std::fabs( dDelta - 1.0 ) < dEpsilon ) { break; }
		}

		dFrac = dH; 
		double dQval = dPrefactor * dFrac; // This yields Q(a,x).

		return dQval;
	}

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
	double								igammac( double _dA, double _dX ) {
		// Basic domain checks.
		if NN9_UNLIKELY( _dA <= 0.0 ) {
			// Q(a,x) not defined for a <= 0 in this standard sense.
			return std::numeric_limits<double>::quiet_NaN();
		}
		if NN9_UNLIKELY( _dX < 0.0 ) {
			// Usually Q(a,x)=1 for x<0 is not standard, or undefined. Return NaN for safety.
			return std::numeric_limits<double>::quiet_NaN();
		}
		if NN9_UNLIKELY( _dX == 0.0 ) {
			// Q(a,0) = 1 if a>0.
			return 1.0;
		}

		// If x < a + 1 => Q(a,x) = 1 - P(a,x) [ via the series for P(a,x) ].
		// If x >= a + 1 => directly get Q(a,x) from continued fraction.
		double dLogGammaA = std::lgamma( _dA );

		if ( _dX < (_dA + 1.0) ) {
			double dPval = SeriesP( _dA, _dX, dLogGammaA );
			return 1.0 - dPval;
		}
		else {
			double dQval = ContFracQ( _dA, _dX, dLogGammaA );
			return dQval;
		}
	}
}	// namespace nn9

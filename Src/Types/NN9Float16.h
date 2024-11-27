/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A software implementation of float16.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <cmath>
#include <type_traits>

namespace nn9 {

	/**
	 * Class float16
	 * \brief A software implementation of float16.
	 *
	 * Description: A software implementation of float16.
	 */
	class float16 {
	public :
		constexpr float16() {}
		//constexpr explicit float16( uint16_t _ui16Bits ) : m_u16Value( _ui16Bits ) {}

		float16( float _fVal ) :
			m_u16Value( FloatToUint16( _fVal ) ) {
		}
		float16( double _dVal ) :
			m_u16Value( FloatToUint16( float( _dVal ) ) ) {
		}
		float16( uint8_t _ui8Value ) :
			m_u16Value( FloatToUint16( float( _ui8Value ) ) ) {
		}
		template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
		float16( T _tValue ) :
			m_u16Value( FloatToUint16( static_cast<float>(_tValue) ) ) {
		}


		// == Operators.
		/**
		 * Cast to float.
		 * 
		 * \return Returns the float value of the float16.
		 **/
		inline operator							float() const {
			return Uint16ToFloat( m_u16Value );
		}

		// ===============================
		// Arithmetic Operators
		// ===============================
		inline float16							operator + ( const float16 &_fOther ) const {
			return float16( static_cast<float>((*this)) + static_cast<float>(_fOther) );
		}

		inline float16							operator - ( const float16 &_fOther ) const {
			return float16( static_cast<float>((*this)) - static_cast<float>(_fOther) );
		}

		inline float16							operator * ( const float16 &_fOther ) const {
			return float16( static_cast<float>((*this)) * static_cast<float>(_fOther) );
		}

		inline float16							operator / ( const float16 &_fOther ) const {
			return float16( static_cast<float>((*this)) / static_cast<float>(_fOther) );
		}

		// ===============================
		// Compound Assignment Operators
		// ===============================
		inline float16 &						operator += ( const float16 &_fOther ) {
			(*this) = (*this) + _fOther;
			return (*this);
		}

		inline float16 &						operator -= ( const float16 &_fOther ) {
			(*this) = (*this) - _fOther;
			return (*this);
		}

		inline float16 &						operator *= ( const float16 &_fOther ) {
			(*this) = (*this) * _fOther;
			return (*this);
		}

		inline float16 &						operator /= ( const float16 &_fOther ) {
			(*this) = (*this) / _fOther;
			return (*this);
		}

		// ===============================
		// Comparison Operators
		// ===============================
		inline bool								operator == ( const float16 &_fOther ) const {
			return static_cast<float>((*this)) == static_cast<float>(_fOther);
		}

		inline bool								operator != ( const float16 &_fOther ) const {
			return !((*this) == _fOther);
		}

		inline bool								operator < ( const float16 &_fOther ) const {
			return static_cast<float>((*this)) < static_cast<float>(_fOther);
		}

		inline bool								operator > ( const float16 &_fOther ) const {
			return static_cast<float>((*this)) > static_cast<float>(_fOther);
		}

		inline bool								operator <= ( const float16 &_fOther ) const {
			return static_cast<float>((*this)) <= static_cast<float>(_fOther);
		}

		inline bool								operator >= ( const float16 &_fOther ) const {
			return static_cast<float>((*this)) >= static_cast<float>(_fOther);
		}


		// == Functions.
		/**
		 * Converts a float to float16.
		 * 
		 * \param _fVal The floating-point value to convert.
		 * \return Returns the 16-bit representation of the given float value as a float16.
		 **/
		static inline uint16_t					FloatToUint16( float _fVal );

		/**
		 * Converts a float16 (represented as a uint16_t) to a float.
		 * 
		 * \param _ui16Val The value to convert to a float.
		 * \return Returns the float value of the given 16-bit representation of a float16.
		 **/
		static inline float						Uint16ToFloat( uint16_t _ui16Val );

#ifdef __AVX512F__
		/**
		 * This function loads 16 float16 values from the source array, converts them to float32,
		 * and stores the results in the destination array.
		 *
		 * \param _pf16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
		 * \return Returns a register with the loaded values.
		 */
		static inline __m512					Convert16Float16ToFloat32( const float16 * _pf16Src );

		/**
		 * This function loads 16 float16 values from the source array, converts them to float32,
		 * and stores the results in the destination array.
		 *
		 * \param _pui16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
		 * \return Returns a register with the loaded values.
		 */
		static inline __m512					Convert16Float16ToFloat32( const uint16_t * _pui16Src ) {
			return Convert16Float16ToFloat32( reinterpret_cast<const float16 *>(_pui16Src) );
		}

		/**
		 * This function loads 16 float16 values from the source array, converts them to float32,
		 * and stores the results in the destination array.
		 *
		 * \param _pf16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
		 * \param _pfDst Pointer to the destination array where 16 float16 values will be stored as uint16_t.
		 * \return Returns _pfDst.
		 */
		static inline float *					Convert16Float16ToFloat32( const float16 * _pf16Src, float * _pfDst ) {
			_mm512_storeu_ps( _pfDst, Convert16Float16ToFloat32( _pf16Src ) );
			return _pfDst;
		}

		/**
		 * This function loads 16 float16 values from the source array, converts them to float32,
		 * and stores the results in the destination array.
		 *
		 * \param _pui16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
		 * \param _pfDst Pointer to the destination array where 16 float16 values will be stored as uint16_t.
		 * \return Returns _pfDst.
		 */
		static inline float *					Convert16Float16ToFloat32( const uint16_t * _pui16Src, float * _pfDst ) {
			return Convert16Float16ToFloat32( reinterpret_cast<const float16 *>(_pui16Src), _pfDst );
		}

		/**
		 * This function takes 16 float32 values from the source array, converts them to float16,
		 * and stores the results in the destination array as uint16_t.
		 *
		 * \param _pfSrc Pointer to the source array containing 16 float32 values.
		 * \return Returns a register with the loaded values.
		 */
		static inline __m256i					Convert16Float32ToFloat16( const float * _pfSrc );

		/**
		 * This function takes 16 float32 values from the source array, converts them to float16,
		 * and stores the results in the destination array as uint16_t.
		 *
		 * \param _pfSrc Pointer to the source array containing 16 float32 values.
		 * \param _pui16Dst Pointer to the destination array where 16 float16 values will be stored as uint16_t.
		 * \return Returns _pui16Dst.
		 */
		static inline uint16_t *				Convert16Float32ToFloat16( const float * _pfSrc, uint16_t * _pui16Dst ) {
			_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pui16Dst), Convert16Float32ToFloat16( _pfSrc ) );
			return _pui16Dst;
		}
#endif	// #ifdef __AVX512F__


		// ===============================
		// Numeric Limits
		// ===============================
		/**
		 * Returns the smallest positive normalized value.
		 * 
		 * \return Returns the smallest positive normalized value.
		 **/
		static constexpr float16				min() noexcept {
			return FromBits( 0x0400 );			// Exponent = 1, Mantissa = 0.
		}

		/**
		 * Returns the largest finite value.
		 * 
		 * \return Returns the largest finite value.
		 **/
		static constexpr float16				max() noexcept {
			return FromBits( 0x7BFF );			// Exponent = 30, Mantissa = all ones.
		}

		/**
		 * Returns the lowest finite value (most negative).
		 * 
		 * \return Returns the lowest finite value (most negative).
		 **/
		static constexpr float16				lowest() noexcept {
			return FromBits( 0xFBFF );			// Sign bit set, exponent = 30, Mantissa = all ones.
		}

		/**
		 * Difference between 1 and the next representable value.
		 * 
		 * \return Difference between 1 and the next representable value.
		 **/
		static constexpr float16				epsilon() noexcept {
			return FromBits( 0x1400 );			// Exponent = 5, Mantissa = 0.
		}

		/**
		 * Smallest positive subnormal value.
		 * 
		 * \return Smallest positive subnormal value.
		 **/
		static constexpr float16				denorm_min() noexcept {
			return FromBits( 0x0001 );			// Exponent = 0, Mantissa = 1.
		}

		/**
		 * Returns positive infinity.
		 * 
		 * \return Returns positive infinity.
		 **/
		static constexpr float16				infinity() noexcept {
			return FromBits( 0x7C00 );			// Exponent = 31 (all ones), Mantissa = 0.
		}

		/**
		 * Returns a quiet NaN.
		 * 
		 * \return Returns a quiet NaN.
		 **/
		static constexpr float16				quiet_NaN() noexcept {
			return FromBits( 0x7E00 );			// Exponent = 31, Mantissa = non-zero.
		}

		/**
		 * Returns a signaling NaN.
		 * 
		 * \return Returns a signaling NaN.
		 **/
		static constexpr float16				signaling_NaN() noexcept {
			return FromBits( 0x7D00 );			// Exponent = 31, Mantissa = specific pattern.
		}

		/**
		 * Utility function to create float16 from raw bits.
		 * 
		 * \param _ui16Bits The 16-bit representation to express as a float16.
		 * \return Returns the 16-bit value expressed as a float16.
		 **/
		static constexpr float16				FromBits( uint16_t _ui16Bits ) noexcept {
			float16 f16This;
			f16This.m_u16Value = _ui16Bits;
			return f16This;
		}

		/**
		 * Method to retrieve raw bits for hashing or other purposes.
		 * 
		 * \return Returns the bitwise representation of the type as a constant expression.
		 **/
		constexpr uint16_t						ToBits() const noexcept {
			return m_u16Value;
		}

		// == Members.
		/** The backing value. */
		uint16_t								m_u16Value;
	};


	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// DEFINITIONS
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// == Functions.
	/**
	 * Converts a float to float16.
	 * 
	 * \param _fVal The floating-point value to convert.
	 * \return Returns the 16-bit representation of the given float value as a float16.
	 **/
	inline uint16_t float16::FloatToUint16( float _fVal ) {
		static_assert( sizeof( uint32_t ) == sizeof( float ) );

		const uint32_t ui32Bits = (*reinterpret_cast<uint32_t *>(&_fVal)) + 0x00001000;
		const uint32_t ui32Expo = (ui32Bits & 0x7F800000) >> 23;
		const uint32_t ui32Mant = ui32Bits & 0x007FFFFF;
		uint16_t ui16Tmp = (ui32Bits & 0x80000000) >> 16 | (ui32Expo > 112) * ((((ui32Expo - 112) << 10) & 0x7C00) | ui32Mant >> 13) | (( ui32Expo < 113) & (ui32Expo > 101)) * ((((0x007FF000 + ui32Mant) >> (125 - ui32Expo)) + 1) >> 1) | (ui32Expo > 143) * 0x7FFF;

		//bool bIsInf = std::isinf( _fVal );
		bool bIsSrcNan = std::isnan( _fVal );
		bool bIsInf = (ui16Tmp & 0x7C00) == 0x7C00 && !bIsSrcNan;

		ui16Tmp = (ui16Tmp * !bIsInf) | ((bIsInf * 0x7C00) | (ui16Tmp & 0x8000));

		return ui16Tmp;
	}

	/**
	 * Converts a float16 (represented as a uint16_t) to a float.
	 * 
	 * \param _ui16Val The value to convert to a float.
	 * \return Returns the float value of the given 16-bit representation of a float16.
	 **/
	inline float float16::Uint16ToFloat( uint16_t _ui16Val ) {
		static_assert( sizeof( uint32_t ) == sizeof( float ) );

		const uint32_t ui32Expo = (_ui16Val & 0x7C00) >> 10;
		const uint32_t ui32Mant = (_ui16Val & 0x03FF) << 13;
		float fManTmp = float( ui32Mant );
		const uint32_t ui32LdZ0 = (*reinterpret_cast<const uint32_t *>(&fManTmp)) >> 23;
		uint32_t ui32Tmp = (_ui16Val & 0x8000) << 16 | (ui32Expo != 0) * ((ui32Expo + 112) << 23 | ui32Mant) | ((ui32Expo == 0) & (ui32Mant != 0)) * ((ui32LdZ0 - 37) << 23 | ((ui32Mant << (150 - ui32LdZ0)) & 0x007FE000));

		// Handle +/-INF.
		bool bIsInf = (_ui16Val & 0x7C00) == 0x7C00;

		// Handle NaN.
		bool bIsNan = bIsInf && (_ui16Val & 0x03FF);
		ui32Tmp = ui32Tmp | (bIsNan * 0x7FC00000);

		bIsInf ^= bIsNan;		
		ui32Tmp = (ui32Tmp * !bIsInf) | (((0x7F800000 * bIsInf) | (ui32Tmp & 0x80000000)));

		return (*reinterpret_cast<float *>(&ui32Tmp));
	}

#ifdef __AVX512F__
	/**
	 * This function loads 16 float16 values from the source array, converts them to float32,
	 * and stores the results in the destination array.
	 *
	 * \param _pf16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
	 * \param _pfDst Pointer to the destination array where 16 float32 results will be stored.
	 */
	inline __m512 float16::Convert16Float16ToFloat32( const float16 * _pf16Src ) {
		// Load 16 uint16_t values into a __m256i vector.
		__m256i mui16Val = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pf16Src) );

		// Zero-extend to 32 bits (convert 16 x uint16_t to 16 x uint32_t).
		__m512i mui32Val = _mm512_cvtepu16_epi32( mui16Val );

		// Extract sign, exponent, and mantissa.
		__m512i mSign = _mm512_slli_epi32( _mm512_and_epi32( mui32Val, _mm512_set1_epi32( 0x8000 ) ), 16 );
		__m512i mExpo = _mm512_and_epi32( mui32Val, _mm512_set1_epi32( 0x7C00 ) );
		__m512i mMant = _mm512_and_epi32( mui32Val, _mm512_set1_epi32( 0x03FF ) );

		// Shift mantissa to align with float32 mantissa bits.
		__m512i mMantShifted = _mm512_slli_epi32( mMant, 13 );

		// Convert mantissa to float to compute leading zeros.
		__m512 mfManTmp = _mm512_cvtepi32_ps( mMantShifted );
		__m512i mfManTmpBits = _mm512_castps_si512( mfManTmp );
		__m512i mui32LdZ0 = _mm512_srli_epi32( mfManTmpBits, 23 );

		// Create masks for different cases.
		__mmask16 mNormal = _mm512_cmpneq_epi32_mask( mExpo, _mm512_setzero_si512() );
		__mmask16 mSubnormal = _mm512_kand(
			_mm512_cmpeq_epi32_mask( mExpo, _mm512_setzero_si512() ),
			_mm512_cmpneq_epi32_mask( mMant, _mm512_setzero_si512() )
		);
		__mmask16 mZero = _mm512_kand(
			_mm512_cmpeq_epi32_mask( mExpo, _mm512_setzero_si512() ),
			_mm512_cmpeq_epi32_mask( mMant, _mm512_setzero_si512() )
		);
		__mmask16 mInf = _mm512_cmpeq_epi32_mask( mExpo, _mm512_set1_epi32( 0x7C00 ) );
		__mmask16 mNan = _mm512_kand( mInf, _mm512_cmpneq_epi32_mask( mMant, _mm512_setzero_si512() ) );

		// Compute for normalized numbers.
		__m512i mExpoNormal = _mm512_add_epi32( _mm512_srli_epi32( mExpo, 10 ), _mm512_set1_epi32( 112 ) );
		mExpoNormal = _mm512_slli_epi32( mExpoNormal, 23 );
		__m512i mui32TmpNormal = _mm512_or_epi32( mSign, _mm512_or_epi32( mExpoNormal, mMantShifted ) );

		// Compute for subnormal numbers.
		__m512i mui32LdZ0Minus37 = _mm512_sub_epi32( mui32LdZ0, _mm512_set1_epi32( 37 ) );
		__m512i mExpoSubnormal = _mm512_slli_epi32( mui32LdZ0Minus37, 23 );
		__m512i mShiftAmount = _mm512_sub_epi32( _mm512_set1_epi32( 150 ), mui32LdZ0 );
		// Compute the mantissa for subnormal numbers.
		__m512i mMantSubnormal = _mm512_and_epi32(
			_mm512_sllv_epi32( mMantShifted, mShiftAmount ),
			_mm512_set1_epi32( 0x007FE000 )
		);
		__m512i mui32TmpSubnormal = _mm512_or_epi32( mSign, _mm512_or_epi32( mExpoSubnormal, mMantSubnormal ) );

		// Initialize result vector.
		__m512i mui32Tmp = _mm512_setzero_si512();

		// Set values for normalized numbers.
		mui32Tmp = _mm512_mask_mov_epi32( mui32Tmp, mNormal, mui32TmpNormal );

		// Set values for subnormal numbers.
		mui32Tmp = _mm512_mask_mov_epi32( mui32Tmp, mSubnormal, mui32TmpSubnormal );

		// Set values for zeros.
		mui32Tmp = _mm512_mask_mov_epi32( mui32Tmp, mZero, mSign );

		// Handle NaNs.
		mui32Tmp = _mm512_mask_or_epi32( mui32Tmp, mNan, mui32Tmp, _mm512_set1_epi32( 0x7FC00000 ) );

		// Handle infinities (excluding NaNs).
		__mmask16 m_inf_only = _kandn_mask16( mNan, mInf );
		mui32Tmp = _mm512_mask_mov_epi32( mui32Tmp, m_inf_only, _mm512_or_epi32(
			mSign, _mm512_set1_epi32( 0x7F800000 )
		));

		// Convert to float32.
		return _mm512_castsi512_ps( mui32Tmp );
	}

	/**
	 * This function takes 16 float32 values from the source array, converts them to float16,
	 * and stores the results in the destination array as uint16_t.
	 *
	 * \param _pfSrc Pointer to the source array containing 16 float32 values.
	 */
	inline __m256i float16::Convert16Float32ToFloat16( const float * _pfSrc ) {
		// Load 16 float32 values.
		__m512 mf32Val = _mm512_loadu_ps( _pfSrc );

		// Reinterpret as integers and add rounding bias.
		__m512i mBits = _mm512_castps_si512( mf32Val );
		__m512i mBitsRounded = _mm512_add_epi32( mBits, _mm512_set1_epi32( 0x00001000 ) );

		// Extract sign, exponent, mantissa.
		__m512i mSign = _mm512_srli_epi32( _mm512_and_epi32( mBitsRounded, _mm512_set1_epi32( 0x80000000 ) ), 16 );
		__m512i mExpo = _mm512_srli_epi32( _mm512_and_epi32( mBitsRounded, _mm512_set1_epi32( 0x7F800000 ) ), 23 );
		__m512i mMant = _mm512_and_epi32( mBitsRounded, _mm512_set1_epi32( 0x007FFFFF ) );

		// Conditions for different cases.
		__mmask16 mExpoGt112 = _mm512_cmp_epi32_mask( mExpo, _mm512_set1_epi32( 112 ), _MM_CMPINT_GT );
		//__mmask16 mExpoLt113 = _mm512_cmp_epi32_mask( mExpo, _mm512_set1_epi32( 113 ), _MM_CMPINT_LT );
		__mmask16 mExpoGt101 = _mm512_cmp_epi32_mask( mExpo, _mm512_set1_epi32( 101 ), _MM_CMPINT_GT );
		__mmask16 mExpoGt143 = _mm512_cmp_epi32_mask( mExpo, _mm512_set1_epi32( 143 ), _MM_CMPINT_GT );
		__mmask16 mExpoLe101 = _mm512_cmp_epi32_mask( mExpo, _mm512_set1_epi32( 101 ), _MM_CMPINT_LE );
		//__mmask16 mExpoEq255 = _mm512_cmp_epi32_mask( mExpo, _mm512_set1_epi32( 255 ), _MM_CMPINT_EQ );

		// NaN detection.
		__mmask16 mIsNan = _mm512_cmp_ps_mask( mf32Val, mf32Val, _CMP_UNORD_Q );

		// Compute normalized numbers.
		__m512i mNorm = _mm512_slli_epi32( _mm512_sub_epi32( mExpo, _mm512_set1_epi32( 112 ) ), 10 );
		mNorm = _mm512_and_epi32( mNorm, _mm512_set1_epi32( 0x7C00 ) );
		mNorm = _mm512_or_epi32( mNorm, _mm512_srli_epi32( mMant, 13 ) );
		mNorm = _mm512_or_epi32( mNorm, mSign );

		// Compute subnormal numbers.
		__m512i mMantSubnorm = _mm512_add_epi32( mMant, _mm512_set1_epi32( 0x007FF000 ) );
		__m512i mShift = _mm512_sub_epi32( _mm512_set1_epi32( 125 ), mExpo );
		__m512i mSubnorm = _mm512_srlv_epi32( mMantSubnorm, mShift );
		mSubnorm = _mm512_srli_epi32( _mm512_add_epi32( mSubnorm, _mm512_set1_epi32( 1 ) ), 1 );
		mSubnorm = _mm512_and_epi32( mSubnorm, _mm512_set1_epi32( 0x03FF ) );
		mSubnorm = _mm512_or_epi32( mSubnorm, mSign );

		// Special cases (NaN and Infinity).
		__m512i mSpecial = _mm512_set1_epi32( 0x7FFF );

		// Initialize result.
		__m512i mui16Tmp = _mm512_setzero_si512();

		// Apply conditions.
		// Special cases: Exponent > 143.
		mui16Tmp = _mm512_mask_mov_epi32(mui16Tmp, mExpoGt143, mSpecial);

		// Normalized numbers: Exponent > 112 and not special case.
		__mmask16 mNormMask = _mm512_kand( mExpoGt112, _knot_mask16( mExpoGt143 ) );
		mui16Tmp = _mm512_mask_mov_epi32( mui16Tmp, mNormMask, mNorm );

		// Subnormal numbers: 101 < Exponent < 113.
		__mmask16 mSubnormMask = _mm512_kand(
			_mm512_kand( _knot_mask16( mExpoGt112 ), mExpoGt101 ),
			_knot_mask16( mExpoGt143 )
		);

		mui16Tmp = _mm512_mask_mov_epi32( mui16Tmp, mSubnormMask, mSubnorm );

		// Zeros: Exponent ≤ 101.
		mui16Tmp = _mm512_mask_mov_epi32( mui16Tmp, mExpoLe101, mSign );

		// Adjust for Infinity.
		__mmask16 mIsInfMask = _mm512_kand(
			_mm512_cmp_epi32_mask( _mm512_and_epi32( mui16Tmp, _mm512_set1_epi32( 0x7C00 ) ), _mm512_set1_epi32( 0x7C00 ), _MM_CMPINT_EQ ),
			_knot_mask16( mIsNan )
		);

		mui16Tmp = _mm512_mask_blend_epi32(
			mIsInfMask,
			mui16Tmp,
			_mm512_or_epi32( mSign, _mm512_set1_epi32( 0x7C00 ) )
		);

		// For NaN, set to 0x7FFF.
		mui16Tmp = _mm512_mask_mov_epi32( mui16Tmp, mIsNan, _mm512_set1_epi32( 0x7FFF ) );

		// Convert to uint16_t.
		return _mm512_cvtepi32_epi16( mui16Tmp );

		// Store result.
		//_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pui16Dst), mf16 );
	}

#endif	// #ifdef __AVX512F__

}	// namespace nn9

namespace std {
	template<>
	struct is_floating_point<nn9::float16> : std::true_type {};
	template<>
	struct is_arithmetic<nn9::float16> : std::true_type {};
	template<>
	struct is_signed<nn9::float16> : std::true_type {};
	template<>
	struct is_scalar<nn9::float16> : std::true_type {};

	template<>
	struct is_standard_layout<nn9::float16> : std::true_type {};
	template<>
	struct is_trivial<nn9::float16> : std::true_type {};


	template<>
    class numeric_limits<nn9::float16> {
    public:
        static constexpr bool is_specialized	= true;

        static constexpr nn9::float16			min() noexcept { return nn9::float16::min(); }
        static constexpr nn9::float16			max() noexcept { return nn9::float16::max(); }
        static constexpr nn9::float16			lowest() noexcept { return nn9::float16::lowest(); }

        static constexpr int digits				= 11;  // mantissa bits including the implicit bit
        static constexpr int digits10			= 3;   // floor(digits * log10(2))
        static constexpr int max_digits10		= 5;   // ceil(1 + digits * log10(2))

        static constexpr bool is_signed			= true;
        static constexpr bool is_integer		= false;
        static constexpr bool is_exact			= false;
        static constexpr int radix				= 2;

        static constexpr nn9::float16			epsilon() noexcept { return nn9::float16::epsilon(); }
        static constexpr nn9::float16			round_error() noexcept { return nn9::float16::FromBits( 0x3800 ); }	// 0.5f.

        static constexpr int min_exponent		= -14;
        static constexpr int min_exponent10		= -4;
        static constexpr int max_exponent		= 15;
        static constexpr int max_exponent10		= 4;

        static constexpr bool has_infinity		= true;
        static constexpr bool has_quiet_NaN		= true;
        static constexpr bool has_signaling_NaN	= true;
        static constexpr std::float_denorm_style has_denorm = std::denorm_present;
        static constexpr bool has_denorm_loss	= false;

        static constexpr nn9::float16			infinity() noexcept { return nn9::float16::infinity(); }
        static constexpr nn9::float16			quiet_NaN() noexcept { return nn9::float16::quiet_NaN(); }
        static constexpr nn9::float16			signaling_NaN() noexcept { return nn9::float16::signaling_NaN(); }
        static constexpr nn9::float16			denorm_min() noexcept { return nn9::float16::denorm_min(); }

        static constexpr bool is_iec559			= true;
        static constexpr bool is_bounded		= true;
        static constexpr bool is_modulo			= false;

        static constexpr bool traps				= false;
        static constexpr bool tinyness_before	= false;
        static constexpr std::float_round_style	round_style = std::round_to_nearest;
    };


	template<>
	struct is_pod<nn9::float16> : std::true_type {};


	template<>
	struct common_type<nn9::float16, float> {
        using type = float;
    };

	template<>
	struct common_type<float, nn9::float16> {
        using type = float;
    };

	template<>
	struct common_type<nn9::float16, double> {
        using type = double;
    };

	template<>
	struct common_type<double, nn9::float16> {
        using type = double;
    };


	template<>
	struct hash<nn9::float16> {
        std::size_t								operator()( const nn9::float16 &_f16Val ) const noexcept {
            return std::hash<uint16_t>()( _f16Val.ToBits() );
        }
    };

}

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

	class bfloat16;
	
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

		float16( bfloat16 _bfval );
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

		/**
		 * Cast to double.
		 * 
		 * \return Returns the double value of the float16.
		 **/
		inline operator							double() const {
			return Uint16ToFloat( m_u16Value );
		}

		/**
		 * Casts to an integer type.
		 * 
		 * \tparam T The integer type to which to cast this object.
		 * \return Returns an integer value of the float16.
		 **/
		template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
		inline operator							T() const {
			return T( static_cast<float>(*this) );
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

		inline float16							operator + ( double _dOther ) const {
			return float16( static_cast<float>((*this)) + _dOther );
		}

		inline float16							operator - ( double _dOther ) const {
			return float16( static_cast<float>((*this)) - _dOther );
		}

		inline float16							operator * ( double _dOther ) const {
			return float16( static_cast<float>((*this)) * _dOther );
		}

		inline float16							operator / ( double _dOther ) const {
			return float16( static_cast<float>((*this)) / _dOther );
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
			return static_cast<float>((*this)) != static_cast<float>(_fOther);
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
		 * Loads 16 float16 values from the source array, converts them to float32,
		 *	and stores the results in the destination array.
		 *
		 * \param _pf16Src Pointer to the source array containing 16 float16 values.
		 * \return Returns a register with the loaded values.
		 */
		static inline __m512					Convert16Float16ToFloat32( const float16 * _pf16Src );

		/**
		 * Loads 16 float16 values from the source array, converts them to float32,
		 *	and stores the results in the destination array.
		 *
		 * \param _pui16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
		 * \return Returns a register with the loaded values.
		 */
		static inline __m512					Convert16Float16ToFloat32( const uint16_t * _pui16Src ) {
			return Convert16Float16ToFloat32( reinterpret_cast<const float16 *>(_pui16Src) );
		}

		/**
		 * Loads 16 float16 values from the source array, converts them to float32,
		 *	and stores the results in the destination array.
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
		 * Loads 16 float16 values from the source array, converts them to float32,
		 *	and stores the results in the destination array.
		 *
		 * \param _pui16Src Pointer to the source array containing 16 uint16_t values representing float16 numbers.
		 * \param _pfDst Pointer to the destination array where 16 float16 values will be stored as uint16_t.
		 * \return Returns _pfDst.
		 */
		static inline float *					Convert16Float16ToFloat32( const uint16_t * _pui16Src, float * _pfDst ) {
			return Convert16Float16ToFloat32( reinterpret_cast<const float16 *>(_pui16Src), _pfDst );
		}

		/**
		 * Takes 16 float32 values from the source array, converts them to float16,
		 *	and stores the results in the destination array as uint16_t.
		 *
		 * \param _pfSrc Pointer to the source array containing 16 float32 values.
		 * \return Returns a register with the loaded values.
		 */
		static inline void						Convert16Float32ToFloat16( float16 * _pf16Dst, __m512 _mf32Val );

#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		// Helper function to invert a mask (for <=, etc.)
		static inline __m256i					mm256_not_si256( __m256i a ) {
			return _mm256_xor_si256( a, _mm256_set1_epi32( -1 ) );
		}

		// Create a mask for (a <= b) using (a <= b) <=> !(a > b).
		static inline __m256i					mm256_cmple_epi32( __m256i a, __m256i b ) {
			__m256i gt = _mm256_cmpgt_epi32( a, b ); // 0xFFFFFFFF where a > b, else 0x0
			return mm256_not_si256( gt );            // Invert it to get <=
		}

		// Create a mask for (a != 0) using (a != 0) <=> !(a == 0).
		static inline __m256i					mm256_cmpneq_epi32_zero( __m256i a ) {
			__m256i eq = _mm256_cmpeq_epi32( a, _mm256_setzero_si256() );
			return mm256_not_si256( eq );
		}

		// Blend using a mask: if mask bit is set, select from b, else from a
		// mask should be 0xFFFFFFFF for true, 0x00000000 for false.
		static inline __m256i					mm256_blendv_epi32( __m256i a, __m256i b, __m256i mask ) {
			return _mm256_blendv_epi8( a, b, mask );
		}

		// Helper: a != b.
		static inline __m256i					mm256_cmpneq_epi32( __m256i a, __m256i b ) {
			__m256i eq = _mm256_cmpeq_epi32( a, b );
			return mm256_not_si256( eq );
		}

		/**
		 * Loads 8 float16 values from the source array, converts them to float32,
		 *	and stores the results in the destination array.
		 *
		 * \param _pf16Src Pointer to the source array containing 8 float16 values.
		 * \return Returns a register with the loaded values.
		 */
		static inline __m256					Convert8Float16ToFloat32( const float16 * _pf16Src );

		/**
		 * Takes 8 float32 values from the source array, converts them to float16,
		 *	and stores the results in the destination array as uint16_t.
		 *
		 * \param _pfSrc Pointer to the source array containing 16 float32 values.
		 * \return Returns a register with the loaded values.
		 */
		static inline void						Convert8Float32ToFloat16( float16 * _pf16Dst, __m256 _mf32Val );
#endif	// #ifdef __AVX2__


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
	 * Loads 16 float16 values from the source array, converts them to float32,
	 *  and stores the results in the destination array.
	 *
	 * \param _pf16Src Pointer to the source array containing 16 float16 values.
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
	 * Takes 16 float32 values from the source array, converts them to float16,
	 *  and stores the results in the destination array as uint16_t.
	 *
	 * \param _pfSrc Pointer to the source array containing 16 float32 values.
	 */
	inline void float16::Convert16Float32ToFloat16( float16 * _pf16Dst, __m512 _mf32Val ) {

		// Reinterpret as integers and add rounding bias.
		__m512i mBits = _mm512_castps_si512( _mf32Val );
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
		__mmask16 mIsNan = _mm512_cmp_ps_mask( _mf32Val, _mf32Val, _CMP_UNORD_Q );

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

		// Store result.
		_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pf16Dst), _mm512_cvtepi32_epi16( mui16Tmp ) );
	}

#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
	/**
	 * Loads 8 float16 values from the source array, converts them to float32,
	 *  and stores the results in the destination array.
	 *
	 * \param _pf16Src Pointer to the source array containing 8 float16 values.
	 * \return Returns a register with the loaded values.
	 */
	inline __m256 float16::Convert8Float16ToFloat32( const float16 * _pf16Src ) {
		// Load 8 uint16_t values (16 bytes).
		__m128i halfVec = _mm_loadu_si128( reinterpret_cast<const __m128i *>(_pf16Src) );

		// Convert the lower four half floats to 32-bit.
		__m128i lower16 = _mm_cvtepu16_epi32( halfVec );

		// Shift right by 8 bytes to access the upper four half floats.
		__m128i upperHalf = _mm_srli_si128( halfVec, 8 );
		__m128i upper16 = _mm_cvtepu16_epi32( upperHalf );

		// Combine the lower and upper conversions into a single __m256i.
		__m256i mui32Val = _mm256_set_m128i( upper16, lower16 );

		// Extract sign, exponent, and mantissa.
		__m256i mSign = _mm256_slli_epi32( _mm256_and_si256( mui32Val, _mm256_set1_epi32( 0x8000 ) ), 16 );
		__m256i mExpo = _mm256_and_si256( mui32Val, _mm256_set1_epi32( 0x7C00 ) );
		__m256i mMant = _mm256_and_si256( mui32Val, _mm256_set1_epi32( 0x03FF ) );

		// Shift mantissa to float32 position.
		__m256i mMantShifted = _mm256_slli_epi32( mMant, 13 );

		// Convert mantissa to float to compute leading zeros.
		__m256 mfManTmp = _mm256_cvtepi32_ps( mMantShifted );
		__m256i mfManTmpBits = _mm256_castps_si256( mfManTmp );
		__m256i mui32LdZ0 = _mm256_srli_epi32( mfManTmpBits, 23 );

		// Create masks (normal, subnormal, zero, inf, nan).
		__m256i zero = _mm256_setzero_si256();
		__m256i expoEqZero = _mm256_cmpeq_epi32( mExpo, zero );
		__m256i mantEqZero = _mm256_cmpeq_epi32( mMant, zero );
		__m256i mNormal = mm256_cmpneq_epi32( mExpo, zero );
		__m256i mSubnormal = _mm256_and_si256( expoEqZero, mm256_cmpneq_epi32( mMant, zero ) );
		__m256i mZero = _mm256_and_si256( expoEqZero, mantEqZero );

		__m256i expo7C00 = _mm256_set1_epi32( 0x7C00 );
		__m256i mInf = _mm256_cmpeq_epi32( mExpo, expo7C00 );
		__m256i mNan = _mm256_and_si256( mInf, mm256_cmpneq_epi32( mMant, zero ) );

		// Normalized numbers.
		__m256i adjExpo = _mm256_add_epi32( _mm256_srli_epi32( mExpo, 10 ), _mm256_set1_epi32( 112 ) );
		__m256i mExpoNormal = _mm256_slli_epi32( adjExpo, 23 );
		__m256i mui32TmpNormal = _mm256_or_si256( mSign, _mm256_or_si256( mExpoNormal, mMantShifted ) );

		// Subnormal numbers.
		__m256i mui32LdZ0Minus37 = _mm256_sub_epi32( mui32LdZ0, _mm256_set1_epi32( 37 ) );
		__m256i mExpoSubnormal = _mm256_slli_epi32( mui32LdZ0Minus37, 23 );
		__m256i mShiftAmount = _mm256_sub_epi32( _mm256_set1_epi32( 150 ), mui32LdZ0 );
		__m256i mMantSubnormalMask = _mm256_set1_epi32( 0x007FE000 );
		__m256i mMantSubnormal = _mm256_and_si256( _mm256_sllv_epi32( mMantShifted, mShiftAmount ), mMantSubnormalMask );
		__m256i mui32TmpSubnormal = _mm256_or_si256( mSign, _mm256_or_si256( mExpoSubnormal, mMantSubnormal ) );

		// Initialize result.
		__m256i mui32Tmp = zero;

		// Apply conditions using blend.
		mui32Tmp = mm256_blendv_epi32( mui32Tmp, mui32TmpNormal, mNormal );
		mui32Tmp = mm256_blendv_epi32( mui32Tmp, mui32TmpSubnormal, mSubnormal );
		mui32Tmp = mm256_blendv_epi32( mui32Tmp, mSign, mZero );

		// NaNs: or with 0x7FC00000.
		__m256i nanVal = _mm256_set1_epi32( 0x7FC00000 );
		mui32Tmp = mm256_blendv_epi32( mui32Tmp, _mm256_or_si256( mui32Tmp, nanVal ), mNan );

		// Infinities.
		__m256i notNan = mm256_not_si256( mNan );
		__m256i m_inf_only = _mm256_and_si256( notNan, mInf );
		__m256i infVal = _mm256_or_si256( mSign, _mm256_set1_epi32( 0x7F800000 ) );
		mui32Tmp = mm256_blendv_epi32( mui32Tmp, infVal, m_inf_only );

		// Convert to float32.
		return _mm256_castsi256_ps( mui32Tmp );
	}

	/**
	 * Takes 8 float32 values from the source array, converts them to float16,
	 *	and stores the results in the destination array as uint16_t.
	 *
	 * \param _pfSrc Pointer to the source array containing 16 float32 values.
	 * \return Returns a register with the loaded values.
	 */
	inline void float16::Convert8Float32ToFloat16( float16 * _pf16Dst, __m256 _mf32Val ) {
		// Reinterpret as integers and add rounding bias.
		__m256i mBits = _mm256_castps_si256( _mf32Val );
		__m256i mBitsRounded = _mm256_add_epi32( mBits, _mm256_set1_epi32( 0x00001000 ) );

		// Extract sign, exponent, mantissa.
		__m256i mSign = _mm256_srli_epi32( _mm256_and_si256( mBitsRounded, _mm256_set1_epi32( 0x80000000 ) ), 16 );
		__m256i mExpo = _mm256_srli_epi32( _mm256_and_si256( mBitsRounded, _mm256_set1_epi32( 0x7F800000 ) ), 23 );
		__m256i mMant = _mm256_and_si256( mBitsRounded, _mm256_set1_epi32( 0x007FFFFF ) );

		// Compute masks for comparisons.
		__m256i mExpoGt112 = _mm256_cmpgt_epi32( mExpo, _mm256_set1_epi32( 112 ) );
		__m256i mExpoGt101 = _mm256_cmpgt_epi32( mExpo, _mm256_set1_epi32( 101 ) );
		__m256i mExpoGt143 = _mm256_cmpgt_epi32( mExpo, _mm256_set1_epi32( 143 ) );
		__m256i mExpoLe101 = mm256_cmple_epi32( mExpo, _mm256_set1_epi32( 101 ) );

		// NaN detection: NaN != NaN.
		__m256 nan_mask_ps = _mm256_cmp_ps( _mf32Val, _mf32Val, _CMP_UNORD_Q );
		__m256i mIsNan = _mm256_castps_si256( nan_mask_ps );

		// Compute normalized numbers.
		__m256i mNorm = _mm256_slli_epi32( _mm256_sub_epi32( mExpo, _mm256_set1_epi32( 112 ) ), 10 );
		mNorm = _mm256_and_si256( mNorm, _mm256_set1_epi32( 0x7C00 ) );
		mNorm = _mm256_or_si256( mNorm, _mm256_srli_epi32( mMant, 13 ) );
		mNorm = _mm256_or_si256( mNorm, mSign );

		// Compute subnormal numbers.
		__m256i mMantSubnorm = _mm256_add_epi32( mMant, _mm256_set1_epi32( 0x007FF000 ) );
		__m256i mShift = _mm256_sub_epi32( _mm256_set1_epi32( 125 ), mExpo );
		// Shift right by mShift.
		// For variable shifts in AVX2, use _mm256_srlv_epi32.
		__m256i mSubnormTemp = _mm256_srlv_epi32( mMantSubnorm, mShift );
		mSubnormTemp = _mm256_srli_epi32( _mm256_add_epi32( mSubnormTemp, _mm256_set1_epi32( 1 ) ), 1 );
		mSubnormTemp = _mm256_and_si256( mSubnormTemp, _mm256_set1_epi32( 0x03FF ) );
		__m256i mSubnorm = _mm256_or_si256( mSubnormTemp, mSign );

		// Special cases (NaN and Infinity).
		__m256i mSpecial = _mm256_set1_epi32( 0x7FFF );

		// Initialize result.
		__m256i mui16Tmp = _mm256_setzero_si256();

		// Conditions:
		// Infinity and NaN check:
		// We'll fix these after setting normal cases

		// Special cases: Exponent > 143.
		mui16Tmp = mm256_blendv_epi32( mui16Tmp, mSpecial, mExpoGt143 );

		// Normalized numbers: Exponent > 112 and not special.
		// Already handled special if expo > 143, so just ensure that.
		// normal means expo > 112 and expo ≤ 143.
		__m256i not_mExpoGt143 = mm256_not_si256( mExpoGt143 );
		__m256i mNormMask = _mm256_and_si256( mExpoGt112, not_mExpoGt143 );
		mui16Tmp = mm256_blendv_epi32( mui16Tmp, mNorm, mNormMask );

		// Subnormal: 101 < Exponent ≤ 112.
		// This means (expo > 101) & (expo ≤ 112), and not special.
		__m256i not_mExpoGt112 = mm256_not_si256( mExpoGt112 );
		__m256i mSubnormMask = _mm256_and_si256( _mm256_and_si256( not_mExpoGt112, mExpoGt101 ), not_mExpoGt143 );
		mui16Tmp = mm256_blendv_epi32( mui16Tmp, mSubnorm, mSubnormMask );

		// Zeros: Exponent ≤ 101.
		mui16Tmp = mm256_blendv_epi32( mui16Tmp, mSign, mExpoLe101 );

		// Infinity:
		// Identify infinity bits: if bits have 0x7C00 in exponent field and not NaN
		// Infinity after conversion: Check if upper 5 bits (exponent) = 0x7C00
		// We'll check final mui16Tmp bits:
		__m256i exp_mask = _mm256_and_si256( mui16Tmp, _mm256_set1_epi32( 0x7C00 ) );
		__m256i eq_7c00 = _mm256_cmpeq_epi32( exp_mask, _mm256_set1_epi32( 0x7C00 ) );
		__m256i not_nan = mm256_not_si256( mIsNan );
		__m256i mIsInfMask = _mm256_and_si256( eq_7c00, not_nan );

		__m256i infVal = _mm256_or_si256( mSign, _mm256_set1_epi32( 0x7C00 ) );
		mui16Tmp = mm256_blendv_epi32( mui16Tmp, infVal, mIsInfMask );

		// For NaN, set to 0x7FFF.
		mui16Tmp = mm256_blendv_epi32( mui16Tmp, _mm256_set1_epi32( 0x7FFF ), mIsNan );


		// Extract the lower and upper 128 bits.
		__m128i low128 = _mm256_castsi256_si128( mui16Tmp );
		__m128i high128 = _mm256_extracti128_si256( mui16Tmp, 1 );

		// Now pack these 8 int32 values (4 in low128, 4 in high128) into 8 int16 values.
		__m128i result_16 = _mm_packus_epi32( low128, high128 );

		// Store result (8 uint16_t).
		_mm_storeu_si128( reinterpret_cast<__m128i *>(_pf16Dst), result_16 );
	}

#endif	// #ifdef __AVX2__


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

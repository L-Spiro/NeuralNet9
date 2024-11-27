/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A software implementation of bfloat16.  Can be seamlessly swapped out for hardware-supported bfloat16.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <type_traits>

namespace nn9 {

	/**
	 * Class bfloat16
	 * \brief A software implementation of bfloat16.
	 *
	 * Description: A software implementation of bfloat16.  Can be seamlessly swapped out for hardware-supported bfloat16.
	 */
	class bfloat16 {
	public :
		constexpr bfloat16() {}
		//constexpr explicit bfloat16( uint16_t _ui16Bits ) : m_u16Value( _ui16Bits ) {}

		bfloat16( float _fValue ) {
			// Truncate the float to 16-bit by discarding the lower 16 bits.
#if 1
			// Benchmark against (1000000*5000) values.
			// Hi:	7.06102
			// Lo:	6.86468
			// Av:	6.885046666666666
			struct s {
				uint16_t						ui16Low;
				uint16_t						ui16High;
			};
			m_u16Value = (*reinterpret_cast<s *>(&_fValue)).ui16High;
			/*
			 *	00007FF62AB83E6E  movd        ebx,xmm0  
			 *	00007FF62AB83E72  shr         ebx,10h 
			 */
#elif 0
			// Benchmark against (1000000*5000) values.
			// Hi:	6.99877
			// Lo:	6.88991
			// Av:	6.955083333333333
			union {
				struct {
					uint16_t					ui16Low;
					uint16_t					ui16High;
				};
				float							fVal;
			} uTmp;
			uTmp.fVal = _fValue;
			m_u16Value = uTmp.ui16High;

			/*
			 *	00007FF6609F3E6E  movd        ebx,xmm0  
			 *	00007FF6609F3E72  shr         ebx,10h  
			 */
#else
			// Benchmark against (1000000*5000) values.
			// Hi:	6.92265
			// Lo:	6.81515
			// Av:	6.89350333333333
			m_u16Value = static_cast<uint16_t>((*reinterpret_cast<uint32_t *>(&_fValue)) >> 16);

			/*
			 *	00007FF6609F3E6E  movd        ebx,xmm0  
			 *	00007FF6609F3E72  shr         ebx,10h  
			 */
#endif
		}

		template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
		bfloat16( T tVal ) {
			static_assert( sizeof( uint32_t ) == sizeof( float ) );

			float fValue = float( tVal );
			// Truncate the float to 16-bit by discarding the lower 16 bits.
#if 1
			struct s {
				uint16_t						ui16Low;
				uint16_t						ui16High;
			};
			m_u16Value = (*reinterpret_cast<s *>(&fValue)).ui16High;
#elif 0
			union {
				struct {
					uint16_t					ui16Low;
					uint16_t					ui16High;
				};
				float							fVal;
			} uTmp;
			uTmp.fVal = fValue;
			m_u16Value = uTmp.ui16High;
#else
			m_u16Value = static_cast<uint16_t>((*reinterpret_cast<uint32_t *>(&fValue)) >> 16);
#endif
		}
		

		// == Operators.
		/**
		 * Cast to float.
		 * 
		 * \return Returns the float value of the bfloat16.
		 **/
		inline operator							float() const {
			uint32_t ui32Val = static_cast<uint32_t>(m_u16Value) << 16;
			return (*reinterpret_cast<float *>(&ui32Val));
		}

		// ===============================
		// Arithmetic Operators
		// ===============================
		inline bfloat16							operator + ( const bfloat16 &_fOther ) const {
			return bfloat16( static_cast<float>((*this)) + static_cast<float>(_fOther) );
		}

		inline bfloat16							operator - ( const bfloat16 &_fOther ) const {
			return bfloat16( static_cast<float>((*this)) - static_cast<float>(_fOther) );
		}

		inline bfloat16							operator * ( const bfloat16 &_fOther ) const {
			return bfloat16( static_cast<float>((*this)) * static_cast<float>(_fOther) );
		}

		inline bfloat16							operator / ( const bfloat16 &_fOther ) const {
			return bfloat16( static_cast<float>((*this)) / static_cast<float>(_fOther) );
		}

		// ===============================
		// Compound Assignment Operators
		// ===============================
		inline bfloat16 &						operator += ( const bfloat16 &_fOther ) {
			(*this) = (*this) + _fOther;
			return (*this);
		}

		inline bfloat16 &						operator -= ( const bfloat16 &_fOther ) {
			(*this) = (*this) - _fOther;
			return (*this);
		}

		inline bfloat16 &						operator *= ( const bfloat16 &_fOther ) {
			(*this) = (*this) * _fOther;
			return (*this);
		}

		inline bfloat16 &						operator /= ( const bfloat16 &_fOther ) {
			(*this) = (*this) / _fOther;
			return (*this);
		}

		// ===============================
		// Comparison Operators
		// ===============================
		inline bool								operator == ( const bfloat16 &_fOther ) const {
			return static_cast<float>((*this)) == static_cast<float>(_fOther);
		}

		inline bool								operator != ( const bfloat16 &_fOther ) const {
			return !((*this) == _fOther);
		}

		inline bool								operator < ( const bfloat16 &_fOther ) const {
			return static_cast<float>((*this)) < static_cast<float>(_fOther);
		}

		inline bool								operator > ( const bfloat16 &_fOther ) const {
			return static_cast<float>((*this)) > static_cast<float>(_fOther);
		}

		inline bool								operator <= ( const bfloat16 &_fOther ) const {
			return static_cast<float>((*this)) <= static_cast<float>(_fOther);
		}

		inline bool								operator >= ( const bfloat16 &_fOther ) const {
			return static_cast<float>((*this)) >= static_cast<float>(_fOther);
		}


		// ===============================
		// Numeric Limits
		// ===============================
		/**
		 * Returns the smallest positive normalized value.
		 * 
		 * \return Returns the smallest positive normalized value.
		 **/
		static constexpr bfloat16				min() noexcept {
			return FromBits( 0x0080 );			// Exponent = 1, Mantissa = 0.
		}

		/**
		 * Returns the largest finite value.
		 * 
		 * \return Returns the largest finite value.
		 **/
		static constexpr bfloat16				max() noexcept {
			return FromBits( 0x7F7F );			// Exponent = 254, Mantissa = all ones.
		}

		/**
		 * Returns the lowest finite value (most negative).
		 * 
		 * \return Returns the lowest finite value (most negative).
		 **/
		static constexpr bfloat16				lowest() noexcept {
			return FromBits( 0xFF7F );			// Sign bit set, exponent = 256, Mantissa = all ones.
		}

		/**
		 * Difference between 1 and the next representable value.
		 * 
		 * \return Difference between 1 and the next representable value.
		 **/
		static constexpr bfloat16				epsilon() noexcept {
			return FromBits( 0x3C00 );			// Exponent = 120, Mantissa = 0.
		}

		/**
		 * Smallest positive subnormal value.
		 * 
		 * \return Smallest positive subnormal value.
		 **/
		static constexpr bfloat16				denorm_min() noexcept {
			return FromBits( 0x0001 );			// Exponent = 0, Mantissa = 1.
		}

		/**
		 * Returns positive infinity.
		 * 
		 * \return Returns positive infinity.
		 **/
		static constexpr bfloat16				infinity() noexcept {
			return FromBits( 0x7F80 );			// Exponent = 255 (all ones), Mantissa = 0.
		}

		/**
		 * Returns a quiet NaN.
		 * 
		 * \return Returns a quiet NaN.
		 **/
		static constexpr bfloat16				quiet_NaN() noexcept {
			return FromBits( 0x7FC0 );			// Exponent = 255, Mantissa = non-zero.
		}

		/**
		 * Returns a signaling NaN.
		 * 
		 * \return Returns a signaling NaN.
		 **/
		static constexpr bfloat16				signaling_NaN() noexcept {
			return FromBits( 0x7FA0 );			// Exponent = 255, Mantissa = specific pattern.
		}

		/**
		 * Utility function to create bfloat16 from raw bits.
		 * 
		 * \param _ui16Bits The 16-bit representation to express as a bfloat16.
		 * \return Returns the 16-bit value expressed as a bfloat16.
		 **/
		static constexpr bfloat16				FromBits( uint16_t _ui16Bits ) noexcept {
			bfloat16 f16This;
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


		// == Intrinsics.
#ifdef __AVX512F__
		// ===============================
		// Storage
		// ===============================
		/**
		 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
		 *
		 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
		 * \return A __m512 vector containing the converted single-precision floating-point values.
		 **/
		static inline __m512					loadu_bf16_to_fp32( const uint16_t * _pBF16 );

		/**
		 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
		 *
		 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
		 * \return A __m512 vector containing the converted single-precision floating-point values.
		 **/
		static inline __m512					load_bf16_to_fp32( const uint16_t * _pBF16 );

		/**
		 * Load 32 bfloat16 values from memory into a __m512bh vector.
		 *
		 * \param _pBF16 Pointer to the memory containing 32 bfloat16 values.
		 * \return A __m512bh vector containing the loaded bfloat16 values.
		 **/
		static	inline __m512bh					loadu_bf16_to_m512bh( const uint16_t * _pBF16 );

		/**
		 * Load 32 bfloat16 values from memory into a __m512bh vector.
		 *
		 * \param _pBF16 Pointer to the memory containing 32 bfloat16 values.
		 * \return A __m512bh vector containing the loaded bfloat16 values.
		 **/
		static	inline __m512bh					load_bf16_to_m512bh( const uint16_t * _pBF16 );

		/**
		 * Store 16 floats from a __m512 vector to memory as 16 bfloat16 values.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512 vector containing 16 single-precision floats.
		 **/
		static inline void						storeu_fp32_to_bf16( uint16_t * _pDst, __m512 _mSrc );

		/**
		 * Store 16 floats from a __m512 vector to memory as 16 bfloat16 values.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512 vector containing 16 single-precision floats.
		 **/
		static inline void						store_fp32_to_bf16( uint16_t * _pDst, __m512 _mSrc );

		/**
		 * Store 32 bfloat16 values from a __m512bh vector to memory.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512bh vector containing 32 bfloat16 values.
		 **/
		static inline void						storeu_m512bh_to_bf16( uint16_t * _pDst, __m512bh _mSrc );

		/**
		 * Store 32 bfloat16 values from a __m512bh vector to memory.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512bh vector containing 32 bfloat16 values.
		 **/
		static inline void						store_m512bh_to_bf16( uint16_t * _pDst, __m512bh _mSrc );

		// ===============================
		// VCVTNE2PS2BF16
		// ===============================
		/**
		 * Emulated _mm_cvtne2ps_pbh: Convert two __m128 (float) vectors to a single __m128bh (bfloat16) vector.
		 *
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing the converted values from _mA and _mB.
		 **/
		static inline __m128bh					_mm_cvtne2ps_pbh_emu( __m128 _mA, __m128 _mB );

		/**
		 * Emulated _mm_mask_cvtne2ps_pbh: Masked conversion of two __m128 (float) vectors to a __m128bh (bfloat16) vector.
		 *
		 * \param _mSrc The source bfloat16 vector to merge with.
		 * \param _kMask The mask specifying which elements to convert.
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing converted and merged values based on the mask.
		 **/
		static inline __m128bh					_mm_mask_cvtne2ps_pbh_emu( __m128bh _mSrc, __mmask8 _kMask, __m128 _mA, __m128 _mB );

		/**
		 * Emulated _mm_maskz_cvtne2ps_pbh: Masked conversion of two __m128 (float) vectors to a __m128bh (bfloat16) vector with zero masking.
		 *
		 * \param _kMask The mask specifying which elements to convert; others are set to zero.
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
		 **/
		static inline __m128bh					_mm_maskz_cvtne2ps_pbh_emu( __mmask8 _kMask, __m128 _mA, __m128 _mB );

		/**
		 * Emulated _mm256_cvtne2ps_pbh: Convert two __m256 (float) vectors to a single __m256bh (bfloat16) vector.
		 *
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing the converted values from _mA and _mB.
		 **/
		static inline __m256bh					_mm256_cvtne2ps_pbh_emu( __m256 _mA, __m256 _mB );

		/**
		 * Emulated _mm256_mask_cvtne2ps_pbh: Masked conversion of two __m256 (float) vectors to a __m256bh (bfloat16) vector.
		 *
		 * \param _mSrc The source bfloat16 vector to merge with.
		 * \param _kMask The mask specifying which elements to convert.
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing converted and merged values based on the mask.
		 **/
		static inline __m256bh					_mm256_mask_cvtne2ps_pbh_emu( __m256bh _mSrc, __mmask16 _kMask, __m256 _mA, __m256 _mB );

		/**
		 * Emulated _mm256_maskz_cvtne2ps_pbh: Masked conversion of two __m256 (float) vectors to a __m256bh (bfloat16) vector with zero masking.
		 *
		 * \param _kMask The mask specifying which elements to convert; others are set to zero.
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
		 **/
		static inline __m256bh					_mm256_maskz_cvtne2ps_pbh_emu( __mmask16 _kMask, __m256 _mA, __m256 _mB );

		/**
		 * Emulated _mm512_cvtne2ps_pbh: Convert two __m512 (float) vectors to a single __m512bh (bfloat16) vector.
		 *
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing the converted values from _mA and _mB.
		 **/
		static inline __m512bh					_mm512_cvtne2ps_pbh_emu( __m512 _mA, __m512 _mB );

		/**
		 * Emulated _mm512_mask_cvtne2ps_pbh: Masked conversion of two __m512 (float) vectors to a __m512bh (bfloat16) vector.
		 *
		 * \param _mSrc The source bfloat16 vector to merge with.
		 * \param _kMask The mask specifying which elements to convert.
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing converted and merged values based on the mask.
		 **/
		static inline __m512bh					_mm512_mask_cvtne2ps_pbh_emu( __m512bh _mSrc, __mmask32 _kMask, __m512 _mA, __m512 _mB );

		/**
		 * Emulated _mm512_maskz_cvtne2ps_pbh: Masked conversion of two __m512 (float) vectors to a __m512bh (bfloat16) vector with zero masking.
		 *
		 * \param _kMask The mask specifying which elements to convert; others are set to zero.
		 * \param _mA The first float vector to convert.
		 * \param _mB The second float vector to convert.
		 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
		 **/
		static inline __m512bh					_mm512_maskz_cvtne2ps_pbh_emu( __mmask32 _kMask, __m512 _mA, __m512 _mB );

		// ===============================
		// VCVTNEPS2BF16
		// ===============================
		/**
		 * Emulated _mm_cvtneps_pbh: Convert __m128 (float) to __m128bh (bfloat16).
		 *
		 * \param _mA The float vector to convert to bfloat16.
		 * \return A bfloat16 vector containing the converted values from _mA.
		 **/
		static inline __m128bh					_mm_cvtneps_pbh_emu( __m128 _mA );

		/**
		 * Emulated _mm_mask_cvtneps_pbh: Masked conversion of __m128 (float) to __m128bh (bfloat16).
		 *
		 * \param _mSrc The source bfloat16 vector to merge with.
		 * \param _kMask The 8-bit mask specifying which elements to convert.
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing converted and merged values based on the mask.
		 **/
		static inline __m128bh					_mm_mask_cvtneps_pbh_emu( __m128bh _mSrc, __mmask8 _kMask, __m128 _mA );

		/**
		 * Emulated _mm_maskz_cvtneps_pbh: Masked conversion of __m128 (float) to __m128bh (bfloat16) with zero masking.
		 *
		 * \param _kMask The 8-bit mask specifying which elements to convert; others are set to zero.
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
		 **/
		static inline __m128bh					_mm_maskz_cvtneps_pbh_emu( __mmask8 _kMask, __m128 _mA );

		/**
		 * Emulated _mm256_cvtneps_pbh: Convert __m256 (float) to __m128bh (bfloat16).
		 *
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing the converted values from _mA.
		 **/
		static inline __m128bh					_mm256_cvtneps_pbh_emu( __m256 _mA );

		/**
		 * Emulated _mm256_mask_cvtneps_pbh: Masked conversion of __m256 (float) to __m128bh (bfloat16).
		 *
		 * \param _mSrc The source bfloat16 vector to merge with.
		 * \param _kMask The 8-bit mask specifying which elements to convert.
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing converted and merged values based on the mask.
		 **/
		static inline __m128bh					_mm256_mask_cvtneps_pbh_emu( __m128bh _mSrc, __mmask8 _kMask, __m256 _mA );

		/**
		 * Emulated _mm256_maskz_cvtneps_pbh: Masked conversion of __m256 (float) to __m128bh (bfloat16) with zero masking.
		 *
		 * \param _kMask The 8-bit mask specifying which elements to convert; others are set to zero.
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
		 **/
		static inline __m128bh					_mm256_maskz_cvtneps_pbh_emu( __mmask8 _kMask, __m256 _mA );

		/**
		 * Emulated _mm512_cvtneps_pbh: Convert __m512 (float) to __m256bh (bfloat16).
		 *
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing the converted values from _mA.
		 **/
		static inline __m256bh					_mm512_cvtneps_pbh_emu( __m512 _mA );

		/**
		 * Emulated _mm512_mask_cvtneps_pbh: Masked conversion of __m512 (float) to __m256bh (bfloat16).
		 *
		 * \param _mSrc The source bfloat16 vector to merge with.
		 * \param _kMask The 16-bit mask specifying which elements to convert.
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing converted and merged values based on the mask.
		 **/
		static inline __m256bh					_mm512_mask_cvtneps_pbh_emu( __m256bh _mSrc, __mmask16 _kMask, __m512 _mA );

		/**
		 * Emulated _mm512_maskz_cvtneps_pbh: Masked conversion of __m512 (float) to __m256bh (bfloat16) with zero masking.
		 *
		 * \param _kMask The 16-bit mask specifying which elements to convert; others are set to zero.
		 * \param _mA The float vector to convert.
		 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
		 **/
		static inline __m256bh					_mm512_maskz_cvtneps_pbh_emu( __mmask16 _kMask, __m512 _mA );

		// ===============================
		// VDPBF16PS
		// ===============================
		/**
		 * Emulated _mm_dpbf16_ps: Perform dot product of BF16 vectors with FP32 accumulation.
		 *
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator after performing the dot product of _mA and _mB.
		 **/
		static inline __m128					_mm_dpbf16_ps_emu( __m128 _mAcc, __m128bh _mA, __m128bh _mB );

		/**
		 * Emulated _mm_mask_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation.
		 *
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _kMask The 8-bit mask specifying which elements to update.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator after performing the masked dot product.
		 **/
		static inline __m128					_mm_mask_dpbf16_ps_emu( __m128 _mAcc, __mmask8 _kMask, __m128bh _mA, __m128bh _mB );

		/**
		 * Emulated _mm_maskz_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation and zero masking.
		 *
		 * \param _kMask The 8-bit mask specifying which elements to update; others are set to zero.
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator with masked elements set to zero.
		 **/
		static inline __m128					_mm_maskz_dpbf16_ps_emu( __mmask8 _kMask, __m128 _mAcc, __m128bh _mA, __m128bh _mB );

		/**
		 * Emulated _mm256_dpbf16_ps: Perform dot product of BF16 vectors with FP32 accumulation.
		 *
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator after performing the dot product.
		 **/
		static inline __m256					_mm256_dpbf16_ps_emu( __m256 _mAcc, __m256bh _mA, __m256bh _mB );

		/**
		 * Emulated _mm256_mask_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation.
		 *
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _kMask The 8-bit mask specifying which elements to update.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator after performing the masked dot product.
		 **/
		static inline __m256					_mm256_mask_dpbf16_ps_emu( __m256 _mAcc, __mmask8 _kMask, __m256bh _mA, __m256bh _mB );

		/**
		 * Emulated _mm256_maskz_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation and zero masking.
		 *
		 * \param _kMask The 8-bit mask specifying which elements to update; others are set to zero.
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator with masked elements set to zero.
		 **/
		static inline __m256					_mm256_maskz_dpbf16_ps_emu( __mmask8 _kMask, __m256 _mAcc, __m256bh _mA, __m256bh _mB );

		/**
		 * Emulated _mm512_dpbf16_ps: Perform dot product of BF16 vectors with FP32 accumulation.
		 *
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator after performing the dot product.
		 **/
		static inline __m512					_mm512_dpbf16_ps_emu( __m512 _mAcc, __m512bh _mA, __m512bh _mB );

		/**
		 * Emulated _mm512_mask_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation.
		 *
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _kMask The 16-bit mask specifying which elements to update.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator after performing the masked dot product.
		 **/
		static inline __m512					_mm512_mask_dpbf16_ps_emu( __m512 _mAcc, __mmask16 _kMask, __m512bh _mA, __m512bh _mB );

		/**
		 * Emulated _mm512_maskz_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation and zero masking.
		 *
		 * \param _kMask The 16-bit mask specifying which elements to update; others are set to zero.
		 * \param _mAcc The FP32 accumulator vector.
		 * \param _mA The first BF16 vector operand.
		 * \param _mB The second BF16 vector operand.
		 * \return The updated accumulator with masked elements set to zero.
		 **/
		static inline __m512					_mm512_maskz_dpbf16_ps_emu( __mmask16 _kMask, __m512 _mAcc, __m512bh _mA, __m512bh _mB );
#endif	// #ifdef __AVX512F__

		// == Members.
		/** The backing value. */
		uint16_t								m_u16Value;
	};


	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// DEFINITIONS
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// == Intrinsics.
#ifdef __AVX512F__
	// ===============================
	// Storage
	// ===============================
	/**
	 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
	 *
	 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
	 * \return A __m512 vector containing the converted single-precision floating-point values.
	 **/
	inline __m512 bfloat16::loadu_bf16_to_fp32( const uint16_t * _pBF16 ) {
		// Step 1: Load 16 bfloat16 values (16 * 16 bits = 256 bits) into a __m256i vector.
		__m256i mBF16 = _mm256_loadu_si256( reinterpret_cast<const __m256i *>(_pBF16) );

		// Step 2: Zero-extend the 16-bit bfloat16 values to 32-bit integers.
		__m512i mInt32 = _mm512_cvtepu16_epi32( mBF16 );

		// Step 3: Shift left by 16 bits to align bfloat16 bits to the upper 16 bits of 32-bit floats.
		__m512i mShifted = _mm512_slli_epi32( mInt32, 16 );

		// Step 4: Reinterpret the 32-bit integers as single-precision floats.
		return _mm512_castsi512_ps( mShifted );
	}

	/**
	 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
	 *
	 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
	 * \return A __m512 vector containing the converted single-precision floating-point values.
	 **/
	inline __m512 bfloat16::load_bf16_to_fp32( const uint16_t * _pBF16 ) {
		// Step 1: Load 16 bfloat16 values (16 * 16 bits = 256 bits) into a __m256i vector.
		__m256i mBF16 = _mm256_load_si256( reinterpret_cast<const __m256i *>(_pBF16) );

		// Step 2: Zero-extend the 16-bit bfloat16 values to 32-bit integers.
		__m512i mInt32 = _mm512_cvtepu16_epi32( mBF16 );

		// Step 3: Shift left by 16 bits to align bfloat16 bits to the upper 16 bits of 32-bit floats.
		__m512i mShifted = _mm512_slli_epi32( mInt32, 16 );

		// Step 4: Reinterpret the 32-bit integers as single-precision floats.
		return _mm512_castsi512_ps( mShifted );
	}

	/**
	 * Load 32 bfloat16 values from memory into a __m512bh vector.
	 *
	 * \param _pBF16 Pointer to the memory containing 32 bfloat16 values.
	 * \return A __m512bh vector containing the loaded bfloat16 values.
	 **/
	inline __m512bh bfloat16::loadu_bf16_to_m512bh( const uint16_t * _pBF16 ) {
		// Step 1: Load 512 bits (32 bfloat16 values) from memory into a __m512i vector.
		__m512i mBF16Int = _mm512_loadu_si512( reinterpret_cast<const __m512i *>(_pBF16) );

		// Step 2: Use a union to reinterpret __m512i as __m512bh.
		union {
			__m512i m512iVal;
			__m512bh m512bhVal;
		} mCast;
		mCast.m512iVal = mBF16Int;

		// Step 3: Return the __m512bh vector.
		return mCast.m512bhVal;
	}

	/**
	 * Load 32 bfloat16 values from memory into a __m512bh vector.
	 *
	 * \param _pBF16 Pointer to the memory containing 32 bfloat16 values.
	 * \return A __m512bh vector containing the loaded bfloat16 values.
	 **/
	inline __m512bh bfloat16::load_bf16_to_m512bh( const uint16_t * _pBF16 ) {
		// Step 1: Load 512 bits (32 bfloat16 values) from memory into a __m512i vector.
		__m512i mBF16Int = _mm512_load_si512( reinterpret_cast<const __m512i *>(_pBF16) );

		// Step 2: Use a union to reinterpret __m512i as __m512bh.
		union {
			__m512i m512iVal;
			__m512bh m512bhVal;
		} mCast;
		mCast.m512iVal = mBF16Int;

		// Step 3: Return the __m512bh vector.
		return mCast.m512bhVal;
	}

	/**
	 * Store 16 floats from a __m512 vector to memory as 16 bfloat16 values.
	 *
	 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	 * \param _mSrc The __m512 vector containing 16 single-precision floats.
	 **/
	inline void bfloat16::storeu_fp32_to_bf16( uint16_t * _pDst, __m512 _mSrc ) {
		// Step 1: Reinterpret the float bits as 32-bit integers.
		__m512i mIntRepr = _mm512_castps_si512( _mSrc );

		// Step 2: Truncate the lower 16 bits to get the bfloat16 representation.
		__m512i mBF16Int = _mm512_srli_epi32( mIntRepr, 16 );

		// Step 3: Pack the 32-bit integers into 16-bit integers.
		__m256i mPacked = _mm512_cvtepi32_epi16( mBF16Int );

		// Step 4: Store the 16 bfloat16 values to memory.
		_mm256_storeu_si256( reinterpret_cast<__m256i *>(_pDst), mPacked );
	}

	/**
	 * Store 16 floats from a __m512 vector to memory as 16 bfloat16 values.
	 *
	 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	 * \param _mSrc The __m512 vector containing 16 single-precision floats.
	 **/
	inline void bfloat16::store_fp32_to_bf16( uint16_t * _pDst, __m512 _mSrc ) {
		// Step 1: Reinterpret the float bits as 32-bit integers.
		__m512i mIntRepr = _mm512_castps_si512( _mSrc );

		// Step 2: Truncate the lower 16 bits to get the bfloat16 representation.
		__m512i mBF16Int = _mm512_srli_epi32( mIntRepr, 16 );

		// Step 3: Pack the 32-bit integers into 16-bit integers.
		__m256i mPacked = _mm512_cvtepi32_epi16( mBF16Int );

		// Step 4: Store the 16 bfloat16 values to memory.
		_mm256_store_si256( reinterpret_cast<__m256i *>(_pDst), mPacked );
	}

	/**
	 * Store 32 bfloat16 values from a __m512bh vector to memory.
	 *
	 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	 * \param _mSrc The __m512bh vector containing 32 bfloat16 values.
	 **/
	inline void bfloat16::storeu_m512bh_to_bf16( uint16_t * _pDst, __m512bh _mSrc ) {
		// Use a union to reinterpret __m512bh as __m512i.
		union {
			__m512bh m512bhVal;
			__m512i m512iVal;
		} mCast;
		mCast.m512bhVal = _mSrc;

		// Store the __m512i vector to memory as 32 bfloat16 values.
		_mm512_storeu_si512( reinterpret_cast<__m512i *>(_pDst), mCast.m512iVal );
	}

	/**
	 * Store 32 bfloat16 values from a __m512bh vector to memory.
	 *
	 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	 * \param _mSrc The __m512bh vector containing 32 bfloat16 values.
	 **/
	inline void bfloat16::store_m512bh_to_bf16( uint16_t * _pDst, __m512bh _mSrc ) {
		// Use a union to reinterpret __m512bh as __m512i.
		union {
			__m512bh m512bhVal;
			__m512i m512iVal;
		} mCast;
		mCast.m512bhVal = _mSrc;

		// Store the __m512i vector to memory as 32 bfloat16 values.
		_mm512_store_si512( reinterpret_cast<__m512i *>(_pDst), mCast.m512iVal );
	}

	// ===============================
	// VCVTNE2PS2BF16
	// ===============================
	/**
	 * Emulated _mm_cvtne2ps_pbh: Convert two __m128 (float) vectors to a single __m128bh (bfloat16) vector.
	 *
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing the converted values from _mA and _mB.
	 **/
	inline __m128bh bfloat16::_mm_cvtne2ps_pbh_emu( __m128 _mA, __m128 _mB ) {
		// Combine _mA and _mB into a __m256.
		__m256 mCombined = _mm256_castps128_ps256( _mA );
		mCombined = _mm256_insertf128_ps( mCombined, _mB, 1 );

		// Reinterpret as integers.
		__m256i mIntRepr = _mm256_castps_si256( mCombined );

		// Truncate lower 16 bits.
		__m256i mBF16Int = _mm256_srli_epi32( mIntRepr, 16 );

		// Pack the 32-bit integers into 16-bit integers.
		__m128i mPacked = _mm_packus_epi32( _mm256_castsi256_si128( mBF16Int ),
			_mm256_extractf128_si256( mBF16Int, 1 ) );

		// Use union to reinterpret __m128i as __m128bh.
		union {
			__m128i m128iVal;
			__m128bh m128bhVal;
		} mCast;
		mCast.m128iVal = mPacked;

		return mCast.m128bhVal;
	}

	/**
	 * Emulated _mm_mask_cvtne2ps_pbh: Masked conversion of two __m128 (float) vectors to a __m128bh (bfloat16) vector.
	 *
	 * \param _mSrc The source bfloat16 vector to merge with.
	 * \param _kMask The mask specifying which elements to convert.
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing converted and merged values based on the mask.
	 **/
	inline __m128bh bfloat16::_mm_mask_cvtne2ps_pbh_emu( __m128bh _mSrc, __mmask8 _kMask, __m128 _mA, __m128 _mB ) {
		__m128bh mResult = _mm_cvtne2ps_pbh_emu(_mA, _mB);

		// Use reinterpret_cast to get __m128i representations.
		__m128i mSrcInt = reinterpret_cast<__m128i &>(_mSrc);
		__m128i mResInt = reinterpret_cast<__m128i &>(mResult);

		// Apply the mask.
		__m128i mMasked = _mm_mask_mov_epi16( mSrcInt, _kMask, mResInt );

		// Reinterpret back to __m128bh.
		return reinterpret_cast<__m128bh &>(mMasked);
	}

	/**
	 * Emulated _mm_maskz_cvtne2ps_pbh: Masked conversion of two __m128 (float) vectors to a __m128bh (bfloat16) vector with zero masking.
	 *
	 * \param _kMask The mask specifying which elements to convert; others are set to zero.
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
	 **/
	inline __m128bh bfloat16::_mm_maskz_cvtne2ps_pbh_emu( __mmask8 _kMask, __m128 _mA, __m128 _mB ) {
		__m128bh mResult = _mm_cvtne2ps_pbh_emu( _mA, _mB );

		// Use reinterpret_cast to get __m128i representation.
		__m128i mResInt = reinterpret_cast<__m128i &>(mResult);

		// Apply the zero mask.
		__m128i mMasked = _mm_maskz_mov_epi16( _kMask, mResInt );

		// Reinterpret back to __m128bh.
		return reinterpret_cast<__m128bh &>(mMasked);
	}

	/**
	 * Emulated _mm256_cvtne2ps_pbh: Convert two __m256 (float) vectors to a single __m256bh (bfloat16) vector.
	 *
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing the converted values from _mA and _mB.
	 **/
	inline __m256bh bfloat16::_mm256_cvtne2ps_pbh_emu( __m256 _mA, __m256 _mB ) {
		// Combine _mA and _mB into a __m512.
		__m512 mCombined = _mm512_castps256_ps512( _mA );
		mCombined = _mm512_insertf32x8( mCombined, _mB, 1 );

		// Reinterpret as integers.
		__m512i mIntRepr = _mm512_castps_si512( mCombined );

		// Truncate lower 16 bits.
		__m512i mBF16Int = _mm512_srli_epi32( mIntRepr, 16 );

		// Pack 32-bit integers into 16-bit integers.
		__m256i mPacked = _mm512_cvtepi32_epi16( mBF16Int );

		// Use union to reinterpret __m256i as __m256bh.
		union {
			__m256i m256iVal;
			__m256bh m256bhVal;
		} mCast;
		mCast.m256iVal = mPacked;

		return mCast.m256bhVal;
	}

	/**
	 * Emulated _mm256_mask_cvtne2ps_pbh: Masked conversion of two __m256 (float) vectors to a __m256bh (bfloat16) vector.
	 *
	 * \param _mSrc The source bfloat16 vector to merge with.
	 * \param _kMask The mask specifying which elements to convert.
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing converted and merged values based on the mask.
	 **/
	inline __m256bh bfloat16::_mm256_mask_cvtne2ps_pbh_emu( __m256bh _mSrc, __mmask16 _kMask, __m256 _mA, __m256 _mB ) {
		__m256bh mResult = _mm256_cvtne2ps_pbh_emu(_mA, _mB);

		// Use reinterpret_cast to get __m256i representations.
		__m256i mSrcInt = reinterpret_cast<__m256i&>(_mSrc);
		__m256i mResInt = reinterpret_cast<__m256i&>(mResult);

		// Apply the mask.
		__m256i mMasked = _mm256_mask_mov_epi16(mSrcInt, _kMask, mResInt);

		// Reinterpret back to __m256bh.
		return reinterpret_cast<__m256bh&>(mMasked);
	}

	/**
	 * Emulated _mm256_maskz_cvtne2ps_pbh: Masked conversion of two __m256 (float) vectors to a __m256bh (bfloat16) vector with zero masking.
	 *
	 * \param _kMask The mask specifying which elements to convert; others are set to zero.
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
	 **/
	inline __m256bh bfloat16::_mm256_maskz_cvtne2ps_pbh_emu( __mmask16 _kMask, __m256 _mA, __m256 _mB ) {
		__m256bh mResult = _mm256_cvtne2ps_pbh_emu(_mA, _mB);

		// Use reinterpret_cast to get __m256i representation.
		__m256i mResInt = reinterpret_cast<__m256i&>(mResult);

		// Apply the zero mask.
		__m256i mMasked = _mm256_maskz_mov_epi16(_kMask, mResInt);

		// Reinterpret back to __m256bh.
		return reinterpret_cast<__m256bh&>(mMasked);
	}

	/**
	 * Emulated _mm512_cvtne2ps_pbh: Convert two __m512 (float) vectors to a single __m512bh (bfloat16) vector.
	 *
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing the converted values from _mA and _mB.
	 **/
	inline __m512bh bfloat16::_mm512_cvtne2ps_pbh_emu( __m512 _mA, __m512 _mB ) {
		// Reinterpret as integers.
		__m512i mIntReprA = _mm512_castps_si512( _mA );
		__m512i mIntReprB = _mm512_castps_si512( _mB );

		// Truncate lower 16 bits.
		__m512i mBF16IntA = _mm512_srli_epi32( mIntReprA, 16 );
		__m512i mBF16IntB = _mm512_srli_epi32( mIntReprB, 16 );

		// Pack 32-bit integers into 16-bit integers.
		__m512i mPacked = _mm512_packus_epi32( mBF16IntA, mBF16IntB );

		// Use union to reinterpret __m512i as __m512bh.
		union {
			__m512i m512iVal;
			__m512bh m512bhVal;
		} mCast;
		mCast.m512iVal = mPacked;

		return mCast.m512bhVal;
	}

	/**
	 * Emulated _mm512_mask_cvtne2ps_pbh: Masked conversion of two __m512 (float) vectors to a __m512bh (bfloat16) vector.
	 *
	 * \param _mSrc The source bfloat16 vector to merge with.
	 * \param _kMask The mask specifying which elements to convert.
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing converted and merged values based on the mask.
	 **/
	inline __m512bh bfloat16::_mm512_mask_cvtne2ps_pbh_emu( __m512bh _mSrc, __mmask32 _kMask, __m512 _mA, __m512 _mB ) {
		__m512bh mResult = _mm512_cvtne2ps_pbh_emu( _mA, _mB );

		// Use reinterpret_cast to get __m512i representations.
		__m512i mSrcInt = reinterpret_cast<__m512i &>(_mSrc);
		__m512i mResInt = reinterpret_cast<__m512i &>(mResult);

		// Apply the mask.
		__m512i mMasked = _mm512_mask_mov_epi16( mSrcInt, _kMask, mResInt );

		// Reinterpret back to __m512bh.
		return reinterpret_cast<__m512bh &>(mMasked);
	}

	/**
	 * Emulated _mm512_maskz_cvtne2ps_pbh: Masked conversion of two __m512 (float) vectors to a __m512bh (bfloat16) vector with zero masking.
	 *
	 * \param _kMask The mask specifying which elements to convert; others are set to zero.
	 * \param _mA The first float vector to convert.
	 * \param _mB The second float vector to convert.
	 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
	 **/
	inline __m512bh bfloat16::_mm512_maskz_cvtne2ps_pbh_emu( __mmask32 _kMask, __m512 _mA, __m512 _mB ) {
		__m512bh mResult = _mm512_cvtne2ps_pbh_emu( _mA, _mB );

		// Use reinterpret_cast to get __m512i representation.
		__m512i mResInt = reinterpret_cast<__m512i &>(mResult);

		// Apply the zero mask.
		__m512i mMasked = _mm512_maskz_mov_epi16( _kMask, mResInt );

		// Reinterpret back to __m512bh.
		return reinterpret_cast<__m512bh &>(mMasked);
	}

	// ===============================
	// VCVTNEPS2BF16
	// ===============================
	/**
	 * Emulated _mm_cvtneps_pbh: Convert __m128 (float) to __m128bh (bfloat16).
	 *
	 * \param _mA The float vector to convert to bfloat16.
	 * \return A bfloat16 vector containing the converted values from _mA.
	 **/
	inline __m128bh bfloat16::_mm_cvtneps_pbh_emu( __m128 _mA ) {
		// Reinterpret as integers.
		__m128i mIntRepr = _mm_castps_si128( _mA );

		// Truncate lower 16 bits.
		__m128i mBF16Int = _mm_srli_epi32( mIntRepr, 16 );

		// Pack 32-bit integers into 16-bit integers.
		__m128i mPacked = _mm_packus_epi32( mBF16Int, _mm_setzero_si128() );

		// Zero-extend to fill __m128bh (128 bits).
		__m128i mResult = _mm_unpacklo_epi16( mPacked, _mm_setzero_si128() );

		// Use union to reinterpret __m128i as __m128bh.
		union {
			__m128i m128iVal;
			__m128bh m128bhVal;
		} mCast;
		mCast.m128iVal = mResult;

		return mCast.m128bhVal;
	}

	/**
	 * Emulated _mm_mask_cvtneps_pbh: Masked conversion of __m128 (float) to __m128bh (bfloat16).
	 *
	 * \param _mSrc The source bfloat16 vector to merge with.
	 * \param _kMask The 8-bit mask specifying which elements to convert.
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing converted and merged values based on the mask.
	 **/
	inline __m128bh bfloat16::_mm_mask_cvtneps_pbh_emu( __m128bh _mSrc, __mmask8 _kMask, __m128 _mA ) {
		__m128bh mResult = _mm_cvtneps_pbh_emu( _mA );

		// Use reinterpret_cast to get __m128i representations.
		__m128i mSrcInt = reinterpret_cast<__m128i &>(_mSrc);
		__m128i mResInt = reinterpret_cast<__m128i &>(mResult);

		// Apply the mask.
		__m128i mMasked = _mm_mask_mov_epi16( mSrcInt, _kMask, mResInt );

		// Reinterpret back to __m128bh.
		return reinterpret_cast<__m128bh &>(mMasked);
	}

	/**
	 * Emulated _mm_maskz_cvtneps_pbh: Masked conversion of __m128 (float) to __m128bh (bfloat16) with zero masking.
	 *
	 * \param _kMask The 8-bit mask specifying which elements to convert; others are set to zero.
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
	 **/
	inline __m128bh bfloat16::_mm_maskz_cvtneps_pbh_emu( __mmask8 _kMask, __m128 _mA ) {
		__m128bh mResult = _mm_cvtneps_pbh_emu( _mA );

		// Use reinterpret_cast to get __m128i representation.
		__m128i mResInt = reinterpret_cast<__m128i &>(mResult);

		// Apply the zero mask.
		__m128i mMasked = _mm_maskz_mov_epi16( _kMask, mResInt );

		// Reinterpret back to __m128bh.
		return reinterpret_cast<__m128bh &>(mMasked);
	}

	/**
	 * Emulated _mm256_cvtneps_pbh: Convert __m256 (float) to __m128bh (bfloat16).
	 *
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing the converted values from _mA.
	 **/
	inline __m128bh bfloat16::_mm256_cvtneps_pbh_emu( __m256 _mA ) {
		// Reinterpret as integers.
		__m256i mIntRepr = _mm256_castps_si256( _mA );

		// Truncate lower 16 bits.
		__m256i mBF16Int = _mm256_srli_epi32( mIntRepr, 16 );

		// Pack 32-bit integers into 16-bit integers.
		__m128i mPacked = _mm256_cvtepi32_epi16( mBF16Int );

		// Use union to reinterpret __m128i as __m128bh.
		union {
			__m128i m128iVal;
			__m128bh m128bhVal;
		} mCast;
		mCast.m128iVal = mPacked;

		return mCast.m128bhVal;
	}

	/**
	 * Emulated _mm256_mask_cvtneps_pbh: Masked conversion of __m256 (float) to __m128bh (bfloat16).
	 *
	 * \param _mSrc The source bfloat16 vector to merge with.
	 * \param _kMask The 8-bit mask specifying which elements to convert.
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing converted and merged values based on the mask.
	 **/
	inline __m128bh bfloat16::_mm256_mask_cvtneps_pbh_emu( __m128bh _mSrc, __mmask8 _kMask, __m256 _mA ) {
		__m128bh mResult = _mm256_cvtneps_pbh_emu( _mA );

		// Use reinterpret_cast to get __m128i representations.
		__m128i mSrcInt = reinterpret_cast<__m128i &>(_mSrc);
		__m128i mResInt = reinterpret_cast<__m128i &>(mResult);

		// Apply the mask.
		__m128i mMasked = _mm_mask_mov_epi16( mSrcInt, _kMask, mResInt );

		// Reinterpret back to __m128bh.
		return reinterpret_cast<__m128bh &>(mMasked);
	}

	/**
	 * Emulated _mm256_maskz_cvtneps_pbh: Masked conversion of __m256 (float) to __m128bh (bfloat16) with zero masking.
	 *
	 * \param _kMask The 8-bit mask specifying which elements to convert; others are set to zero.
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
	 **/
	inline __m128bh bfloat16::_mm256_maskz_cvtneps_pbh_emu( __mmask8 _kMask, __m256 _mA ) {
		__m128bh mResult = _mm256_cvtneps_pbh_emu( _mA );

		// Use reinterpret_cast to get __m128i representation.
		__m128i mResInt = reinterpret_cast<__m128i &>(mResult);

		// Apply the zero mask.
		__m128i mMasked = _mm_maskz_mov_epi16(_kMask, mResInt);

		// Reinterpret back to __m128bh.
		return reinterpret_cast<__m128bh &>(mMasked);
	}

	/**
	 * Emulated _mm512_cvtneps_pbh: Convert __m512 (float) to __m256bh (bfloat16).
	 *
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing the converted values from _mA.
	 **/
	inline __m256bh bfloat16::_mm512_cvtneps_pbh_emu( __m512 _mA ) {
		// Reinterpret as integers.
		__m512i mIntRepr = _mm512_castps_si512( _mA );

		// Truncate lower 16 bits.
		__m512i mBF16Int = _mm512_srli_epi32( mIntRepr, 16 );

		// Pack 32-bit integers into 16-bit integers.
		__m256i mPacked = _mm512_cvtepi32_epi16( mBF16Int );

		// Use union to reinterpret __m256i as __m256bh.
		union {
			__m256i m256iVal;
			__m256bh m256bhVal;
		} mCast;
		mCast.m256iVal = mPacked;

		return mCast.m256bhVal;
	}

	/**
	 * Emulated _mm512_mask_cvtneps_pbh: Masked conversion of __m512 (float) to __m256bh (bfloat16).
	 *
	 * \param _mSrc The source bfloat16 vector to merge with.
	 * \param _kMask The 16-bit mask specifying which elements to convert.
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing converted and merged values based on the mask.
	 **/
	inline __m256bh bfloat16::_mm512_mask_cvtneps_pbh_emu( __m256bh _mSrc, __mmask16 _kMask, __m512 _mA ) {
		__m256bh mResult = _mm512_cvtneps_pbh_emu( _mA );

		// Use reinterpret_cast to get __m256i representations.
		__m256i mSrcInt = reinterpret_cast<__m256i &>(_mSrc);
		__m256i mResInt = reinterpret_cast<__m256i &>(mResult);

		// Apply the mask.
		__m256i mMasked = _mm256_mask_mov_epi16( mSrcInt, _kMask, mResInt );

		// Reinterpret back to __m256bh.
		return reinterpret_cast<__m256bh &>(mMasked);
	}

	/**
	 * Emulated _mm512_maskz_cvtneps_pbh: Masked conversion of __m512 (float) to __m256bh (bfloat16) with zero masking.
	 *
	 * \param _kMask The 16-bit mask specifying which elements to convert; others are set to zero.
	 * \param _mA The float vector to convert.
	 * \return A bfloat16 vector containing converted values where the mask is set, zero elsewhere.
	 **/
	inline __m256bh bfloat16::_mm512_maskz_cvtneps_pbh_emu( __mmask16 _kMask, __m512 _mA ) {
		__m256bh mResult = _mm512_cvtneps_pbh_emu( _mA );

		// Use reinterpret_cast to get __m256i representation.
		__m256i mResInt = reinterpret_cast<__m256i &>(mResult);

		// Apply the zero mask.
		__m256i mMasked = _mm256_maskz_mov_epi16( _kMask, mResInt );

		// Reinterpret back to __m256bh.
		return reinterpret_cast<__m256bh &>(mMasked);
	}

	// ===============================
	// VDPBF16PS
	// ===============================
	/**
	 * Emulated _mm_dpbf16_ps: Perform dot product of BF16 vectors with FP32 accumulation.
	 *
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator after performing the dot product of _mA and _mB.
	 **/
	inline __m128 bfloat16::_mm_dpbf16_ps_emu( __m128 _mAcc, __m128bh _mA, __m128bh _mB ) {
		// Reinterpret BF16 vectors as __m128i.
		__m128i mAInt = reinterpret_cast<__m128i &>(_mA);
		__m128i mBInt = reinterpret_cast<__m128i &>(_mB);

		// Zero-extend BF16 to 32 bits.
		__m256i mAInt32 = _mm256_cvtepu16_epi32( mAInt );
		__m256i mBInt32 = _mm256_cvtepu16_epi32( mBInt );

		// Shift left by 16 bits to restore FP32 format.
		__m256i mAInt32Shifted = _mm256_slli_epi32( mAInt32, 16 );
		__m256i mBInt32Shifted = _mm256_slli_epi32( mBInt32, 16 );

		// Reinterpret as __m256 (FP32).
		__m256 mAFP32 = _mm256_castsi256_ps( mAInt32Shifted );
		__m256 mBFP32 = _mm256_castsi256_ps( mBInt32Shifted );

		// Multiply elements.
		__m256 mMul = _mm256_mul_ps( mAFP32, mBFP32 );

		// Sum adjacent pairs.
		__m256 mShuf = _mm256_permute_ps( mMul, _MM_SHUFFLE( 2, 3, 0, 1 ) );
		__m256 mSum = _mm256_add_ps( mMul, mShuf );

		// Extract lower 128 bits.
		__m128 mSumPairs = _mm256_castps256_ps128( mSum );

		// Accumulate into _mAcc.
		__m128 mResult = _mm_add_ps( _mAcc, mSumPairs );

		return mResult;
	}

	/**
	 * Emulated _mm_mask_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation.
	 *
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _kMask The 8-bit mask specifying which elements to update.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator after performing the masked dot product.
	 **/
	inline __m128 bfloat16::_mm_mask_dpbf16_ps_emu( __m128 _mAcc, __mmask8 _kMask, __m128bh _mA, __m128bh _mB ) {
		__m128 mResult = _mm_dpbf16_ps_emu( _mAcc, _mA, _mB );

		// Apply the mask.
		return _mm_mask_mov_ps( _mAcc, _kMask, mResult );
	}

	/**
	 * Emulated _mm_maskz_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation and zero masking.
	 *
	 * \param _kMask The 8-bit mask specifying which elements to update; others are set to zero.
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator with masked elements set to zero.
	 **/
	inline __m128 bfloat16::_mm_maskz_dpbf16_ps_emu( __mmask8 _kMask, __m128 _mAcc, __m128bh _mA, __m128bh _mB ) {
		__m128 mResult = _mm_dpbf16_ps_emu( _mAcc, _mA, _mB );

		// Apply the zero mask.
		return _mm_maskz_mov_ps( _kMask, mResult );
	}

	/**
	 * Emulated _mm256_dpbf16_ps: Perform dot product of BF16 vectors with FP32 accumulation.
	 *
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator after performing the dot product.
	 **/
	inline __m256 bfloat16::_mm256_dpbf16_ps_emu( __m256 _mAcc, __m256bh _mA, __m256bh _mB ) {
		// Reinterpret BF16 vectors as __m256i.
		__m256i mAInt = reinterpret_cast<__m256i &>(_mA);
		__m256i mBInt = reinterpret_cast<__m256i &>(_mB);

		// Zero-extend BF16 to 32 bits.
		__m512i mAInt32 = _mm512_cvtepu16_epi32( mAInt );
		__m512i mBInt32 = _mm512_cvtepu16_epi32( mBInt );

		// Shift left by 16 bits to restore FP32 format.
		__m512i mAInt32Shifted = _mm512_slli_epi32( mAInt32, 16 );
		__m512i mBInt32Shifted = _mm512_slli_epi32( mBInt32, 16 );

		// Reinterpret as __m512 (FP32).
		__m512 mAFP32 = _mm512_castsi512_ps( mAInt32Shifted );
		__m512 mBFP32 = _mm512_castsi512_ps( mBInt32Shifted );

		// Multiply elements.
		__m512 mMul = _mm512_mul_ps( mAFP32, mBFP32 );

		// Sum adjacent pairs.
		__m512i _mIdx = _mm512_set_epi32( 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0 );
		__m512 mSum = _mm512_permutexvar_ps( _mIdx, mMul );
		mSum = _mm512_add_ps( mSum, mMul );

		// Extract lower 256 bits.
		__m256 mSumPairs = _mm512_castps512_ps256( mSum );

		// Accumulate into _mAcc.
		__m256 mResult = _mm256_add_ps( _mAcc, mSumPairs );

		return mResult;
	}

	/**
	 * Emulated _mm256_mask_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation.
	 *
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _kMask The 8-bit mask specifying which elements to update.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator after performing the masked dot product.
	 **/
	inline __m256 bfloat16::_mm256_mask_dpbf16_ps_emu( __m256 _mAcc, __mmask8 _kMask, __m256bh _mA, __m256bh _mB ) {
		__m256 mResult = _mm256_dpbf16_ps_emu( _mAcc, _mA, _mB );

		// Apply the mask.
		return _mm256_mask_mov_ps( _mAcc, _kMask, mResult );
	}

	/**
	 * Emulated _mm256_maskz_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation and zero masking.
	 *
	 * \param _kMask The 8-bit mask specifying which elements to update; others are set to zero.
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator with masked elements set to zero.
	 **/
	inline __m256 bfloat16::_mm256_maskz_dpbf16_ps_emu( __mmask8 _kMask, __m256 _mAcc, __m256bh _mA, __m256bh _mB ) {
		__m256 mResult = _mm256_dpbf16_ps_emu( _mAcc, _mA, _mB );

		// Apply the zero mask.
		return _mm256_maskz_mov_ps( _kMask, mResult );
	}

	/**
	 * Emulated _mm512_dpbf16_ps: Perform dot product of BF16 vectors with FP32 accumulation.
	 *
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator after performing the dot product.
	 **/
	inline __m512 bfloat16::_mm512_dpbf16_ps_emu( __m512 _mAcc, __m512bh _mA, __m512bh _mB ) {
		// Reinterpret BF16 vectors as __m512i.
		__m512i mAInt = reinterpret_cast<__m512i &>(_mA);
		__m512i mBInt = reinterpret_cast<__m512i &>(_mB);

		// Zero-extend BF16 to 32 bits.
		__m512i mAInt32_Lo = _mm512_cvtepu16_epi32( _mm512_castsi512_si256( mAInt ) );
		__m512i mAInt32_Hi = _mm512_cvtepu16_epi32( _mm512_extracti64x4_epi64( mAInt, 1 ) );
		__m512i mBInt32_Lo = _mm512_cvtepu16_epi32( _mm512_castsi512_si256( mBInt ) );
		__m512i mBInt32_Hi = _mm512_cvtepu16_epi32( _mm512_extracti64x4_epi64( mBInt, 1 ) );

		// Shift left by 16 bits to restore FP32 format.
		__m512i mAInt32Shifted_Lo = _mm512_slli_epi32( mAInt32_Lo, 16 );
		__m512i mAInt32Shifted_Hi = _mm512_slli_epi32( mAInt32_Hi, 16 );
		__m512i mBInt32Shifted_Lo = _mm512_slli_epi32( mBInt32_Lo, 16 );
		__m512i mBInt32Shifted_Hi = _mm512_slli_epi32( mBInt32_Hi, 16 );

		// Reinterpret as __m512 (FP32).
		__m512 mA_FP32_Lo = _mm512_castsi512_ps( mAInt32Shifted_Lo );
		__m512 mA_FP32_Hi = _mm512_castsi512_ps( mAInt32Shifted_Hi );
		__m512 mB_FP32_Lo = _mm512_castsi512_ps( mBInt32Shifted_Lo );
		__m512 mB_FP32_Hi = _mm512_castsi512_ps( mBInt32Shifted_Hi );

		// Multiply elements.
		__m512 mMul_Lo = _mm512_mul_ps( mA_FP32_Lo, mB_FP32_Lo );
		__m512 mMul_Hi = _mm512_mul_ps( mA_FP32_Hi, mB_FP32_Hi );

		// Sum adjacent pairs.
		__m512 mShuf_Lo = _mm512_permute_ps( mMul_Lo, _MM_SHUFFLE( 2, 3, 0, 1 ) );
		__m512 mSum_Lo = _mm512_add_ps( mMul_Lo, mShuf_Lo );
		__m512 mShuf_Hi = _mm512_permute_ps( mMul_Hi, _MM_SHUFFLE( 2, 3, 0, 1 ) );
		__m512 mSum_Hi = _mm512_add_ps( mMul_Hi, mShuf_Hi );

		// Combine results.
		__m512 mSumPairs = _mm512_castps256_ps512( _mm512_castps512_ps256( mSum_Lo ) );
		mSumPairs = _mm512_insertf32x8( mSumPairs, _mm512_castps512_ps256( mSum_Hi ), 1 );

		// Accumulate into _mAcc.
		return _mm512_add_ps( _mAcc, mSumPairs );
	}

	/**
	 * Emulated _mm512_mask_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation.
	 *
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _kMask The 16-bit mask specifying which elements to update.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator after performing the masked dot product.
	 **/
	inline __m512 bfloat16::_mm512_mask_dpbf16_ps_emu( __m512 _mAcc, __mmask16 _kMask, __m512bh _mA, __m512bh _mB ) {
		__m512 mResult = _mm512_dpbf16_ps_emu( _mAcc, _mA, _mB );

		// Apply the mask.
		return _mm512_mask_mov_ps( _mAcc, _kMask, mResult );
	}

	/**
	 * Emulated _mm512_maskz_dpbf16_ps: Masked dot product of BF16 vectors with FP32 accumulation and zero masking.
	 *
	 * \param _kMask The 16-bit mask specifying which elements to update; others are set to zero.
	 * \param _mAcc The FP32 accumulator vector.
	 * \param _mA The first BF16 vector operand.
	 * \param _mB The second BF16 vector operand.
	 * \return The updated accumulator with masked elements set to zero.
	 **/
	inline __m512 bfloat16::_mm512_maskz_dpbf16_ps_emu( __mmask16 _kMask, __m512 _mAcc, __m512bh _mA, __m512bh _mB ) {
		__m512 mResult = _mm512_dpbf16_ps_emu( _mAcc, _mA, _mB );

		// Apply the zero mask.
		return _mm512_maskz_mov_ps( _kMask, mResult );
	}

#define _mm_cvtne2ps_pbh						nn9::bfloat16::_mm_cvtne2ps_pbh_emu
#define _mm_mask_cvtne2ps_pbh					nn9::bfloat16::_mm_mask_cvtne2ps_pbh_emu
#define _mm_maskz_cvtne2ps_pbh					nn9::bfloat16::_mm_maskz_cvtne2ps_pbh_emu
#define _mm256_cvtne2ps_pbh						nn9::bfloat16::_mm256_cvtne2ps_pbh_emu
#define _mm256_mask_cvtne2ps_pbh				nn9::bfloat16::_mm256_mask_cvtne2ps_pbh_emu
#define _mm256_maskz_cvtne2ps_pbh				nn9::bfloat16::_mm256_maskz_cvtne2ps_pbh_emu
#define _mm512_cvtne2ps_pbh						nn9::bfloat16::_mm512_cvtne2ps_pbh_emu
#define _mm512_mask_cvtne2ps_pbh				nn9::bfloat16::_mm512_mask_cvtne2ps_pbh_emu
#define _mm512_maskz_cvtne2ps_pbh				nn9::bfloat16::_mm512_maskz_cvtne2ps_pbh_emu

#define _mm_cvtneps_pbh							nn9::bfloat16::_mm_cvtneps_pbh_emu
#define _mm_mask_cvtneps_pbh					nn9::bfloat16::_mm_mask_cvtneps_pbh_emu
#define _mm_maskz_cvtneps_pbh					nn9::bfloat16::_mm_maskz_cvtneps_pbh_emu
#define _mm256_cvtneps_pbh						nn9::bfloat16::_mm256_cvtneps_pbh_emu
#define _mm256_mask_cvtneps_pbh					nn9::bfloat16::_mm256_mask_cvtneps_pbh_emu
#define _mm256_maskz_cvtneps_pbh				nn9::bfloat16::_mm256_maskz_cvtneps_pbh_emu
#define _mm512_cvtneps_pbh						nn9::bfloat16::_mm512_cvtneps_pbh_emu
#define _mm512_mask_cvtneps_pbh					nn9::bfloat16::_mm512_mask_cvtneps_pbh_emu
#define _mm512_maskz_cvtneps_pbh				nn9::bfloat16::_mm512_maskz_cvtneps_pbh_emu

#define _mm_dpbf16_ps							nn9::bfloat16::_mm_dpbf16_ps_emu
#define _mm_mask_dpbf16_ps						nn9::bfloat16::_mm_mask_dpbf16_ps_emu
#define _mm_maskz_dpbf16_ps						nn9::bfloat16::_mm_maskz_dpbf16_ps_emu
#define _mm256_dpbf16_ps						nn9::bfloat16::_mm256_dpbf16_ps_emu
#define _mm256_mask_dpbf16_ps					nn9::bfloat16::_mm256_mask_dpbf16_ps_emu
#define _mm256_maskz_dpbf16_ps					nn9::bfloat16::_mm256_maskz_dpbf16_ps_emu
#define _mm512_dpbf16_ps						nn9::bfloat16::_mm512_dpbf16_ps_emu
#define _mm512_mask_dpbf16_ps					nn9::bfloat16::_mm512_mask_dpbf16_ps_emu
#define _mm512_maskz_dpbf16_ps					nn9::bfloat16::_mm512_maskz_dpbf16_ps_emu
#endif	// #ifdef __AVX512F__

#define bfloat16_t								nn9::bfloat16


}	// namespace nn9

namespace std {
    template<>
    struct is_floating_point<nn9::bfloat16> : std::true_type {};
	template<>
    struct is_arithmetic<nn9::bfloat16> : std::true_type {};
	template<>
    struct is_signed<nn9::bfloat16> : std::true_type {};
	template<>
    struct is_scalar<nn9::bfloat16> : std::true_type {};

	template<>
    struct is_standard_layout<nn9::bfloat16> : std::true_type {};
    template<>
    struct is_trivial<nn9::bfloat16> : std::true_type {};


	template<>
    class numeric_limits<nn9::bfloat16> {
    public:
        static constexpr bool is_specialized	= true;

        static constexpr nn9::bfloat16			min() noexcept { return nn9::bfloat16::min(); }
        static constexpr nn9::bfloat16			max() noexcept { return nn9::bfloat16::max(); }
        static constexpr nn9::bfloat16			lowest() noexcept { return nn9::bfloat16::lowest(); }

        static constexpr int digits				= 8;	// Mantissa bits including the implicit bit.
        static constexpr int digits10			= 2;	// floor(digits * log10(2)).
        static constexpr int max_digits10		= 4;	// ceil(1 + digits * log10(2)).

        static constexpr bool is_signed			= true;
        static constexpr bool is_integer		= false;
        static constexpr bool is_exact			= false;
        static constexpr int radix				= 2;

        static constexpr nn9::bfloat16			epsilon() noexcept { return nn9::bfloat16::epsilon(); }
        static constexpr nn9::bfloat16			round_error() noexcept { return nn9::bfloat16::FromBits( 0x3F00 ); }	// 0.5f.

        static constexpr int min_exponent		= std::numeric_limits<float>::min_exponent;
        static constexpr int min_exponent10		= std::numeric_limits<float>::min_exponent10;
        static constexpr int max_exponent		= std::numeric_limits<float>::max_exponent;
        static constexpr int max_exponent10		= std::numeric_limits<float>::max_exponent10;

        static constexpr bool has_infinity		= true;
        static constexpr bool has_quiet_NaN		= true;
        static constexpr bool has_signaling_NaN	= true;
        static constexpr std::float_denorm_style has_denorm = std::denorm_present;
        static constexpr bool has_denorm_loss	= false;

        static constexpr nn9::bfloat16			infinity() noexcept { return nn9::bfloat16::infinity(); }
        static constexpr nn9::bfloat16			quiet_NaN() noexcept { return nn9::bfloat16::quiet_NaN(); }
        static constexpr nn9::bfloat16			signaling_NaN() noexcept { return nn9::bfloat16::signaling_NaN(); }
        static constexpr nn9::bfloat16			denorm_min() noexcept { return nn9::bfloat16::denorm_min(); }

        static constexpr bool is_iec559			= true;
        static constexpr bool is_bounded		= true;
        static constexpr bool is_modulo			= false;

        static constexpr bool traps				= false;
        static constexpr bool tinyness_before	= false;
        static constexpr std::float_round_style	round_style = std::round_toward_zero;
    };


	template<>
    struct is_pod<nn9::bfloat16> : std::true_type {};


	template<>
    struct common_type<nn9::bfloat16, float> {
        using type = float;
    };

    template<>
    struct common_type<float, nn9::bfloat16> {
        using type = float;
    };

    template<>
    struct common_type<nn9::bfloat16, double> {
        using type = double;
    };

    template<>
    struct common_type<double, nn9::bfloat16> {
        using type = double;
    };


	template<>
    struct hash<nn9::bfloat16> {
        std::size_t								operator()( const nn9::bfloat16 &_bf16Val ) const noexcept {
            return std::hash<uint16_t>()( _bf16Val.ToBits() );
        }
    };

}

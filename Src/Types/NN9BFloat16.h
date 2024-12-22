/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A software implementation of bfloat16.  Can be seamlessly swapped out for hardware-supported bfloat16.
 */

#pragma once

#include "../Foundation/NN9Macros.h"

#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <type_traits>


namespace nn9 {

	class float16;

	/**
	 * Class bfloat16
	 * \brief A software implementation of bfloat16.
	 *
	 * Description: A software implementation of bfloat16.  Can be seamlessly swapped out for hardware-supported bfloat16.
	 */
	class bfloat16 {
	public :
		constexpr bfloat16() {}
		
		bfloat16( double _dValue ) {
			// Truncate the float to 16-bit by discarding the lower 16 bits.
			float fValue = float( _dValue );
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

		bfloat16( float _fValue ) {
			// Truncate the float to 16-bit by discarding the lower 16 bits.
			// Benchmark against (1000000*5000) values.
			// Hi:	7.06102
			// Lo:	6.86468
			// Av:	6.885046666666666
			struct s {
				uint16_t						ui16Low;
				uint16_t						ui16High;
			};
			m_u16Value = (*reinterpret_cast<s *>(&_fValue)).ui16High;
		}
		bfloat16( float16 );

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
		 * Casts to float.
		 * 
		 * \return Returns the float value of the bfloat16.
		 **/
		inline operator							float() const {
			uint32_t ui32Val = static_cast<uint32_t>(m_u16Value) << 16;
			return (*reinterpret_cast<float *>(&ui32Val));
		}

		/**
		 * Casts to double.
		 * 
		 * \return Returns the double value of the bfloat16.
		 **/
		inline operator							double() const {
			return static_cast<float>(*this);
		}

		/**
		 * Casts to an integer type.
		 * 
		 * \tparam T The integer type to which to cast this object.
		 * \return Returns an integer value of the bfloat16.
		 **/
		template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
		inline operator							T() const {
			return T( static_cast<float>(*this) );
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
		
		inline bfloat16							operator + ( double _dOther ) const {
			return bfloat16( static_cast<float>((*this)) + _dOther );
		}

		inline bfloat16							operator - ( double _dOther ) const {
			return bfloat16( static_cast<float>((*this)) - _dOther );
		}

		inline bfloat16							operator * ( double _dOther ) const {
			return bfloat16( static_cast<float>((*this)) * _dOther );
		}

		inline bfloat16							operator / ( double _dOther ) const {
			return bfloat16( static_cast<float>((*this)) / _dOther );
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
			return static_cast<float>((*this)) != static_cast<float>(_fOther);
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
		static inline __m512					loadu_bf16_to_fp32_16( const uint16_t * _pBF16 );

		/**
		 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
		 *
		 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
		 * \return A __m512 vector containing the converted single-precision floating-point values.
		 **/
		static inline __m512					load_bf16_to_fp32_16( const uint16_t * _pBF16 );

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

		/**
		 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
		 *
		 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
		 * \return A __m512 vector containing the converted single-precision floating-point values.
		 **/
		static inline __m512					loadu_bf16_to_fp32_16( const bfloat16 * _pBF16 ) { return loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
		 * Load 16 bfloat16 values from memory and convert them into a __m512 (16 floats).
		 *
		 * \param _pBF16 Pointer to the memory containing 16 bfloat16 values.
		 * \return A __m512 vector containing the converted single-precision floating-point values.
		 **/
		static inline __m512					load_bf16_to_fp32_16( const bfloat16 * _pBF16 ) { return load_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
		 * Load 32 bfloat16 values from memory into a __m512bh vector.
		 *
		 * \param _pBF16 Pointer to the memory containing 32 bfloat16 values.
		 * \return A __m512bh vector containing the loaded bfloat16 values.
		 **/
		static	inline __m512bh					loadu_bf16_to_m512bh( const bfloat16 * _pBF16 ) { return loadu_bf16_to_m512bh( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
		 * Load 32 bfloat16 values from memory into a __m512bh vector.
		 *
		 * \param _pBF16 Pointer to the memory containing 32 bfloat16 values.
		 * \return A __m512bh vector containing the loaded bfloat16 values.
		 **/
		static	inline __m512bh					load_bf16_to_m512bh( const bfloat16 * _pBF16 ) { return load_bf16_to_m512bh( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
		 * Store 16 floats from a __m512 vector to memory as 16 bfloat16 values.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512 vector containing 16 single-precision floats.
		 **/
		static inline void						storeu_fp32_to_bf16( bfloat16 * _pDst, __m512 _mSrc ) { storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
		 * Store 16 floats from a __m512 vector to memory as 16 bfloat16 values.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512 vector containing 16 single-precision floats.
		 **/
		static inline void						store_fp32_to_bf16( bfloat16 * _pDst, __m512 _mSrc ) { store_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
		 * Store 32 bfloat16 values from a __m512bh vector to memory.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512bh vector containing 32 bfloat16 values.
		 **/
		static inline void						storeu_m512bh_to_bf16( bfloat16 * _pDst, __m512bh _mSrc ) { storeu_m512bh_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
		 * Store 32 bfloat16 values from a __m512bh vector to memory.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m512bh vector containing 32 bfloat16 values.
		 **/
		static inline void						store_m512bh_to_bf16( bfloat16 * _pDst, __m512bh _mSrc ) { store_m512bh_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
		/**
		 * \brief Load 8 bfloat16 values from memory and convert them into a __m256 (8 floats) using AVX2.
		 *
		 * Each bfloat16 occupies 16 bits. 8 bfloat16 values = 8 * 2 bytes = 16 bytes total.
		 * We'll:
		 * - Load into a __m128i.
		 * - Unpack and zero-extend to 32 bits per element.
		 * - Shift left to position the bfloat16 bits correctly.
		 * - Reinterpret as floats.
		 *
		 * \param _pBF16 Pointer to memory containing 8 bfloat16 values.
		 * \return A __m256 vector containing 8 floats converted from bfloat16.
		 **/
		static inline __m256					loadu_bf16_to_fp32_8( const uint16_t * _pBF16 );

		/**
		 * \brief Load 8 bfloat16 values from memory and convert them into a __m256 (8 floats) using AVX2.
		 *
		 * Each bfloat16 occupies 16 bits. 8 bfloat16 values = 8 * 2 bytes = 16 bytes total.
		 * We'll:
		 * - Load into a __m128i.
		 * - Unpack and zero-extend to 32 bits per element.
		 * - Shift left to position the bfloat16 bits correctly.
		 * - Reinterpret as floats.
		 *
		 * \param _pBF16 Pointer to memory containing 8 bfloat16 values.
		 * \return A __m256 vector containing 8 floats converted from bfloat16.
		 **/
		static inline __m256					load_bf16_to_fp32_8( const uint16_t * _pBF16 );

		/**
	     * \brief Load 8 bfloat16 values from memory into a __m128i vector using AVX2.
	     *
	     * Since AVX2 does not support BF16 directly, we just treat them as 16-byte raw data.
	     * You can then process them as needed. This is not a perfect parallel to __m512bh,
	     * but it gives you a vector holding 8 bfloat16 values.
	     * 
	     * \param _pBF16 Pointer to the memory containing 8 bfloat16 values.
	     * \return A __m128i vector containing the loaded bfloat16 values.
	     */
		static inline __m128i					loadu_bf16_to_m128i( const uint16_t * _pBF16 );

		/**
	     * \brief Load 8 bfloat16 values from memory into a __m128i vector using AVX2.
	     *
	     * Since AVX2 does not support BF16 directly, we just treat them as 16-byte raw data.
	     * You can then process them as needed. This is not a perfect parallel to __m512bh,
	     * but it gives you a vector holding 8 bfloat16 values.
	     * 
	     * \param _pBF16 Pointer to the memory containing 8 bfloat16 values.
	     * \return A __m128i vector containing the loaded bfloat16 values.
	     */
		static inline __m128i					load_bf16_to_m128i( const uint16_t * _pBF16 );

		/**
		 * \brief Store 8 floats from a __m256 vector to memory as 8 bfloat16 values using AVX2.
		 *
		 * This method:
		 * - Reinterprets floats as 32-bit ints.
		 * - Shifts right by 16 to get the top 16 bits of each float.
		 * - Safely converts these to uint16_t by storing intermediate results to a temporary array.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m256 vector containing 8 single-precision floats.
		 */
		static inline void						storeu_fp32_to_bf16( uint16_t * _pDst, __m256 _mSrc );

		/**
		 * \brief Store 8 floats from a __m256 vector to memory as 8 bfloat16 values using AVX2.
		 *
		 * This method:
		 * - Reinterprets floats as 32-bit ints.
		 * - Shifts right by 16 to get the top 16 bits of each float.
		 * - Safely converts these to uint16_t by storing intermediate results to a temporary array.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m256 vector containing 8 single-precision floats.
		 */
		static inline void						store_fp32_to_bf16( uint16_t * _pDst, __m256 _mSrc );

		/**
	     * \brief Store 8 bfloat16 values from a __m128i vector to memory using AVX2.
	     *
	     * \param _pDst Pointer to the memory location where the 8 bfloat16 values will be stored.
	     * \param _mSrc The __m128i vector containing 8 bfloat16 values.
	     */
		static inline void						storeu_m128i_to_bf16( uint16_t * _pDst, __m128i _mSrc );

		/**
	     * \brief Store 8 bfloat16 values from a __m128i vector to memory using AVX2.
	     *
	     * \param _pDst Pointer to the memory location where the 8 bfloat16 values will be stored.
	     * \param _mSrc The __m128i vector containing 8 bfloat16 values.
	     */
		static inline void						store_m128i_to_bf16( uint16_t * _pDst, __m128i _mSrc );

		/**
		 * \brief Load 8 bfloat16 values from memory and convert them into a __m256 (8 floats) using AVX2.
		 *
		 * Each bfloat16 occupies 16 bits. 8 bfloat16 values = 8 * 2 bytes = 16 bytes total.
		 * We'll:
		 * - Load into a __m128i.
		 * - Unpack and zero-extend to 32 bits per element.
		 * - Shift left to position the bfloat16 bits correctly.
		 * - Reinterpret as floats.
		 *
		 * \param _pBF16 Pointer to memory containing 8 bfloat16 values.
		 * \return A __m256 vector containing 8 floats converted from bfloat16.
		 **/
		static inline __m256					loadu_bf16_to_fp32_8( const bfloat16 * _pBF16 ) { return loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
		 * \brief Load 8 bfloat16 values from memory and convert them into a __m256 (8 floats) using AVX2.
		 *
		 * Each bfloat16 occupies 16 bits. 8 bfloat16 values = 8 * 2 bytes = 16 bytes total.
		 * We'll:
		 * - Load into a __m128i.
		 * - Unpack and zero-extend to 32 bits per element.
		 * - Shift left to position the bfloat16 bits correctly.
		 * - Reinterpret as floats.
		 *
		 * \param _pBF16 Pointer to memory containing 8 bfloat16 values.
		 * \return A __m256 vector containing 8 floats converted from bfloat16.
		 **/
		static inline __m256					load_bf16_to_fp32_8( const bfloat16 * _pBF16 ) { return load_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
	     * \brief Load 8 bfloat16 values from memory into a __m128i vector using AVX2.
	     *
	     * Since AVX2 does not support BF16 directly, we just treat them as 16-byte raw data.
	     * You can then process them as needed. This is not a perfect parallel to __m512bh,
	     * but it gives you a vector holding 8 bfloat16 values.
	     * 
	     * \param _pBF16 Pointer to the memory containing 8 bfloat16 values.
	     * \return A __m128i vector containing the loaded bfloat16 values.
	     */
		static inline __m128i					loadu_bf16_to_m128i( const bfloat16 * _pBF16 ) { return loadu_bf16_to_m128i( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
	     * \brief Load 8 bfloat16 values from memory into a __m128i vector using AVX2.
	     *
	     * Since AVX2 does not support BF16 directly, we just treat them as 16-byte raw data.
	     * You can then process them as needed. This is not a perfect parallel to __m512bh,
	     * but it gives you a vector holding 8 bfloat16 values.
	     * 
	     * \param _pBF16 Pointer to the memory containing 8 bfloat16 values.
	     * \return A __m128i vector containing the loaded bfloat16 values.
	     */
		static inline __m128i					load_bf16_to_m128i( const bfloat16 * _pBF16 ) { return load_bf16_to_m128i( reinterpret_cast<const uint16_t *>(_pBF16) ); }

		/**
		 * \brief Store 8 floats from a __m256 vector to memory as 8 bfloat16 values using AVX2.
		 *
		 * This method:
		 * - Reinterprets floats as 32-bit ints.
		 * - Shifts right by 16 to get the top 16 bits of each float.
		 * - Safely converts these to uint16_t by storing intermediate results to a temporary array.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m256 vector containing 8 single-precision floats.
		 */
		static inline void						storeu_fp32_to_bf16( bfloat16 * _pDst, __m256 _mSrc ) { storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
		 * \brief Store 8 floats from a __m256 vector to memory as 8 bfloat16 values using AVX2.
		 *
		 * This method:
		 * - Reinterprets floats as 32-bit ints.
		 * - Shifts right by 16 to get the top 16 bits of each float.
		 * - Safely converts these to uint16_t by storing intermediate results to a temporary array.
		 *
		 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
		 * \param _mSrc The __m256 vector containing 8 single-precision floats.
		 */
		static inline void						store_fp32_to_bf16( bfloat16 * _pDst, __m256 _mSrc ) { store_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
	     * \brief Store 8 bfloat16 values from a __m128i vector to memory using AVX2.
	     *
	     * \param _pDst Pointer to the memory location where the 8 bfloat16 values will be stored.
	     * \param _mSrc The __m128i vector containing 8 bfloat16 values.
	     */
		static inline void						storeu_m128i_to_bf16( bfloat16 * _pDst, __m128i _mSrc ) { storeu_m128i_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
	     * \brief Store 8 bfloat16 values from a __m128i vector to memory using AVX2.
	     *
	     * \param _pDst Pointer to the memory location where the 8 bfloat16 values will be stored.
	     * \param _mSrc The __m128i vector containing 8 bfloat16 values.
	     */
		static inline void						store_m128i_to_bf16( bfloat16 * _pDst, __m128i _mSrc ) { store_m128i_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }
#endif	// #ifdef __AVX2__

#ifdef __AVX__
		/**
	     * \brief Store 4 floats from a __m128 vector to memory as 4 bfloat16 values using SSE intrinsics.
	     *
	     * This method:
	     * - Reinterprets floats as 32-bit integers.
	     * - Shifts right by 16 to extract the top 16 bits of each float, corresponding to bfloat16.
	     * - Packs the 32-bit integers into 16-bit unsigned integers.
	     *
	     * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	     * \param _mSrc The __m128 vector containing 4 single-precision floats.
	     */
		static inline void						storeu_fp32_to_bf16( uint16_t * _pDst, __m128 _mSrc );

		/**
	     * \brief Store 4 floats from a __m128 vector to memory as 4 bfloat16 values using SSE intrinsics.
	     *
	     * This method:
	     * - Reinterprets floats as 32-bit integers.
	     * - Shifts right by 16 to extract the top 16 bits of each float, corresponding to bfloat16.
	     * - Packs the 32-bit integers into 16-bit unsigned integers.
	     *
	     * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	     * \param _mSrc The __m128 vector containing 4 single-precision floats.
	     */
		static inline void						store_fp32_to_bf16( uint16_t * _pDst, __m128 _mSrc ) { storeu_fp32_to_bf16( _pDst, _mSrc ); }

		/**
	     * \brief Store 4 floats from a __m128 vector to memory as 4 bfloat16 values using SSE intrinsics.
	     *
	     * This method:
	     * - Reinterprets floats as 32-bit integers.
	     * - Shifts right by 16 to extract the top 16 bits of each float, corresponding to bfloat16.
	     * - Packs the 32-bit integers into 16-bit unsigned integers.
	     *
	     * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	     * \param _mSrc The __m128 vector containing 4 single-precision floats.
	     */
		static inline void						storeu_fp32_to_bf16( bfloat16 * _pDst, __m128 _mSrc ) { storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }

		/**
	     * \brief Store 4 floats from a __m128 vector to memory as 4 bfloat16 values using SSE intrinsics.
	     *
	     * This method:
	     * - Reinterprets floats as 32-bit integers.
	     * - Shifts right by 16 to extract the top 16 bits of each float, corresponding to bfloat16.
	     * - Packs the 32-bit integers into 16-bit unsigned integers.
	     *
	     * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	     * \param _mSrc The __m128 vector containing 4 single-precision floats.
	     */
		static inline void						store_fp32_to_bf16( bfloat16 * _pDst, __m128 _mSrc ) { storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pDst), _mSrc ); }
#endif	// #ifdef __AVX__

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
	inline __m512 bfloat16::loadu_bf16_to_fp32_16( const uint16_t * _pBF16 ) {
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
	inline __m512 bfloat16::load_bf16_to_fp32_16( const uint16_t * _pBF16 ) {
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
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
	/**
	 * \brief Load 8 bfloat16 values from memory and convert them into a __m256 (8 floats) using AVX2.
	 *
	 * Each bfloat16 occupies 16 bits. 8 bfloat16 values = 8 * 2 bytes = 16 bytes total.
	 * We'll:
	 * - Load into a __m128i.
	 * - Unpack and zero-extend to 32 bits per element.
	 * - Shift left to position the bfloat16 bits correctly.
	 * - Reinterpret as floats.
	 *
	 * \param _pBF16 Pointer to memory containing 8 bfloat16 values.
	 * \return A __m256 vector containing 8 floats converted from bfloat16.
	 **/
	inline __m256 bfloat16::loadu_bf16_to_fp32_8( const uint16_t * _pBF16 ) {
		// Step 1: Load 8 bfloat16 values (16 bytes) into a __m128i.
		__m128i mBF16 = _mm_loadu_si128( reinterpret_cast<const __m128i *>(_pBF16) );

		// We'll need a mZero vector for unpacking.
		__m128i mZero = _mm_setzero_si128();

		// Step 2: Unpack lower and upper 4 16-bit values into 32-bit values.
		// Unpack lower half (4 elements).
		__m128i mLow32  = _mm_unpacklo_epi16( mBF16, mZero ); // 4x32-bit in low half.
		// Unpack upper half (4 elements).
		__m128i mHigh32 = _mm_unpackhi_epi16( mBF16, mZero ); // 4x32-bit in high half.

		// Combine the two __m128i vectors into a single __m256i.
		// mLow32 -> lower 128 bits, mHigh32 -> upper 128 bits.
		__m256i mInt32 = _mm256_castsi128_si256( mLow32 );
		mInt32 = _mm256_inserti128_si256( mInt32, mHigh32, 1 );

		// Step 3: Shift left by 16 bits to position bfloat16 bits in the high half of the 32-bit float.
		mInt32 = _mm256_slli_epi32( mInt32, 16 );

		// Step 4: Reinterpret these bits as single-precision floats.
		__m256 mFloats = _mm256_castsi256_ps( mInt32 );

		return mFloats;
	}

	/**
	 * \brief Load 8 bfloat16 values from memory and convert them into a __m256 (8 floats) using AVX2.
	 *
	 * Each bfloat16 occupies 16 bits. 8 bfloat16 values = 8 * 2 bytes = 16 bytes total.
	 * We'll:
	 * - Load into a __m128i.
	 * - Unpack and zero-extend to 32 bits per element.
	 * - Shift left to position the bfloat16 bits correctly.
	 * - Reinterpret as floats.
	 *
	 * \param _pBF16 Pointer to memory containing 8 bfloat16 values.
	 * \return A __m256 vector containing 8 floats converted from bfloat16.
	 **/
	inline __m256 bfloat16::load_bf16_to_fp32_8( const uint16_t * _pBF16 ) {
		// Step 1: Load 8 bfloat16 values (16 bytes) into a __m128i.
		__m128i mBF16 = _mm_load_si128( reinterpret_cast<const __m128i *>(_pBF16) );

		// We'll need a mZero vector for unpacking.
		__m128i mZero = _mm_setzero_si128();

		// Step 2: Unpack lower and upper 4 16-bit values into 32-bit values.
		// Unpack lower half (4 elements).
		__m128i mLow32  = _mm_unpacklo_epi16( mBF16, mZero ); // 4x32-bit in low half.
		// Unpack upper half (4 elements).
		__m128i mHigh32 = _mm_unpackhi_epi16( mBF16, mZero ); // 4x32-bit in high half.

		// Combine the two __m128i vectors into a single __m256i.
		// mLow32 -> lower 128 bits, mHigh32 -> upper 128 bits.
		__m256i mInt32 = _mm256_castsi128_si256( mLow32 );
		mInt32 = _mm256_inserti128_si256( mInt32, mHigh32, 1 );

		// Step 3: Shift left by 16 bits to position bfloat16 bits in the high half of the 32-bit float.
		mInt32 = _mm256_slli_epi32( mInt32, 16 );

		// Step 4: Reinterpret these bits as single-precision floats.
		__m256 mFloats = _mm256_castsi256_ps( mInt32 );

		return mFloats;
	}

	/**
     * \brief Load 8 bfloat16 values from memory into a __m128i vector using AVX2.
     *
     * Since AVX2 does not support BF16 directly, we just treat them as 16-byte raw data.
     * You can then process them as needed. This is not a perfect parallel to __m512bh,
     * but it gives you a vector holding 8 bfloat16 values.
     * 
     * \param _pBF16 Pointer to the memory containing 8 bfloat16 values.
     * \return A __m128i vector containing the loaded bfloat16 values.
     */
    inline __m128i bfloat16::loadu_bf16_to_m128i( const uint16_t * _pBF16 ) {
        // Load 16 bytes (8 bfloat16 values) into a __m128i.
        return _mm_loadu_si128( reinterpret_cast<const __m128i *>(_pBF16) );
    }

	/**
     * \brief Load 8 bfloat16 values from memory into a __m128i vector using AVX2.
     *
     * Since AVX2 does not support BF16 directly, we just treat them as 16-byte raw data.
     * You can then process them as needed. This is not a perfect parallel to __m512bh,
     * but it gives you a vector holding 8 bfloat16 values.
     * 
     * \param _pBF16 Pointer to the memory containing 8 bfloat16 values.
     * \return A __m128i vector containing the loaded bfloat16 values.
     */
    inline __m128i bfloat16::load_bf16_to_m128i( const uint16_t * _pBF16 ) {
        // Load 16 bytes (8 bfloat16 values) into a __m128i.
        return _mm_load_si128( reinterpret_cast<const __m128i *>(_pBF16) );
    }

	/**
	 * \brief Store 8 floats from a __m256 vector to memory as 8 bfloat16 values using AVX2.
	 *
	 * This method:
	 * - Reinterprets floats as 32-bit ints.
	 * - Shifts right by 16 to get the top 16 bits of each float.
	 *
	 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	 * \param _mSrc The __m256 vector containing 8 single-precision floats.
	 */
	inline void bfloat16::storeu_fp32_to_bf16( uint16_t * _pDst, __m256 _mSrc ) {
		__m256i mIntRepr = _mm256_castps_si256( _mSrc );

		__m256i mBF16Int = _mm256_srli_epi32( mIntRepr, 16 );

		__m128i mBF16Int_lo = _mm256_extracti128_si256( mBF16Int, 0 );
		__m128i mBF16Int_hi = _mm256_extracti128_si256( mBF16Int, 1 );

		__m128i mPacked = _mm_packus_epi32( mBF16Int_lo, mBF16Int_hi );

		_mm_storeu_si128( reinterpret_cast<__m128i *>(_pDst), mPacked );
	}

	/**
	 * \brief Store 8 floats from a __m256 vector to memory as 8 bfloat16 values using AVX2.
	 *
	 * This method:
	 * - Reinterprets floats as 32-bit ints.
	 * - Shifts right by 16 to get the top 16 bits of each float.
	 *
	 * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
	 * \param _mSrc The __m256 vector containing 8 single-precision floats.
	 */
	inline void bfloat16::store_fp32_to_bf16( uint16_t * _pDst, __m256 _mSrc ) {
		__m256i mIntRepr = _mm256_castps_si256( _mSrc );

		__m256i mBF16Int = _mm256_srli_epi32( mIntRepr, 16 );

		__m128i mBF16Int_lo = _mm256_extracti128_si256( mBF16Int, 0 );
		__m128i mBF16Int_hi = _mm256_extracti128_si256( mBF16Int, 1 );

		__m128i mPacked = _mm_packus_epi32( mBF16Int_lo, mBF16Int_hi );

		_mm_store_si128( reinterpret_cast<__m128i *>(_pDst), mPacked );
	}

	/**
     * \brief Store 8 bfloat16 values from a __m128i vector to memory using AVX2.
     *
     * \param _pDst Pointer to the memory location where the 8 bfloat16 values will be stored.
     * \param _mSrc The __m128i vector containing 8 bfloat16 values.
     */
    inline void bfloat16::storeu_m128i_to_bf16( uint16_t * _pDst, __m128i _mSrc ) {
        // Store the 8 bfloat16 values (16 bytes) to memory.
        _mm_storeu_si128( reinterpret_cast<__m128i *>(_pDst), _mSrc );
    }

	/**
     * \brief Store 8 bfloat16 values from a __m128i vector to memory using AVX2.
     *
     * \param _pDst Pointer to the memory location where the 8 bfloat16 values will be stored.
     * \param _mSrc The __m128i vector containing 8 bfloat16 values.
     */
    inline void bfloat16::store_m128i_to_bf16( uint16_t * _pDst, __m128i _mSrc ) {
        // Store the 8 bfloat16 values (16 bytes) to memory.
        _mm_store_si128( reinterpret_cast<__m128i *>(_pDst), _mSrc );
    }
#endif	// #ifdef __AVX2__

#ifdef __AVX__
	/**
     * \brief Store 4 floats from a __m128 vector to memory as 4 bfloat16 values using SSE intrinsics.
     *
     * This method:
     * - Reinterprets floats as 32-bit integers.
     * - Shifts right by 16 to extract the top 16 bits of each float, corresponding to bfloat16.
     * - Packs the 32-bit integers into 16-bit unsigned integers.
     *
     * \param _pDst Pointer to the memory location where the bfloat16 values will be stored.
     * \param _mSrc The __m128 vector containing 4 single-precision floats.
     */
    inline void bfloat16::storeu_fp32_to_bf16( uint16_t * _pDst, __m128 _mSrc ) {
        __m128i mIntRepr = _mm_castps_si128( _mSrc );
        __m128i mBF16Int = _mm_srli_epi32( mIntRepr, 16 );
        __m128i mPacked = _mm_packus_epi32( mBF16Int, mBF16Int );
        _mm_storel_epi64( reinterpret_cast<__m128i *>(_pDst), mPacked );
    }
#endif	// #ifdef __AVX__
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

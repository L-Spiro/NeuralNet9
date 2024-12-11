/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Initialization functions.
 */

#pragma once

#include "NN9Math.h"
#include "../Types/NN9BFloat16.h"
#include "../Types/NN9Float16.h"
#include "../Utilities/NN9Utilities.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>


namespace nn9 {

	/**
	 * Class Init
	 * \brief Initialization functions.
	 *
	 * Description: Initialization functions.
	 */
	class Init {
	public :

		// == Functions.
		/**
		 * Initializes weights using Xavier/Glorot initialization.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _iFanIn Number of input units to the layer.
		 * \param _iFanOut Number of output units from the layer.
		 * \param _vWeights Reference to the vector of weights to be initialized.
		 * \return Returns _vWeights.
		 **/
		template <typename _tType>
		static _tType &												XavierInitialization( int _iFanIn, int _iFanOut, _tType &_vWeights ) {
			double dLimit = std::sqrt( 6.0 / (_iFanIn + _iFanOut) );

			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::uniform_real_distribution<double> urdDis( -dLimit, dLimit );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( urdDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * Initializes weights using He/Kaiming initialization.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _iFanIn Number of input units to the layer.
		 * \param _vWeights Reference to the vector of weights to be initialized.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												HeInitialization( int _iFanIn, _tType &_vWeights ) {
			double dStdDev = std::sqrt( 2.0 / _iFanIn );

			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::normal_distribution<double> ndDis( 0.0, dStdDev );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( ndDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * \brief Initializes weights using LeCun initialization.
		 * 
		 * LeCun initialization is typically used for SELU activation functions.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _iFanIn Number of input units to the layer.
		 * \param _vWeights Reference to the vector of weights to be initialized.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												LeCunInitialization( int _iFanIn, _tType &_vWeights ) {
			double dStdDev = std::sqrt( 1.0 / _iFanIn );

			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::normal_distribution<double> ndDis( 0.0, dStdDev );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( ndDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * \brief Initializes weights using Orthogonal initialization.
		 * 
		 * Orthogonal initialization tries to make the weight matrix orthogonal. 
		 * This is often done for 2D weight tensors (e.g., fully-connected layers).
		 * For simplicity, we assume the weight vector represents a 2D matrix row-major.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _iRows Number of rows in the weight matrix.
		 * \param _iCols Number of columns in the weight matrix.
		 * \param _vWeights Reference to the vector of weights (length must be _iRows*_iCols).
		 * \throws std::runtime_error if _iRows*_iCols does not match the size of _vWeights.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												OrthogonalInitialization( int _iRows, int _iCols, _tType &_vWeights ) {
			if ( static_cast<int>(_vWeights.size()) != _iRows * _iCols ) {
				throw std::runtime_error("Size of _vWeights does not match _iRows*_iCols.");
			}

			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::normal_distribution<double> ndDis( 0.0, 1.0 );

			// Fill _vWeights with random values.
			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( ndDis( mGen ) );
			}

			// Reshape into matrix form (row-major)
			// Compute QR decomposition to get orthogonal matrix
			// For simplicity, weÅfll implement a basic Gram-Schmidt process.
			// This is a simplified orthogonalization and may not be as numerically stable as a full QR decomposition.

			// Convert _vWeights into a vector of vectors for easier manipulation.
			std::vector<std::vector<double>> mMatrix( _iRows, std::vector<double>( _iCols, 0.0 ) );
			for ( int i = 0; i < _iRows; ++i ) {
				for ( int j = 0; j < _iCols; ++j ) {
					mMatrix[i][j] = double( _vWeights[i*_iCols+j] );
				}
			}

			// Gram-Schmidt Orthogonalization.
			for ( int i = 0; i < _iRows; ++i ) {
				// Normalize i-th row against previous rows
				for ( int j = 0; j < i; ++j ) {
					double dDot = 0.0;
					for ( int k = 0; k < _iCols; ++k ) {
						dDot += mMatrix[i][k] * mMatrix[j][k];
					}

					for ( int k = 0; k < _iCols; ++k ) {
						mMatrix[i][k] -= dDot * mMatrix[j][k];
					}
				}
				// Normalize the row
				double dNorm = 0.0;
				for ( int k = 0; k < _iCols; ++k ) {
					dNorm += mMatrix[i][k] * mMatrix[i][k];
				}
				dNorm = std::sqrt( dNorm );
				if ( dNorm > 1e-6 ) {
					for ( int k = 0; k < _iCols; ++k ) {
						mMatrix[i][k] /= dNorm;
					}
				}
			}

			// Copy back to _vWeights.
			for ( int i = 0; i < _iRows; ++i ) {
				for ( int j = 0; j < _iCols; ++j ) {
					_vWeights[i*_iCols+j] = _tType::value_type( mMatrix[i][j] );
				}
			}
			return _vWeights;
		}

		/**
		 * \brief Initializes weights uniformly between a given range.
		 * 
		 * Uniform initialization is simple and often used as a baseline.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _dMin The lower bound of the uniform distribution.
		 * \param _dMax The upper bound of the uniform distribution.
		 * \param _vWeights  Reference to the vector of weights to be initialized.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												UniformInitialization( double _dMin, double _dMax, _tType &_vWeights ) {
			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::uniform_real_distribution<double> urdDis( _dMin, _dMax );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( urdDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * \brief Initializes weights using a normal (Gaussian) distribution.
		 * 
		 * Can be used as a baseline or combined with scaling factors for specific layers.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _dMean Mean of the Gaussian distribution.
		 * \param _dStdDev Standard deviation of the Gaussian distribution.
		 * \param _vWeights Reference to the vector of weights to be initialized.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												NormalInitialization( double _dMean, double _dStdDev, _tType &_vWeights ) {
			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::normal_distribution<double> ndDis( _dMean, _dStdDev );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( ndDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * \brief Initializes weights using a fan-based scaling similar to Xavier or He, but with uniform distribution.
		 * 
		 * This is known as Xavier/Glorot Uniform or He Uniform initialization.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _iFanIn Number of input units.
		 * \param _iFanOut Number of output units (for Xavier), or ignored for pure fan-in scaling (for He).
		 * \param _vWeights Reference to the vector of weights to be initialized.
		 * \param _bUseHe If true, uses He scaling; otherwise uses Xavier scaling.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												ScaledUniformInitialization( int _iFanIn, int _iFanOut, _tType &_vWeights, bool _bUseHe = false ) {
			double dLimit;
			if ( _bUseHe ) {
				// He uniform
				dLimit = std::sqrt( 6.0 / _iFanIn );
			}
			else {
				// Xavier uniform
				dLimit = std::sqrt( 6.0 / (_iFanIn + _iFanOut) );
			}

			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::uniform_real_distribution<double> urdDis( -dLimit, dLimit );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( urdDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * \brief Initializes weights using a fan-based scaling similar to Xavier or He, but with normal distribution.
		 * 
		 * This is known as Xavier/Glorot Normal or He Normal initialization.
		 * 
		 * \tparam _tType The view/container type.
		 * \param _iFanIn Number of input units.
		 * \param _iFanOut Number of output units (for Xavier), or ignored for pure fan-in scaling (for He).
		 * \param _vWeights Reference to the vector of weights to be initialized.
		 * \param _bUseHe If true, uses He scaling; otherwise uses Xavier scaling.
		 * \return Returns _vWeights.
		 */
		template <typename _tType>
		static _tType &												ScaledNormalInitialization( int _iFanIn, int _iFanOut, _tType &_vWeights, bool _bUseHe = false ) {
			double dStdDev;
			if ( _bUseHe ) {
				// He normal
				dStdDev = std::sqrt( 2.0 / _iFanIn );
			}
			else {
				// Xavier normal
				dStdDev = std::sqrt( 2.0 / (_iFanIn + _iFanOut) );
			}

			std::random_device rdDev;
			std::mt19937 mGen( rdDev() );
			std::normal_distribution<double> ndDis( 0.0, dStdDev );

			for ( auto & fWeight : _vWeights ) {
				fWeight = _tType::value_type( ndDis( mGen ) );
			}
			return _vWeights;
		}

		/**
		 * Copies from one view to another, potentially performing a type conversion in the process.
		 * 
		 * \tparam _tTypeIn The input view/container type.
		 * \tparam _tTypeOut The output view/container type.
		 * \param _vIn The source view.
		 * \param _vOut The destination view.
		 * \throw If NN9_SAFETY_CHECK, the function throws if the views are not the same size in elements.
		 * \return Returns 
		 **/
		template <typename _tTypeIn, typename _tTypeOut>
		static _tTypeOut &											CopyView( const _tTypeIn &_vIn, _tTypeOut &_vOut ) {
			using ValueTypeIn = typename _tTypeIn::value_type;
			using ValueTypeOut = typename _tTypeOut::value_type;
#ifdef NN9_SAFETY_CHECK
			if ( _vIn.size() != _vOut.size() ) {
				throw std::runtime_error( "Init::CopyView: The views must both have the same number of elements." );
			}
#endif		// #ifdef NN9_SAFETY_CHECK

			if constexpr ( std::is_same<ValueTypeIn, ValueTypeOut>::value ) {
				std::memcpy( &_vOut[0], &_vIn[0], _vIn.size() * sizeof( ValueTypeIn ) );
				return _vOut;
			}

			// The types differ.
			const ValueTypeIn * pvtiIn = &_vIn[0];
			ValueTypeOut * pvtoOut = &_vOut[0];
			auto sSize = _vIn.size();

			if constexpr ( nn9::Math::IsBFloat16<ValueTypeOut>() ) {
				CopyToBFloat16<ValueTypeIn>( reinterpret_cast<bfloat16_t *>(pvtoOut), reinterpret_cast<const ValueTypeIn *>(pvtiIn), sSize );
				return _vOut;
			}
			if constexpr ( nn9::Math::IsBFloat16<ValueTypeIn>() ) {
				CopyFromBFloat16<ValueTypeOut>( pvtoOut, reinterpret_cast<const bfloat16_t *>(pvtiIn), sSize );
				return _vOut;
			}

			for ( size_t i = 0; i < sSize; ++i ) {
				pvtoOut[i] = static_cast<ValueTypeOut>(pvtiIn[i]);
			}
			return _vOut;
		}


	protected :
		/**
		 * Copies from not-bfloat16_t to bfloat16_t.
		 * 
		 * \tparam _tTypeIn The non-bfloat16_t type from which to copy.
		 * \param _pbfOut Pointer to the target array of bfloat16_t's.
		 * \param _ptiIn Pointer to the array of the source data.
		 * \param _sTotal The total elements to which both pointers point.
		 **/
		template <typename _tTypeIn>
		static void													CopyToBFloat16( bfloat16_t * _pbfOut, const _tTypeIn * _ptiIn, size_t _sTotal ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				// Operate on 16 values at once.

				while ( _sTotal >= 16 ) {
					if constexpr ( nn9::Math::Is32BitFloat<_tTypeIn>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pbfOut), _mm512_loadu_ps( _ptiIn ) );
					}
					else {
						NN9_ALIGN( 64 )
						float fTmp[16];
						fTmp[0] = float( _ptiIn[0] );
						fTmp[1] = float( _ptiIn[1] );
						fTmp[2] = float( _ptiIn[2] );
						fTmp[3] = float( _ptiIn[3] );
						fTmp[4] = float( _ptiIn[4] );
						fTmp[5] = float( _ptiIn[5] );
						fTmp[6] = float( _ptiIn[6] );
						fTmp[7] = float( _ptiIn[7] );
						fTmp[8] = float( _ptiIn[8] );
						fTmp[9] = float( _ptiIn[9] );
						fTmp[10] = float( _ptiIn[10] );
						fTmp[11] = float( _ptiIn[11] );
						fTmp[12] = float( _ptiIn[12] );
						fTmp[13] = float( _ptiIn[13] );
						fTmp[14] = float( _ptiIn[14] );
						fTmp[15] = float( _ptiIn[15] );

						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pbfOut), _mm512_load_ps( fTmp ) );
					}

					_pbfOut += 16;
					_ptiIn += 16;
					_sTotal -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				// Operate on 8 values at once.

				// Have to operate on 16 values at a time for alignment, but only 8 at a time can be used with AVX.
				while ( _sTotal >= 8 ) {
					if constexpr ( nn9::Math::Is32BitFloat<_tTypeIn>() ) {
						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pbfOut), _mm256_loadu_ps( _ptiIn ) );
					}
					else {
						NN9_ALIGN( 32 )
						float fTmp[8];
						fTmp[0] = float( _ptiIn[0] );
						fTmp[1] = float( _ptiIn[1] );
						fTmp[2] = float( _ptiIn[2] );
						fTmp[3] = float( _ptiIn[3] );
						fTmp[4] = float( _ptiIn[4] );
						fTmp[5] = float( _ptiIn[5] );
						fTmp[6] = float( _ptiIn[6] );
						fTmp[7] = float( _ptiIn[7] );

						nn9::bfloat16::storeu_fp32_to_bf16( reinterpret_cast<uint16_t *>(_pbfOut), _mm256_load_ps( fTmp ) );
					}

					_pbfOut += 8;
					_ptiIn += 8;
					_sTotal -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sTotal-- ) {
				(*_pbfOut++) = float( (*_ptiIn++) );
			}
		}

		/**
		 * Copies from bfloat16_t to non-bfloat16_t.
		 * 
		 * \tparam _tTypeOut The non-bfloat16_t type to which to copy.
		 * \param _ptiOut Pointer to the target array of not-bfloat16_t values.
		 * \param _pbfIn Pointer to the array of the source bfloat16_t's.
		 * \param _sTotal The total elements to which both pointers point.
		 **/
		template <typename _tTypeOut>
		static void													CopyFromBFloat16( _tTypeOut * _ptiOut, const bfloat16_t * _pbfIn, size_t _sTotal ) {
#ifdef __AVX512F__
			if ( Utilities::IsAvx512FSupported() ) {
				// Operate on 16 values at once.
				while ( _sTotal >= 16 ) {
					if constexpr ( nn9::Math::Is32BitFloat<_tTypeOut>() ) {
						_mm512_storeu_ps( _ptiOut, nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pbfIn) ) );
					}
					else {
						NN9_ALIGN( 64 )
						float fTmp[16];
						_mm512_store_ps( fTmp, nn9::bfloat16::loadu_bf16_to_fp32_16( reinterpret_cast<const uint16_t *>(_pbfIn) ) );
						_ptiOut[0] = _tTypeOut( fTmp[0] );
						_ptiOut[1] = _tTypeOut( fTmp[1] );
						_ptiOut[2] = _tTypeOut( fTmp[2] );
						_ptiOut[3] = _tTypeOut( fTmp[3] );
						_ptiOut[4] = _tTypeOut( fTmp[4] );
						_ptiOut[5] = _tTypeOut( fTmp[5] );
						_ptiOut[6] = _tTypeOut( fTmp[6] );
						_ptiOut[7] = _tTypeOut( fTmp[7] );
						_ptiOut[8] = _tTypeOut( fTmp[8] );
						_ptiOut[9] = _tTypeOut( fTmp[9] );
						_ptiOut[10] = _tTypeOut( fTmp[10] );
						_ptiOut[11] = _tTypeOut( fTmp[11] );
						_ptiOut[12] = _tTypeOut( fTmp[12] );
						_ptiOut[13] = _tTypeOut( fTmp[13] );
						_ptiOut[14] = _tTypeOut( fTmp[14] );
						_ptiOut[15] = _tTypeOut( fTmp[15] );
					}
					_ptiOut += 16;
					_pbfIn += 16;
					_sTotal -= 16;
				}
			}
#endif	// #ifdef __AVX512F__

#ifdef __AVX2__
			if ( Utilities::IsAvx2Supported() ) {
				// Operate on 8 values at once.
				while ( _sTotal >= 8 ) {
					if constexpr ( nn9::Math::Is32BitFloat<_tTypeOut>() ) {
						_mm256_storeu_ps( _ptiOut, nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pbfIn) ) );
					}
					else {
						NN9_ALIGN( 32 )
						float fTmp[8];
						_mm256_store_ps( fTmp, nn9::bfloat16::loadu_bf16_to_fp32_8( reinterpret_cast<const uint16_t *>(_pbfIn) ) );
						_ptiOut[0] = _tTypeOut( fTmp[0] );
						_ptiOut[1] = _tTypeOut( fTmp[1] );
						_ptiOut[2] = _tTypeOut( fTmp[2] );
						_ptiOut[3] = _tTypeOut( fTmp[3] );
						_ptiOut[4] = _tTypeOut( fTmp[4] );
						_ptiOut[5] = _tTypeOut( fTmp[5] );
						_ptiOut[6] = _tTypeOut( fTmp[6] );
						_ptiOut[7] = _tTypeOut( fTmp[7] );
					}
					_ptiOut += 8;
					_pbfIn += 8;
					_sTotal -= 8;
				}
			}
#endif	// #ifdef __AVX2__

			while ( _sTotal-- ) {
				(*_ptiOut++) = _tTypeOut( (*_pbfIn++) );
			}
		}
	};

}	// namespace nn9

/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Initialization functions.
 */

#pragma once

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

	};

}	// namespace nn9

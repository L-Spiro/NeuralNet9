/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Provides fast access to Buffers.  Views map full buffers or buffer ranges to memory and provide easy and
 *	safe iteration over those ranges.  This allows buffers to keep in memory only the ranges that are being accessed,
 *	which is how gigantic neural networks with billions or trillions of parameters are able to be supported.
 */

#pragma once

#include "../Buffers/NN9Buffer.h"

#include <iterator>
#include <stdexcept>


namespace nn9 {

	/**
	 * Class View
	 * \brief Provides fast access to Buffers.
	 *
	 * Description: Provides fast access to Buffers.  Views map full buffers or buffer ranges to memory and provide easy and
	 *	safe iteration over those ranges.  This allows buffers to keep in memory only the ranges that are being accessed,
	 *	which is how gigantic neural networks with billions or trillions of parameters are able to be supported.
	 */
	template <typename _tDataType>
	class View {
	public :
		View( _tDataType * _pTarget, size_t _sSize, RefCnt * _prcRefObj ) :
			m_pData( _pTarget ),
			m_sTotal( _sSize ),
			m_prcRef( _prcRefObj ) {
			if ( m_prcRef ) { m_prcRef->IncRef(); }
		}
		View( const View<_tDataType> & _vOther ) :
			m_pData( _vOther.m_pData ), 
			m_sTotal( _vOther.m_sTotal ),
			m_prcRef( _vOther.m_prcRef ) {
			if ( m_prcRef ) { m_prcRef->IncRef(); }
		}
		View( View<_tDataType> && _vOther ) :
			m_pData( _vOther.m_pData ), 
			m_sTotal( _vOther.m_sTotal ),
			m_prcRef( _vOther.m_prcRef ) {
			_vOther.m_pData = nullptr;
			_vOther.m_sTotal = 0;
			_vOther.m_prcRef = nullptr;
		}
		~View() {
			if ( m_prcRef ) { m_prcRef->DecRef(); }
		}


		// == Types.
		typedef _tDataType										value_type;							/**< std::vector<>-compatible value_type definition. */

		/**
		 * Class Iterator
		 * \brief Allows for ( : ) iteration over the values.
		 *
		 * Description: Provides fast access to Buffers.  Views map full buffers or buffer ranges to memory and provide easy and
		 *	safe iteration over those ranges.  This allows buffers to keep in memory only the ranges that are being accessed,
		 *	which is how gigantic neural networks with billions or trillions of parameters are able to be supported.
		 */
		class Iterator {
		public :
			// == Iterator Traits.
			using iterator_category								= std::random_access_iterator_tag;
			using value_type									= _tDataType;
			using difference_type								= std::ptrdiff_t;
			using pointer										= _tDataType *;
			using reference										= _tDataType &;

			
			Iterator( _tDataType * pNode ) :
				m_pTarget( pNode ) {}


			// == Operators.
			/**
			 * Dereference operator.
			 * 
			 * \return Returns the value at the current iterator.
			 **/
			reference											operator * () const {
				return (*m_pTarget);
			}

			/**
			 * Arrow operator.
			 * 
			 * \return Returns a pointer to the current value.
			 **/
			pointer												operator -> () {
				return m_pTarget;
			}

			/**
			 * Pre-increment operator.
			 * 
			 * \return Returns this iterator after incrementing the pointer location.
			 **/
			Iterator &											operator ++ () {
				++m_pTarget;
				return (*this);
			}

			/**
			 * Post-increment operator.
			 * 
			 * \return Returns a copy of this iterator prior to incrementing the pointer location.
			 **/
			Iterator											operator ++ ( int ) {
				auto aTmp = (*this);
				++(*this);
				return aTmp;
			}

			/**
			 * Pre-decrement operator.
			 * 
			 * \return Returns this iterator after decrementing the pointer location.
			 **/
			Iterator &											operator -- () {
				--m_pTarget;
				return (*this);
			}

			/**
			 * Post-decrement operator.
			 * 
			 * \return Returns a copy of this iterator prior to decrementing the pointer location.
			 **/
			Iterator											operator -- ( int ) {
				auto aTmp = (*this);
				--(*this);
				return aTmp;
			}

			/**
			 * Moves the iterator forward by n elements.
			 * 
			 * \param _dN The number of positions to advance.
			 * \return A reference to the advanced iterator.
			 */
			Iterator &											operator += ( difference_type _dN ) {
				m_pTarget += _dN;
				return (*this);
			}

			/**
			 * Moves the iterator backward by n elements.
			 * 
			 * \param _dN The number of positions to move backward.
			 * \return A reference to the moved iterator.
			 */
			Iterator &											operator -= ( difference_type _dN ) {
				m_pTarget -= _dN;
				return (*this);
			}

			/**
			 * Returns an iterator pointing _dN elements ahead.
			 * 
			 * \param _dN The number of positions to advance.
			 * \return A new iterator advanced by _dN positions.
			 */
			Iterator											operator + ( difference_type _dN ) const {
				return Iterator( m_pTarget + _dN );
			}

			/**
			 * Returns an iterator pointing _dN elements behind.
			 * 
			 * \param _dN The number of positions to move backward.
			 * \return A new iterator moved backward by _dN positions.
			 */
			Iterator											operator - ( difference_type _dN ) const {
				return Iterator( m_pTarget - _dN );
			}

			/**
			 * Computes the distance between two iterators.
			 * 
			 * \param _itOther Another iterator to compare against.
			 * \return The difference in elements between this iterator and _itOther.
			 */
			difference_type										operator - ( const Iterator & _itOther ) const {
				return m_pTarget - _itOther.m_pTarget;
			}

			/**
			 * Allows random indexing like an array.
			 * 
			 * \param _dN The index offset from the current iterator position.
			 * \return A reference to the element at the given offset.
			 */
			reference											operator [] ( difference_type _dN ) const {
				return (*(m_pTarget + _dN));
			}

			/**
			 * Equality comparison.
			 * 
			 * \param _iOther The iterator against which to compare.
			 * \return Returns true if the iterators point to the same location.
			 **/
			bool												operator == ( const Iterator &_iOther ) const {
				return m_pTarget == _iOther.m_pTarget;
			}

			/**
			 * Inequality comparison.
			 * 
			 * \param _iOther The iterator against which to compare.
			 * \return Returns true if the iterators do not point to the same location.
			 **/
			bool												operator != ( const Iterator &_iOther ) const {
				return m_pTarget != _iOther.m_pTarget;
			}

			/**
			 * Checks if this iterator is less than another iterator.
			 * 
			 * \param _iOther Another iterator.
			 * \return True if this iterator precedes _iOther in the sequence, else false.
			 */
			bool												operator < ( const Iterator &_iOther ) const {
				return m_pTarget < _iOther.m_pTarget;
			}

			/**
			 * Checks if this iterator is less than or equal to another iterator.
			 * 
			 * \param _iOther Another iterator.
			 * \return True if this iterator precedes or equals _iOther, else false.
			 */
			bool												operator <= ( const Iterator &_iOther ) const {
				return m_pTarget <= _iOther.m_pTarget;
			}

			/**
			 * Checks if this iterator is greater than another iterator.
			 * 
			 * \param _iOther Another iterator.
			 * \return True if this iterator follows _iOther in the sequence, else false.
			 */
			bool												operator > ( const Iterator &_iOther ) const {
				return m_pTarget > _iOther.m_pTarget;
			}

			/**
			 * Checks if this iterator is greater than or equal to another iterator.
			 * 
			 * \param _iOther Another iterator.
			 * \return True if this iterator follows or equals _iOther, else false.
			 */
			bool												operator >= ( const Iterator &_iOther ) const {
				return m_pTarget >= _iOther.m_pTarget;
			}

		private :
			_tDataType *										m_pTarget;								/**< The data to which we point. */
		};


		// == Operators.
		/**
		 * Provides indexed access to elements (mutable).
		 * 
		 * \param _sIdx The position of the element.
		 * \return A reference to the element at _sIdx.
		 * \throws std::out_of_range if _sIdx is outside the valid range (_DEBUG only).
		 */
		_tDataType &											operator [] ( std::size_t _sIdx ) {
#ifdef _DEBUG
			if ( _sIdx >= m_sTotal ) {
				throw std::out_of_range( "View::[]: Index out of range." );
			}
#endif	// #ifdef _DEBUG
			return m_pData[_sIdx];
		}

		/**
		 * Provides indexed access to elements (const).
		 * 
		 * \param _sIdx The position of the element.
		 * \return A const reference to the element at _sIdx.
		 * \throws std::out_of_range if _sIdx is outside the valid range (_DEBUG only).
		 */
		const _tDataType &										operator [] ( std::size_t _sIdx ) const {
#ifdef _DEBUG
			if ( _sIdx >= m_sTotal ) {
				throw std::out_of_range( "View::[]: Index out of range." );
			}
#endif	// #ifdef _DEBUG
			return m_pData[_sIdx];
		}

		/**
		 * The std::move operator.
		 * 
		 * \param _vOther The object to copy.
		 * \return Returns a reference to this object following the copy.
		 **/
		View<_tDataType> &										operator = ( View<_tDataType> && _vOther ) {
			if ( this != &_vOther ) {
				m_pData = _vOther.m_pData;
				m_sTotal = _vOther.m_sTotal;
				m_prcRef = _vOther.m_prcRef;

				_vOther.m_pData = nullptr;
				_vOther.m_sTotal = 0;
				_vOther.m_prcRef = nullptr;
			}
			return (*this);
		}


		// == Functions.
		/**
		 * Retrieves the size of the container.
		 * 
		 * \return The number of elements stored.
		 */
		std::size_t												size() const { return m_sTotal; }

		/**
		 * Returns an iterator to the beginning of the container.
		 * 
		 * \return An iterator pointing to the first element.
		 */
		Iterator												begin() { return Iterator( m_pData ); }

		/**
		 * Returns an iterator to the end of the container.
		 * 
		 * \return An iterator pointing past the last element.
		 */
		Iterator												end() { return Iterator( m_pData + m_sTotal ); }

		/**
		 * Returns a const iterator to the beginning of the container.
		 * 
		 * \return A const iterator pointing to the first element.
		 */
		Iterator												begin() const { return Iterator( m_pData ); }

		/**
		 * Returns a const iterator to the end of the container.
		 * 
		 * \return A const iterator pointing past the last element.
		 */
		Iterator												end() const { return Iterator( m_pData + m_sTotal ); }


	private :
		// == Members.
		_tDataType *											m_pData;										/**< A pointer to the data over which we iterate. */
		size_t													m_sTotal;										/**< The total data over which we can iterate. */
		RefCnt *												m_prcRef;										/**< The object we reference-count. */
	};

}	// namespace nn9

/**
 * Adds a specified number of steps to an iterator, returning a new iterator.
 * 
 * \tparam _tDataType The element type of the iterator.
 * \param _dN The number of steps to advance.
 * \param _iIt The iterator to advance.
 * \return A new iterator advanced by _dN steps from it.
 */
template <typename _tDataType>
typename nn9::View<_tDataType>::Iterator						operator + ( typename nn9::View<_tDataType>::Iterator::difference_type _dN, 
	const typename nn9::View<_tDataType>::Iterator & _iIt ) {
    return _iIt + _dN;
}

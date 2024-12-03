/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Provides an interface for reference-counted objects.
 */

#pragma once

#include <atomic>
#include <cstdint>


namespace nn9 {

	/**
	 * Class RefCnt
	 * \brief Provides an interface for reference-counted objects.
	 *
	 * Description: Provides an interface for reference-counted objects.
	 */
	class RefCnt {
	public :
		RefCnt() {}
		virtual ~RefCnt() {}


		// == Functions.
		/**
		 * Increases the reference count.
		 **/
		virtual void																IncRef() { ++m_aCnt; }

		/**
		 * Decreases the reference count.
		 * 
		 * \return Returns the reference count.
		 **/
		virtual int32_t																DecRef() { return --m_aCnt; }

		/**
		 * Gets the reference count.
		 *
		 * \return Returns the reference count.
		 */
		int32_t																		GetRefCnt() const { return m_aCnt; }


	protected :
		// == Members.
		std::atomic<int32_t>														m_aCnt = 0;						/**< The reference count. */

	};

}	// namespace nn9

/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A buffer can be interpreted as any kind of data and be flushed to disk.  They maintain reference counts for sharing and can be fully
 *	or partially mapped to memory.  Buffers are always aligned in memory to 64 bytes.
 */

#pragma once

#include "../Foundation/NN9AlignmentAllocator.h"

#include <vector>


namespace nn9 {

	/**
	 * Class Buffer
	 * \brief A buffer can be interpreted as any kind of data and be flushed to disk.
	 *
	 * Description: A buffer can be interpreted as any kind of data and be flushed to disk.  They maintain reference counts for sharing and can be fully
	 *	or partially mapped to memory.  Buffers are always aligned in memory to 64 bytes.
	 */
	class Buffer {
	public :
		Buffer() {}



	protected :
		// == Members.
		std::vector<uint8_t, AlignmentAllocator<uint8_t, 64>>						m_vBuffer;					/**< The actual data buffer. */
	};

}	// namespace nn9

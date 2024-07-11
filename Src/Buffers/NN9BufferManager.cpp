/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Manages buffers.  Buffers need to be flushed to disk sometimes depending on memory constraints etc.  This class operates behind-the-scenes.
 */
 
 #include "NN9BufferManager.h"


 namespace nn9 {

	BufferManager BufferManager::GblBufferManager;								/**< Behind-the-scenes buffer manager. */

 }	// namespace nn9

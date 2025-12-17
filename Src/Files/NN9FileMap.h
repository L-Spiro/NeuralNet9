/**
 * Copyright L. Spiro 2022
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: A file-mapping.
 */


#pragma once

#include "NN9FileBase.h"

#include <string>
#include <vector>

#ifdef NN9_USE_WINDOWS
#include <Helpers/LSWHelpers.h>
#endif	// #ifdef NN9_USE_WINDOWS

namespace nn9 {

	/**
	 * Class FileMap
	 * \brief A file-mapping.
	 *
	 * Description: A file-mapping.
	 */
	class FileMap : public FileBase {
	public :
		FileMap();
		virtual ~FileMap();


		// == Functions.
		/**
		 * Opens a file.  The path is given in UTF-8.
		 *
		 * \param _pFile Path to the file to open.
		 * \return Returns an error code indicating the result of the operation.
		 */
		virtual NN9_ERRORS									Open( const std::filesystem::path &_pFile );

		/**
		 * Creates a file.  The path is given in UTF-8.
		 *
		 * \param _pFile Path to the file to create.
		 * \return Returns an error code indicating the result of the operation.
		 */
		virtual NN9_ERRORS									Create( const std::filesystem::path &_pFile );

		/**
		 * Closes the opened file.
		 */
		virtual void										Close();

		/**
		 * Gets the size of the file.
		 * 
		 * \return Returns the size of the file.
		 **/
		virtual uint64_t									Size() const;


	protected :
		// == Members.
#ifdef _WIN32
#ifdef NN9_USE_WINDOWS
		lsw::LSW_HANDLE										m_hFile;						/**< The file handle. */
		lsw::LSW_HANDLE										m_hMap;							/**< The file-mapping handle. */
#else
		HANDLE												m_hFile = NULL;					/**< The file handle. */
		HANDLE												m_hMap = NULL;					/**< The file-mapping handle. */
#endif	// #ifdef SBN_USE_WINDOWS
		mutable PBYTE										m_pbMapBuffer;					/**< Mapped bytes. */
		bool												m_bIsEmpty;						/**< Is the file 0-sized? */
		bool												m_bWritable;					/**< Read-only or read-write? */
		mutable uint64_t									m_ui64Size;						/**< Size of the file. */
		mutable uint64_t									m_ui64MapStart;					/**< Map start. */
		mutable DWORD										m_dwMapSize;					/**< Mapped size. */
#else
#endif	// #ifdef _WIN32


		// == Functions.
		/**
		 * Creates the file map.
		 * 
		 * \return Returns true if the file mapping was successfully created.
		 **/
		NN9_ERRORS											CreateFileMap();

	};

}	// namespace nn9

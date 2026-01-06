/**
 * Copyright L. Spiro 2024
 *
 * Written by: Shawn (L. Spiro) Wilcoxen
 *
 * Description: Error codes and functions for handling errors.
 */

#pragma once

#include "../Compression/MiniZ/miniz.h"
#include "../Foundation/NN9Macros.h"
#include "../OS/NN9Os.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <curl/curl.h>
#include <iomanip>
#include <errno.h>
#include <string>

//#ifdef _WIN32
//#include <Winhttp.h>
//#endif	// #ifdef _WIN32

namespace nn9 {

	// == Types.
	/** Error codes. */
	enum NN9_ERRORS : uint16_t {
#define NN9_E_ENUM( ENUM, TXT )			ENUM,
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
	};


	/**
	 * Class Errors
	 * \brief Provides functionality for working with errors and error codes.
	 *
	 * Description: Provides functionality for working with errors and error codes.
	 */
	class Errors {
	public :
		// == Functions.
		/**
		 * Converts an errno_t to one of our error codes.
		 * 
		 * \param _eCode The code to convert.
		 * \return Returns the converted error code.
		 **/
		static inline NN9_ERRORS									ErrNo_T_To_Native( errno_t _eCode );

		/**
		 * Converts a mz_zip_error error code to one of our error codes.
		 * 
		 * \param _zeCode The code to convert.
		 * \return Returns the converted error code.
		 **/
		static inline NN9_ERRORS									ZipError_To_Native( mz_zip_error _zeCode );

		/**
		 * Converts a libcurl error to one of our error codes.
		 * 
		 * \param _cCode The code to convert.
		 * \return Returns the converted error code.
		 **/
		static inline NN9_ERRORS									LibCurl_To_Native( CURLcode _cCode );

#ifdef _WIN32
		/**
		 * Calls ::GetLastError() and converts the error code to one of our error codes.
		 * 
		 * \return Returns the converted error code.
		 **/
		static inline NN9_ERRORS									GetLastError_To_Native();

		/**
		 * Displays the current ::GetLastError() error with a description.
		 * 
		 * \param _dwErr The error code to translate.  If -1, ::GetLastError() is called.
		 **/
		static inline void											DisplayLastError( DWORD _dwErr = DWORD( -1 ) );

#endif	// #ifdef _WIN32

		/**
		 * Gets the string description of an error code.
		 * 
		 * \param _eCode The error code whose description is to be gotten.
		 * \return Returns the text description for the given error.
		 **/
		static inline const char8_t *								ToStrPU8( NN9_ERRORS _eCode );

		/**
		 * Gets the string description of an error code.
		 * 
		 * \param _eCode The error code whose description is to be gotten.
		 * \return Returns the text description for the given error.
		 **/
		static inline std::u8string									ToStrU8( NN9_ERRORS _eCode );

		/**
		 * Gets the string description of an error code.
		 * 
		 * \param _eCode The error code whose description is to be gotten.
		 * \return Returns the text description for the given error.
		 **/
		static inline const char16_t *								ToStrPU16( NN9_ERRORS _eCode );

		/**
		 * Gets the string description of an error code.
		 * 
		 * \param _eCode The error code whose description is to be gotten.
		 * \return Returns the text description for the given error.
		 **/
		static inline std::u16string								ToStrU16( NN9_ERRORS _eCode );

		/**
		 * Gets the name of an error code.
		 * 
		 * \param _eCode The error code whose name is to be gotten.
		 * \return Returns the text name for the given error.
		 **/
		static inline const char8_t *								NamePU8( NN9_ERRORS _eCode );

		/**
		 * Gets the name of an error code.
		 * 
		 * \param _eCode The error code whose name is to be gotten.
		 * \return Returns the text name for the given error.
		 **/
		static inline const char16_t *								NamePU16( NN9_ERRORS _eCode );

	};


	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// DEFINITIONS
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	// == Functions.
	/**
	 * Converts an errno_t to one of our error codes.
	 * 
	 * \param _eCode The code to convert.
	 * \return Returns the converted error code.
	 **/
	inline NN9_ERRORS Errors::ErrNo_T_To_Native( errno_t _eCode ) {
		switch ( _eCode ) {
			case 0 : { return NN9_E_SUCCESS; }
			case EINVAL : { return NN9_E_INVALID_PARAMETER; }
			case EACCES : { return NN9_E_INVALID_PERMISSIONS; }
			case ENOENT : { return NN9_E_FILE_NOT_FOUND; }
			case EMFILE : { return NN9_E_TOO_MANY_FILES_OPENED; }
			case ENOMEM : { return NN9_E_OUT_OF_MEMORY; }
			case EEXIST : { return NN9_E_FILES_EXISTS; }
			case EOVERFLOW : { return NN9_E_FILE_ATTRIBUTE_TOO_LARGE; }
			case EPERM : { return NN9_E_OPERATION_NOT_PERMITTED; }
				
#ifdef EBADF
			case EBADF : { return NN9_E_INVALID_HANDLE; }
#endif

#ifdef ENFILE
			case ENFILE : { return NN9_E_TOO_MANY_FILES; }
#endif

#ifdef ENOSPC
			case ENOSPC : { return NN9_E_DISK_FULL; }
#endif

#ifdef EFBIG
			case EFBIG : { return NN9_E_FILE_TOO_LARGE; }
#endif

#ifdef ENAMETOOLONG
			case ENAMETOOLONG : { return NN9_E_INVALID_NAME; }
#endif

#ifdef ENOTDIR
			case ENOTDIR : { return NN9_E_INVALID_NAME; }
#endif

#ifdef EISDIR
			case EISDIR : { return NN9_E_INVALID_NAME; }
#endif

#ifdef ELOOP
			case ELOOP : { return NN9_E_INVALID_NAME; }
#endif

#ifdef EROFS
			case EROFS : { return NN9_E_INVALID_PERMISSIONS; }
#endif

#ifdef EBUSY
			case EBUSY : { return NN9_E_LOCK_VIOLATION; }
#endif

#ifdef ETXTBSY
			case ETXTBSY : { return NN9_E_LOCK_VIOLATION; }
#endif

#ifdef ENODEV
			case ENODEV : { return NN9_E_NO_SUCH_DEVICE; }
#endif

#ifdef ENXIO
			case ENXIO : { return NN9_E_NO_SUCH_DEVICE; }
#endif

#ifdef EAGAIN
			case EAGAIN : { return NN9_E_TIMEOUT; }
#endif

#ifdef EINTR
			case EINTR : { return NN9_E_TIMEOUT; }
#endif

#ifdef EIO
			case EIO : { return NN9_E_READ_FAILED; }
#endif

#ifdef EPIPE
			case EPIPE : { return NN9_E_WRITE_FAILED; }
#endif
				
			default : { return NN9_E_OTHER; }
		}
	}

	/**
	 * Converts a mz_zip_error error code to one of our error codes.
	 * 
	 * \param _zeCode The code to convert.
	 * \return Returns the converted error code.
	 **/
	inline NN9_ERRORS Errors::ZipError_To_Native( mz_zip_error _zeCode ) {
		switch ( _zeCode ) {
			case MZ_ZIP_NO_ERROR : { return NN9_E_SUCCESS; }
			case MZ_ZIP_UNDEFINED_ERROR : { return NN9_E_OTHER; }
			case MZ_ZIP_TOO_MANY_FILES : { return NN9_E_TOO_MANY_FILES; }
			case MZ_ZIP_FILE_TOO_LARGE : { return NN9_E_FILE_TOO_LARGE; }
			case MZ_ZIP_UNSUPPORTED_METHOD : { return NN9_E_INVALID_OPERATION; }
			case MZ_ZIP_UNSUPPORTED_ENCRYPTION : { return NN9_E_INVALID_ENCRYPTION; }
			case MZ_ZIP_UNSUPPORTED_FEATURE : { return NN9_E_UNSUPPORTED_FEATURE; }
			case MZ_ZIP_FAILED_FINDING_CENTRAL_DIR : { return NN9_E_FAILED_FINDING_CENTRAL_DIR; }
			case MZ_ZIP_NOT_AN_ARCHIVE : { return NN9_E_NOT_AN_ARCHIVE; }
			case MZ_ZIP_INVALID_HEADER_OR_CORRUPTED : { return NN9_E_INVALID_HEADER_OR_CORRUPTED; }
			case MZ_ZIP_UNSUPPORTED_MULTIDISK : { return NN9_E_UNSUPPORTED_MULTI_DISK; }
			case MZ_ZIP_DECOMPRESSION_FAILED : { return NN9_E_DECOMPRESSION_FAILED; }
			case MZ_ZIP_COMPRESSION_FAILED : { return NN9_E_COMPRESSION_FAILED; }
			case MZ_ZIP_UNEXPECTED_DECOMPRESSED_SIZE : { return NN9_E_UNEXPECTED_DECOMPRESSED_SIZE; }
			case MZ_ZIP_CRC_CHECK_FAILED : { return NN9_E_BAD_CRC; }
			case MZ_ZIP_UNSUPPORTED_CDIR_SIZE : { return NN9_E_UNSUPPORTED_CDIR_SIZE; }
			case MZ_ZIP_ALLOC_FAILED : { return NN9_E_OUT_OF_MEMORY; }
			case MZ_ZIP_FILE_OPEN_FAILED : { return NN9_E_OPEN_FAILED; }
			case MZ_ZIP_FILE_CREATE_FAILED : { return NN9_E_CREATE_FAILED; }
			case MZ_ZIP_FILE_WRITE_FAILED : { return NN9_E_WRITE_FAILED; }
			case MZ_ZIP_FILE_READ_FAILED : { return NN9_E_READ_FAILED; }
			case MZ_ZIP_FILE_CLOSE_FAILED : { return NN9_E_CLOSE_FAILED; }
			case MZ_ZIP_FILE_SEEK_FAILED : { return NN9_E_SEEK_FAILED; }
			case MZ_ZIP_FILE_STAT_FAILED : { return NN9_E_STAT_FAILED; }
			case MZ_ZIP_INVALID_PARAMETER : { return NN9_E_INVALID_PARAMETER; }
			case MZ_ZIP_INVALID_FILENAME : { return NN9_E_INVALID_NAME; }
			case MZ_ZIP_BUF_TOO_SMALL : { return NN9_E_INSUFFICIENT_BUFFER; }
			case MZ_ZIP_INTERNAL_ERROR : { return NN9_E_INTERNAL_ERROR; }
			case MZ_ZIP_FILE_NOT_FOUND : { return NN9_E_ARCHIVE_FILE_NOT_FOUND; }
			case MZ_ZIP_ARCHIVE_TOO_LARGE : { return NN9_E_ARCHIVE_TOO_LARGE; }
			case MZ_ZIP_VALIDATION_FAILED : { return NN9_E_VALIDATION_FAILED; }
			case MZ_ZIP_WRITE_CALLBACK_FAILED : { return NN9_E_WRITE_CALLBACK_FAILED; }
			default : { return NN9_E_OTHER; }
		}
	}

	/**
	 * Converts a libcurl error to one of our error codes.
	 * 
	 * \param _cCode The code to convert.
	 * \return Returns the converted error code.
	 **/
	inline NN9_ERRORS Errors::LibCurl_To_Native( CURLcode _cCode ) {
#define NN9_CHECK( ERROR )				case ERROR : { return NN9_E_ ## ERROR; }
		switch ( _cCode ) {
			case CURLE_OK : { return NN9_E_SUCCESS; }
			NN9_CHECK( CURLE_UNSUPPORTED_PROTOCOL )
			NN9_CHECK( CURLE_FAILED_INIT )
			NN9_CHECK( CURLE_URL_MALFORMAT )
			NN9_CHECK( CURLE_NOT_BUILT_IN )
			NN9_CHECK( CURLE_COULDNT_RESOLVE_PROXY )
			NN9_CHECK( CURLE_COULDNT_RESOLVE_HOST )
			NN9_CHECK( CURLE_COULDNT_CONNECT )
			NN9_CHECK( CURLE_WEIRD_SERVER_REPLY )
			NN9_CHECK( CURLE_REMOTE_ACCESS_DENIED )
			NN9_CHECK( CURLE_FTP_ACCEPT_FAILED )
			NN9_CHECK( CURLE_FTP_WEIRD_PASS_REPLY )
			NN9_CHECK( CURLE_FTP_ACCEPT_TIMEOUT )
			NN9_CHECK( CURLE_FTP_WEIRD_PASV_REPLY )
			NN9_CHECK( CURLE_FTP_WEIRD_227_FORMAT )
			NN9_CHECK( CURLE_FTP_CANT_GET_HOST )
			NN9_CHECK( CURLE_HTTP2 )
			NN9_CHECK( CURLE_FTP_COULDNT_SET_TYPE )
			NN9_CHECK( CURLE_PARTIAL_FILE )
			NN9_CHECK( CURLE_FTP_COULDNT_RETR_FILE )
			NN9_CHECK( CURLE_OBSOLETE20 )
			NN9_CHECK( CURLE_QUOTE_ERROR )
			NN9_CHECK( CURLE_HTTP_RETURNED_ERROR )
			NN9_CHECK( CURLE_WRITE_ERROR )
			NN9_CHECK( CURLE_OBSOLETE24 )
			NN9_CHECK( CURLE_UPLOAD_FAILED )
			NN9_CHECK( CURLE_READ_ERROR )
			NN9_CHECK( CURLE_OUT_OF_MEMORY )
			NN9_CHECK( CURLE_OPERATION_TIMEDOUT )
			NN9_CHECK( CURLE_OBSOLETE29 )
			NN9_CHECK( CURLE_FTP_PORT_FAILED )
			NN9_CHECK( CURLE_FTP_COULDNT_USE_REST )
			NN9_CHECK( CURLE_OBSOLETE32 )
			NN9_CHECK( CURLE_RANGE_ERROR )
			NN9_CHECK( CURLE_HTTP_POST_ERROR )
			NN9_CHECK( CURLE_SSL_CONNECT_ERROR )
			NN9_CHECK( CURLE_BAD_DOWNLOAD_RESUME )
			NN9_CHECK( CURLE_FILE_COULDNT_READ_FILE )
			NN9_CHECK( CURLE_LDAP_CANNOT_BIND )
			NN9_CHECK( CURLE_LDAP_SEARCH_FAILED )
			NN9_CHECK( CURLE_OBSOLETE40 )
			NN9_CHECK( CURLE_FUNCTION_NOT_FOUND )
			NN9_CHECK( CURLE_ABORTED_BY_CALLBACK )
			NN9_CHECK( CURLE_BAD_FUNCTION_ARGUMENT )
			NN9_CHECK( CURLE_OBSOLETE44 )
			NN9_CHECK( CURLE_INTERFACE_FAILED )
			NN9_CHECK( CURLE_OBSOLETE46 )
			NN9_CHECK( CURLE_TOO_MANY_REDIRECTS )
			NN9_CHECK( CURLE_UNKNOWN_OPTION )
			NN9_CHECK( CURLE_SETOPT_OPTION_SYNTAX )
			NN9_CHECK( CURLE_OBSOLETE50 )
			NN9_CHECK( CURLE_OBSOLETE51 )
			NN9_CHECK( CURLE_GOT_NOTHING )
			NN9_CHECK( CURLE_SSL_ENGINE_NOTFOUND )
			NN9_CHECK( CURLE_SSL_ENGINE_SETFAILED )
			NN9_CHECK( CURLE_SEND_ERROR )
			NN9_CHECK( CURLE_RECV_ERROR )
			NN9_CHECK( CURLE_OBSOLETE57 )
			NN9_CHECK( CURLE_SSL_CERTPROBLEM )
			NN9_CHECK( CURLE_SSL_CIPHER )
			NN9_CHECK( CURLE_PEER_FAILED_VERIFICATION )
			NN9_CHECK( CURLE_BAD_CONTENT_ENCODING )
			NN9_CHECK( CURLE_OBSOLETE62 )
			NN9_CHECK( CURLE_FILESIZE_EXCEEDED )
			NN9_CHECK( CURLE_USE_SSL_FAILED )
			NN9_CHECK( CURLE_SEND_FAIL_REWIND )
			NN9_CHECK( CURLE_SSL_ENGINE_INITFAILED )
			NN9_CHECK( CURLE_LOGIN_DENIED )
			NN9_CHECK( CURLE_TFTP_NOTFOUND )
			NN9_CHECK( CURLE_TFTP_PERM )
			NN9_CHECK( CURLE_REMOTE_DISK_FULL )
			NN9_CHECK( CURLE_TFTP_ILLEGAL )
			NN9_CHECK( CURLE_TFTP_UNKNOWNID )
			NN9_CHECK( CURLE_REMOTE_FILE_EXISTS )
			NN9_CHECK( CURLE_TFTP_NOSUCHUSER )
			NN9_CHECK( CURLE_OBSOLETE75 )
			NN9_CHECK( CURLE_OBSOLETE76 )
			NN9_CHECK( CURLE_SSL_CACERT_BADFILE )
			NN9_CHECK( CURLE_REMOTE_FILE_NOT_FOUND )
			NN9_CHECK( CURLE_SSH )
			NN9_CHECK( CURLE_SSL_SHUTDOWN_FAILED )
			NN9_CHECK( CURLE_AGAIN )
			NN9_CHECK( CURLE_SSL_CRL_BADFILE )
			NN9_CHECK( CURLE_SSL_ISSUER_ERROR )
			NN9_CHECK( CURLE_FTP_PRET_FAILED )
			NN9_CHECK( CURLE_RTSP_CSEQ_ERROR )
			NN9_CHECK( CURLE_RTSP_SESSION_ERROR )
			NN9_CHECK( CURLE_FTP_BAD_FILE_LIST )
			NN9_CHECK( CURLE_CHUNK_FAILED )
			NN9_CHECK( CURLE_NO_CONNECTION_AVAILABLE )
			NN9_CHECK( CURLE_SSL_PINNEDPUBKEYNOTMATCH )
			NN9_CHECK( CURLE_SSL_INVALIDCERTSTATUS )
			NN9_CHECK( CURLE_HTTP2_STREAM )
			NN9_CHECK( CURLE_RECURSIVE_API_CALL )
			NN9_CHECK( CURLE_AUTH_ERROR )
			NN9_CHECK( CURLE_HTTP3 )
			NN9_CHECK( CURLE_QUIC_CONNECT_ERROR )
			NN9_CHECK( CURLE_PROXY )
			NN9_CHECK( CURLE_SSL_CLIENTCERT )
			NN9_CHECK( CURLE_UNRECOVERABLE_POLL )
#if !defined( __APPLE__ )
			NN9_CHECK( CURLE_TOO_LARGE )
			NN9_CHECK( CURLE_ECH_REQUIRED )
#endif	// #if !defined( __APPLE__ )
			//NN9_CHECK( CURL_LAST )

			default : { return NN9_E_OTHER; }
		}
#undef NN9_CHECK
	}

#ifdef _WIN32
	/**
	 * Calls ::GetLastError() and converts the error code to one of our error codes.
	 * 
	 * \return Returns the converted error code.
	 **/
	inline NN9_ERRORS Errors::GetLastError_To_Native() {
		HRESULT dwError = ::GetLastError();
		DisplayLastError( dwError );
		switch ( dwError ) {
			// Success
			case ERROR_SUCCESS : { return NN9_E_SUCCESS; }

			// Memory Errors.
			case ERROR_NOT_ENOUGH_MEMORY : {}	NN9_FALLTHROUGH
			case ERROR_OUTOFMEMORY : { return NN9_E_OUT_OF_MEMORY; }

			// File (read) Errors
			case ERROR_FILE_NOT_FOUND : { return NN9_E_FILE_NOT_FOUND; }
			case ERROR_ACCESS_DENIED : { return NN9_E_INVALID_PERMISSIONS; }
			case ERROR_TOO_MANY_OPEN_FILES : { return NN9_E_TOO_MANY_FILES_OPENED; }
			case ERROR_ALREADY_EXISTS : {}		NN9_FALLTHROUGH
			case ERROR_FILE_EXISTS : { return NN9_E_FILES_EXISTS; }
			case ERROR_FILE_TOO_LARGE : { return NN9_E_FILE_TOO_LARGE; }
			case ERROR_PATH_NOT_FOUND : { return NN9_E_FILE_NOT_FOUND; }
			case ERROR_INVALID_NAME : { return NN9_E_INVALID_NAME; }
			case ERROR_SHARING_VIOLATION : {}	NN9_FALLTHROUGH
			case ERROR_LOCK_VIOLATION : { return NN9_E_LOCK_VIOLATION; }
			case ERROR_DISK_FULL : { return NN9_E_DISK_FULL; }
			case ERROR_BUFFER_OVERFLOW : {}		NN9_FALLTHROUGH
			case ERROR_INSUFFICIENT_BUFFER : { return NN9_E_INSUFFICIENT_BUFFER; }
			case ERROR_SHARING_BUFFER_EXCEEDED : { return NN9_E_SHARING_BUFFER_EXCEEDED; }
			case ERROR_NOT_READY : { return NN9_E_NOT_READY; }
			case ERROR_DEVICE_NOT_CONNECTED : { return NN9_E_DEVICE_NOT_CONNECTED; }
			case ERROR_NO_SUCH_DEVICE : { return NN9_E_NO_SUCH_DEVICE; }
			case ERROR_NETWORK_ACCESS_DENIED : { return NN9_E_NETWORK_ACCESS_DENIED; }
			case ERROR_NETWORK_BUSY : { return NN9_E_NETWORK_BUSY; }
			case ERROR_INVALID_HANDLE : { return NN9_E_INVALID_HANDLE; }
			case ERROR_TIMEOUT : { return NN9_E_TIMEOUT; }

#define NN9_CHECK( ERROR )				case ERROR : { return NN9_E_ ## ERROR; }
			NN9_CHECK( INET_E_DOWNLOAD_FAILURE )
			NN9_CHECK( INET_E_INVALID_CERTIFICATE )
			NN9_CHECK( WININET_E_OUT_OF_HANDLES )
			NN9_CHECK( WININET_E_TIMEOUT )
			NN9_CHECK( WININET_E_EXTENDED_ERROR )
			NN9_CHECK( WININET_E_INTERNAL_ERROR )
			NN9_CHECK( WININET_E_INVALID_URL )
			NN9_CHECK( WININET_E_UNRECOGNIZED_SCHEME )
			NN9_CHECK( WININET_E_NAME_NOT_RESOLVED )
			NN9_CHECK( WININET_E_PROTOCOL_NOT_FOUND )
			NN9_CHECK( WININET_E_INVALID_OPTION )
			NN9_CHECK( WININET_E_BAD_OPTION_LENGTH )
			NN9_CHECK( WININET_E_OPTION_NOT_SETTABLE )
			NN9_CHECK( WININET_E_SHUTDOWN )
			NN9_CHECK( WININET_E_LOGIN_FAILURE )
			NN9_CHECK( WININET_E_OPERATION_CANCELLED )
			NN9_CHECK( WININET_E_INCORRECT_HANDLE_TYPE )
			NN9_CHECK( WININET_E_INCORRECT_HANDLE_STATE )
			NN9_CHECK( WININET_E_NOT_PROXY_REQUEST )
			NN9_CHECK( WININET_E_CANNOT_CONNECT )
			NN9_CHECK( WININET_E_CONNECTION_ABORTED )
			NN9_CHECK( WININET_E_CONNECTION_RESET )
			NN9_CHECK( WININET_E_FORCE_RETRY )
			NN9_CHECK( WININET_E_INVALID_PROXY_REQUEST )
			NN9_CHECK( WININET_E_NEED_UI )
			NN9_CHECK( WININET_E_HANDLE_EXISTS )
			NN9_CHECK( WININET_E_SEC_CERT_DATE_INVALID )
			NN9_CHECK( WININET_E_SEC_CERT_CN_INVALID )
			NN9_CHECK( WININET_E_HTTP_TO_HTTPS_ON_REDIR )
			NN9_CHECK( WININET_E_HTTPS_TO_HTTP_ON_REDIR )
			NN9_CHECK( WININET_E_MIXED_SECURITY )
			NN9_CHECK( WININET_E_CHG_POST_IS_NON_SECURE )
			NN9_CHECK( WININET_E_POST_IS_NON_SECURE )
			NN9_CHECK( WININET_E_CLIENT_AUTH_CERT_NEEDED )
			NN9_CHECK( WININET_E_INVALID_CA )
			NN9_CHECK( WININET_E_CLIENT_AUTH_NOT_SETUP )
			NN9_CHECK( WININET_E_ASYNC_THREAD_FAILED )
			NN9_CHECK( WININET_E_REDIRECT_SCHEME_CHANGE )
			NN9_CHECK( WININET_E_DIALOG_PENDING )
			NN9_CHECK( WININET_E_RETRY_DIALOG )
			NN9_CHECK( WININET_E_SEC_CERT_ERRORS )
			NN9_CHECK( WININET_E_SEC_CERT_REV_FAILED )

			//NN9_CHECK( ERROR_WINHTTP_INTERNAL_ERROR )
#undef NN9_CHECK
			default : { return NN9_E_OTHER; }
		}

	}

	/**
	 * Displays the current ::GetLastError() error with a description.
	 * 
	 * \param _dwErr The error code to translate.  If -1, ::GetLastError() is called.
	 **/
	inline void Errors::DisplayLastError( DWORD _dwErr ) {
		DWORD dwError = _dwErr == DWORD( -1 ) ? _dwErr : ::GetLastError();

		LPVOID lpMsgBuf = nullptr;
#if 0
		HMODULE hWinInet = ::LoadLibraryExW( L"wininet.dll", NULL, LOAD_LIBRARY_AS_DATAFILE );

		if ( hWinInet ) {
			::FormatMessageW(
				FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_IGNORE_INSERTS,
				hWinInet,
				dwError,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				reinterpret_cast<LPWSTR>(&lpMsgBuf),
				0,
				NULL
			);

			::wprintf( L"Error %d: %s\n", dwError, reinterpret_cast<LPWSTR>(lpMsgBuf) );

			::LocalFree( lpMsgBuf );
			::FreeLibrary( hWinInet );
		}
		else 
#endif
		{
			// Fallback to system messages.
			::FormatMessageW(
				FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL,
				dwError,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				reinterpret_cast<LPWSTR>(&lpMsgBuf),
				0,
				NULL
			);

			::wprintf( L"Error %d: %s\n", dwError, reinterpret_cast<LPWSTR>(&lpMsgBuf) );

			::LocalFree( lpMsgBuf );
		}



		//LPWSTR lpMsgBuf = nullptr;

		//::FormatMessageW(
		//	FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		//	NULL,
		//	dwError,
		//	0, // Default language
		//	reinterpret_cast<LPWSTR>(&lpMsgBuf),
		//	0,
		//	NULL
		//);

		//::wprintf( L"Error %d: %s\n", dwError, lpMsgBuf );
		//::LocalFree( lpMsgBuf );
	}
#endif	// #ifdef _WIN32

	/**
	 * Gets the string description of an error code.
	 * 
	 * \param _eCode The error code whose description is to be gotten.
	 * \return Returns the text description for the given error.
	 **/
	inline const char8_t * Errors::ToStrPU8( NN9_ERRORS _eCode ) {
		switch ( _eCode ) {
#define NN9_E_ENUM( ENUM, TXT )			case ENUM : { return u8##TXT; }
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
		}
		return u8"Invalid error code.";
	}

	/**
	 * Gets the string description of an error code.
	 * 
	 * \param _eCode The error code whose description is to be gotten.
	 * \return Returns the text description for the given error.
	 **/
	inline std::u8string Errors::ToStrU8( NN9_ERRORS _eCode ) {
		switch ( _eCode ) {
#define NN9_E_ENUM( ENUM, TXT )			case ENUM : { return std::u8string( u8 ## TXT ); }
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
		}
		return std::u8string( u8"Invalid error code." );
	}

	/**
	 * Gets the string description of an error code.
	 * 
	 * \param _eCode The error code whose description is to be gotten.
	 * \return Returns the text description for the given error.
	 **/
	inline const char16_t * Errors::ToStrPU16( NN9_ERRORS _eCode ) {
		switch ( _eCode ) {
#define NN9_E_ENUM( ENUM, TXT )			case ENUM : { return u ## TXT; }
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
		}
		return u"Invalid error code.";
	}

	/**
	 * Gets the string description of an error code.
	 * 
	 * \param _eCode The error code whose description is to be gotten.
	 * \return Returns the text description for the given error.
	 **/
	inline std::u16string Errors::ToStrU16( NN9_ERRORS _eCode ) {
		switch ( _eCode ) {
#define NN9_E_ENUM( ENUM, TXT )			case ENUM : { return std::u16string( u ## TXT ); }
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
		}
		return std::u16string( u"Invalid error code." );
	}

	/**
	 * Gets the name of an error code.
	 * 
	 * \param _eCode The error code whose name is to be gotten.
	 * \return Returns the text name for the given error.
	 **/
	inline const char8_t * Errors::NamePU8( NN9_ERRORS _eCode ) {
		switch ( _eCode ) {
#define NN9_E_ENUM( ENUM, TXT )			case ENUM : { return &u8 ## # ENUM[6]; }
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
		}

		return u8"";
	}

	/**
	 * Gets the name of an error code.
	 * 
	 * \param _eCode The error code whose name is to be gotten.
	 * \return Returns the text name for the given error.
	 **/
	inline const char16_t * Errors::NamePU16( NN9_ERRORS _eCode ) {
		switch ( _eCode ) {
#define NN9_E_ENUM( ENUM, TXT )			case ENUM : { return &u ## # ENUM[6]; }
#include "NN9ErrorEnum.inl"
#undef NN9_E_ENUM
		}

		return u"";
	}

}	// namespace nn9

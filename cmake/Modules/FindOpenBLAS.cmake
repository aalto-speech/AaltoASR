# - Finds OpenBLAS and all dependencies
# Once done this will define
#  OPENBLAS_FOUND - System has OpenBLAS
#  OPENBLAS_INCLUDE_DIRS - The OpenBLAS include directories
#  OPENBLAS_LIBRARIES - The libraries needed to use Openblas


find_library(OPENBLAS_LIBRARY
    NAMES libopenblas.so libopenblas.lib libopenblas.a
    PATHS ${OPENBLAS_ROOT}/lib
	/usr/lib/openblas-base
	/usr/local/lib
    )

if(NOT OPENBLAS_IGNORE_HEADERS)
	find_path(OPENBLAS_INCLUDE_DIR
		NAMES openblas_config.h
		PATHS ${OPENBLAS_ROOT}/include /usr/include /usr/local/include
		)
endif()

if( ( OPENBLAS_LIBRARY STREQUAL "OPENBLAS_LIBRARY-NOTFOUND") OR ( OPENBLAS_INCLUDE_DIR STREQUAL "OPENBLAS_INCLUDE_DIR-NOTFOUND") )
    set(OPENBLAS_ROOT "" CACHE PATH "Path to the root of a OpenBLAS installation")
    set(OPENBLAS_FOUND 0)
    message(WARNING "OpenBLAS not found. Please try specifying OPENBLAS_ROOT")
else()
    set(OPENBLAS_FOUND 1)
    set(OPENBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
    set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
	if( CMAKE_COMPILER_IS_GNUCC)
		set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARIES} gfortran pthread)
	endif()

endif()

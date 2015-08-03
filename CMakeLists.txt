cmake_minimum_required(VERSION 2.8)
set(CMAKE_GENERATOR_CC gcc cc)
set(CMAKE_GENERATOR_CXX g++ c++)
PROJECT(AaltoASR)

include(ExternalProject)
include(CMakePushCheckState)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
ADD_DEFINITIONS(-DDLLIMPORT=)
ADD_DEFINITIONS(-std=gnu++0x)


Find_Package ( SNDFILE REQUIRED )

### Find ATLAS or OpenBLAS and set LAPACKPP_CONFIGURE if needed.
Find_Package ( BLAS QUIET )
IF(NOT BLAS_FOUND)
  Find_Package ( OpenBLAS )
  IF(NOT OPENBLAS_FOUND)
    MESSAGE(FATAL_ERROR "BLAS or OpenBLAS not found.")
  ELSE(NOT OPENBLAS_FOUND)
    set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
    set(LAPACK_LIBRARIES ${BLAS_LIBRARIES} "-lgfortran")
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
    check_function_exists("cheev_" LAPACK_LIBRARY_WORKS)
    mark_as_advanced(LAPACK_LIBRARY_WORKS)
    cmake_pop_check_state()
    if( NOT LAPACK_LIBRARY_WORKS )
      message(FATAL_ERROR "LAPACK features in OpenBLAS not found")
    else( )
      SET(LAPACK_FOUND TRUE)
      SET(LAPACKPP_CONFIGURE --disable-atlas --with-blas=openblas --with-lapack=openblas)
      message(STATUS "Working build of OpenBLAS with LAPACK found")
    endif( )
  ENDIF(NOT OPENBLAS_FOUND)
ELSE(NOT BLAS_FOUND)
  Find_Package ( LAPACK REQUIRED )
ENDIF(NOT BLAS_FOUND)


### MINGW CROSS-COMPILATION
IF(CROSS_MINGW)

  message(STATUS "Cross-compiling for Windows / MinGW")

  ADD_DEFINITIONS(-fPIC)
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

  SET(KISS_FFT 1)

  # Build also shared version of lapackpp for client applications.
  SET(LAPACKPP_CONFIGURE ${LAPACKPP_CONFIGURE} --enable-shared=yes --host=${CROSS_TARGET} CFLAGS=-I${CMAKE_PREFIX_PATH}/include LDFLAGS=-L${CMAKE_PREFIX_PATH}/lib)
  install(FILES ${CMAKE_BINARY_DIR}/vendor/lapackpp/lib/liblapackpp.dll.a DESTINATION lib)
  install(FILES ${CMAKE_BINARY_DIR}/vendor/lapackpp/lib/liblapackpp.la DESTINATION lib)

### NORMAL BUILD OR IN UNIX OR MINGW
ELSE(CROSS_MINGW)

  # Use position-independent code in x64 platform for some reason.
  IF(UNIX AND NOT WIN32)
    FIND_PROGRAM(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
    IF(CMAKE_UNAME)
      EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
      SET(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} CACHE INTERNAL "processor type (i386 and x86_64)")
      IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        ADD_DEFINITIONS(-fPIC)
      ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    ENDIF(CMAKE_UNAME)
  ENDIF(UNIX AND NOT WIN32)

ENDIF(CROSS_MINGW)
### END SYSTEM DEPENDENT PART


# Build patched LapackPP last, so that it will use the flags in ${LAPACKPP_CONFIGURE}.
ExternalProject_Add(lapackpp_ext
  PREFIX ${CMAKE_BINARY_DIR}/vendor/lapackpp
  URL http://downloads.sourceforge.net/project/lapackpp/lapackpp-2.5.4.tar.gz
  URL_MD5 9ad74fb73fbcbf378c36a19a5dccbda7
  PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/vendor/lapackpp-2.5.4.ics.patch
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-static=yes ${LAPACKPP_CONFIGURE} CXXFLAGS=-fPIC
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND make install
  CXXFLAGS=-fPIC
)
SET(LapackPP_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/vendor/lapackpp/include
                          ${CMAKE_BINARY_DIR}/vendor/lapackpp/include/lapackpp)
SET(LapackPP_LIBRARIES LapackPP)
add_library( LapackPP UNKNOWN IMPORTED )
set_property( TARGET LapackPP PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/vendor/lapackpp/lib/liblapackpp.a )
add_dependencies( LapackPP lapackpp_ext )
install(DIRECTORY ${CMAKE_BINARY_DIR}/vendor/lapackpp/include/lapackpp DESTINATION include)
install(FILES ${CMAKE_BINARY_DIR}/vendor/lapackpp/lib/liblapackpp.a DESTINATION lib)


option(DISABLE_SWIG "Disable building swig/python interface for decoder")
option(DISABLE_TOOLS "Disable building tools")

add_subdirectory( decoder )
if(NOT DISABLE_TOOLS)
  add_subdirectory( tools )
ENDIF(NOT DISABLE_TOOLS)
add_subdirectory( aku )

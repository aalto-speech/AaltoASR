cmake_minimum_required(VERSION 2.8)
set(CMAKE_GENERATOR_CC gcc cc)
set(CMAKE_GENERATOR_CXX g++ c++)
PROJECT(AaltoASR)

include(ExternalProject)

# with -fPIC
IF(UNIX AND NOT WIN32)
  FIND_PROGRAM(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
  IF(CMAKE_UNAME)
    EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
    SET(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} CACHE INTERNAL
"processor type (i386 and x86_64)")
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      ADD_DEFINITIONS(-fPIC)
    ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  ENDIF(CMAKE_UNAME)
ENDIF(UNIX AND NOT WIN32)

IF(CROSS_MINGW)
  message(STATUS "Cross-compiling for Windows / MinGW")

  SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${CROSS_DEPS}/include)
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CROSS_DEPS}/lib)
  SET(LAPACKPP_CONFIGURE --disable-atlas --with-blas=openblas --with-lapack=openblas --host=${CROSS_TARGET} CFLAGS=-I${CROSS_DEPS}/include LDFLAGS=-L${CROSS_DEPS}/lib)
  SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/cross-mingw")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
  SET(KISS_FFT 1)
  ADD_DEFINITIONS("-DDLLIMPORT=")

  ExternalProject_Add(libogg
    PREFIX ${CROSS_DEPS}
    URL http://downloads.xiph.org/releases/ogg/libogg-1.3.2.tar.gz
    URL_MD5 b72e1a1dbadff3248e4ed62a4177e937
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET}
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install
  )

  ExternalProject_add(libvorbis
    DEPENDS libogg
    PREFIX ${CROSS_DEPS}
    URL http://downloads.xiph.org/releases/vorbis/libvorbis-1.3.4.tar.gz
    URL_MD5 8851c593a52d1ef9c526d95174873852
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install
  )

  ExternalProject_Add(flac
    DEPENDS libogg
    PREFIX ${CROSS_DEPS}
    URL http://downloads.xiph.org/releases/flac/flac-1.2.1.tar.gz
    URL_MD5 153c8b15a54da428d1f0fadc756c22c7
    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/cmake/cross-mingw/flac-1.2.1.patch
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install
  )

  ExternalProject_Add(sndfile
    DEPENDS libogg libvorbis flac
    PREFIX ${CROSS_DEPS}
    URL http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
    URL_MD5 e2b7bb637e01022c7d20f95f9c3990a2
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} --disable-sqlite --disable-alsa CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install
  )

  IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET(OPENBLAS_BITS 64)
  ELSE(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET(OPENBLAS_BITS 32)
  ENDIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  ExternalProject_Add(OpenBLAS
    PREFIX ${CROSS_DEPS}
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
    GIT_TAG v0.2.11
    CONFIGURE_COMMAND true
    BUILD_COMMAND make BINARY=${OPENBLAS_BITS} CC=${CMAKE_C_COMPILER} FC=${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32-gfortran HOSTCC=gcc TARGET=CORE2
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make install PREFIX=${CROSS_DEPS}
  )
ENDIF(CROSS_MINGW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

ADD_DEFINITIONS(-std=gnu++0x)

ExternalProject_Add(lapackpp_ext
   PREFIX ${CMAKE_BINARY_DIR}/vendor/lapackpp
   URL http://downloads.sourceforge.net/project/lapackpp/lapackpp-2.5.4.tar.gz
   URL_MD5 9ad74fb73fbcbf378c36a19a5dccbda7
  # SOURCE_DIR vendor/lapackpp-2.5.4
   PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/vendor/lapackpp-2.5.4.ics.patch
   CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> CXXFLAGS=-fPIC
   BUILD_COMMAND make
   BUILD_IN_SOURCE 1
   INSTALL_COMMAND make install
   CXXFLAGS=-fPIC
)

add_library( LapackPP UNKNOWN IMPORTED )
set_property( TARGET LapackPP PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/vendor/lapackpp/lib/liblapackpp.a )
add_dependencies( LapackPP lapackpp_ext )

set(LapackPP_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/vendor/lapackpp/include
                          ${CMAKE_BINARY_DIR}/vendor/lapackpp/include/lapackpp )
set(LapackPP_LIBRARIES LapackPP)

install(DIRECTORY ${CMAKE_BINARY_DIR}/vendor/lapackpp/include/lapackpp DESTINATION include)
install(FILES ${CMAKE_BINARY_DIR}/vendor/lapackpp/lib/liblapackpp.a DESTINATION lib)

option(DISABLE_SWIG "Disable building swig/python interface for decoder")

add_subdirectory( decoder )
IF(NOT CROSS_MINGW)
  add_subdirectory( tools )
ENDIF(NOT CROSS_MINGW)
add_subdirectory( aku )

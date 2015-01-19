# MinGW cross compilation settings for the AaltoASR project

if(NOT CROSS_DEPS)
  message(FATAL_ERROR "CROSS_DEPS variable not set; unable to find dependencies")
endif(NOT CROSS_DEPS)

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${CROSS_DEPS}/include)
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CROSS_DEPS}/lib)
SET(LAPACKPP_CONFIGURE --disable-atlas --with-blas=openblas --with-lapack=openblas --host=${CROSS_TARGET} CFLAGS=-I${CROSS_DEPS}/include LDFLAGS=-L${CROSS_DEPS}/lib)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/mingw/")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
SET(KISS_FFT 1)
ADD_DEFINITIONS("-DDLLIMPORT=")

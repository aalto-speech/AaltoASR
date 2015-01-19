# FindBLAS module configured to use the internal OpenBLAS library

include(CheckFunctionExists)
include(CheckFortranFunctionExists)
include(CMakePushCheckState)

get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
if( NOT ((_LANGUAGES_ MATCHES C) OR (_LANGUAGES_ MATCHES CXX)) )
  message(FATAL_ERROR "Cross-compiler FindBLAS only supports C or C++")
endif( )

find_library(BLAS_LIBRARIES "libopenblas.a")

if( BLAS_LIBRARIES )
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  check_function_exists("sgemm_" BLAS_LIBRARY_WORKS)
  mark_as_advanced(BLAS_LIBRARY_WORKS)
  cmake_pop_check_state()
endif( BLAS_LIBRARIES )

if( NOT BLAS_LIBRARY_WORKS )
  message(FATAL_ERROR "Working build of OpenBLAS not found")
else( )
  SET(BLAS_FOUND TRUE)
  message(STATUS "Working build of OpenBLAS found")
endif( )

# FindLAPACK module configured to use the internal OpenBLAS library

include(CheckFunctionExists)
include(CheckFortranFunctionExists)
include(CMakePushCheckState)

get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
if( NOT ((_LANGUAGES_ MATCHES C) OR (_LANGUAGES_ MATCHES CXX)) )
  message(FATAL_ERROR "Cross-compiler FindLAPACK only supports C or C++")
endif( )

if( BLAS_LIBRARIES )
  set(LAPACK_LIBRARIES ${BLAS_LIBRARIES} "-lgfortran")
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
  check_function_exists("cheev_" LAPACK_LIBRARY_WORKS)
  mark_as_advanced(LAPACK_LIBRARY_WORKS)
  cmake_pop_check_state()
endif( BLAS_LIBRARIES )

if( NOT LAPACK_LIBRARY_WORKS )
  message(FATAL_ERROR "LAPACK features in OpenBLAS not found")
else( )
  SET(LAPACK_FOUND TRUE)
  message(STATUS "Working build of OpenBLAS with LAPACK found")
endif( )

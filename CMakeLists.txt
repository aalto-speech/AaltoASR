cmake_minimum_required(VERSION 2.8)
set(CMAKE_GENERATOR_CC gcc cc)
set(CMAKE_GENERATOR_CXX g++ c++)
PROJECT(AaltoASR)

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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

ADD_DEFINITIONS(-std=gnu++0x)

include(ExternalProject)
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
add_subdirectory( tools )
add_subdirectory( aku )






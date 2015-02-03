cmake_minimum_required(VERSION 2.8)
set(CMAKE_GENERATOR_CC gcc cc)
set(CMAKE_GENERATOR_CXX g++ c++)
PROJECT (aku)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")


### MINGW CROSS-COMPILATION
### Make sure we find the cross-compiled dependencies.
IF(CROSS_MINGW)

  SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../cmake/cross-mingw)
  SET(CMAKE_INCLUDE_PATH ${CROSS_DEPS}/include ${CMAKE_INCLUDE_PATH})
  SET(CMAKE_LIBRARY_PATH ${CROSS_DEPS}/lib ${CMAKE_LIBRARY_PATH})
  SET(LapackPP_INCLUDE_DIRS ${CROSS_DEPS}/include ${CROSS_DEPS}/include/lapackpp)
  SET(LapackPP_LIBRARIES ${CROSS_DEPS}/lib/liblapackpp.a)

  ADD_DEFINITIONS(-fPIC)
  ADD_DEFINITIONS(-DDLLIMPORT=)

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


MESSAGE(STATUS "CMAKE_MODULE_PATH: " "${CMAKE_MODULE_PATH}")
MESSAGE(STATUS "CMAKE_INCLUDE_PATH: " "${CMAKE_INCLUDE_PATH}")
MESSAGE(STATUS "CMAKE_LIBRARY_PATH: " "${CMAKE_LIBRARY_PATH}")

Find_Package ( SNDFILE REQUIRED )
Find_Package ( BLAS REQUIRED )
Find_Package ( LAPACK REQUIRED )

MESSAGE(STATUS "SNDFILE_LIBRARIES: " "${SNDFILE_LIBRARIES}")
MESSAGE(STATUS "LapackPP_INCLUDE_DIRS: " "${LapackPP_INCLUDE_DIRS}")
MESSAGE(STATUS "LapackPP_LIBRARIES: " "${LapackPP_LIBRARIES}")
MESSAGE(STATUS "LAPACK_LIBRARIES: " "${LAPACK_LIBRARIES}")
MESSAGE(STATUS "BLAS_LIBRARIES: " "${BLAS_LIBRARIES}")

link_libraries (
    ${LapackPP_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

include_directories (
    ${LapackPP_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)

set(AKUSOURCES 
    FeatureGenerator.cc 
    FeatureModules.cc 
    AudioReader.cc 
    ModuleConfig.cc 
    HmmSet.cc
    PhnReader.cc 
    ModelModules.cc 
    SpeakerConfig.cc
    Recipe.cc 
    conf.cc 
    io.cc 
    str.cc 
    endian.cc 
    Distributions.cc
    LinearAlgebra.cc 
    HmmNetBaumWelch.cc
    Lattice.cc 
    Viterbi.cc 
    PhonePool.cc
    MllrTrainer.cc 
    ziggurat.cc 
    mtw.cc 
    LmbfgsOptimize.cc
    RegClassTree.cc
    SegErrorEvaluator.cc 
    util.cc
    PhoneProbsToolbox.cc
    ${LapackPP_HEADER}
)

SET(KISS_FFT 0 CACHE BOOL "Use KissFFT instead of FFTW")

IF(NOT KISS_FFT)
  Find_Package ( FFTW )
  IF(NOT FFTW_FOUND)
    MESSAGE("FFTW not Found, using KissFFT instead")
  ENDIF(NOT FFTW_FOUND)
ENDIF(NOT KISS_FFT)

IF(FFTW_FOUND AND NOT KISS_FFT)
  link_libraries(
      ${FFTW_LIBRARIES}
  )
  include_directories(
      ${FFTW_INCLUDE_DIRS}
  )
  MESSAGE("FFTW is used")
ELSE(FFTW_FOUND AND NOT KISS_FFT)
  add_definitions(-DKISS_FFT)
  include_directories( ../vendor/kiss_fft )
  set( AKUSOURCES "${AKUSOURCES}" ../vendor/kiss_fft/kiss_fft.c ../vendor/kiss_fft/kiss_fftr.c ) 
ENDIF(FFTW_FOUND AND NOT KISS_FFT)

add_library( aku ${AKUSOURCES} )
add_dependencies(aku lapackpp_ext)

set(AKU_CMDS feacat feadot feanorm phone_probs segfea vtln quanteq stats estimate align tie dur_est gconvert mllr logl gcluster lda optmodel cmpmodel combine_stats regtree clsstep clskld opt_ebw_d )

foreach(AKU_CMD ${AKU_CMDS})
    add_executable ( ${AKU_CMD} ${AKU_CMD}.cc )
    target_link_libraries ( ${AKU_CMD} aku )
    install(TARGETS ${AKU_CMD} DESTINATION bin)
endforeach(AKU_CMD)

file(GLOB AKU_HEADERS "*.hh") 
install(FILES ${AKU_HEADERS} DESTINATION include)
install(TARGETS aku DESTINATION lib)
if(NOT DISABLE_SWIG)
  add_subdirectory( swig )
endif(NOT DISABLE_SWIG )

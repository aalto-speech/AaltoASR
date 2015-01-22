PROJECT (aku)

Find_Package ( SNDFILE REQUIRED )

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

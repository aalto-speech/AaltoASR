Find_Package ( FFTW REQUIRED )
Find_Package ( SNDFILE REQUIRED )
Find_Package ( BLAS REQUIRED )
Find_Package ( LAPACK REQUIRED )

link_libraries (
    ${FFTW_LIBRARIES}
    ${LapackPP_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

include_directories (
    ${FFTW_INCLUDE_DIRS}
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
    ${LapackPP_HEADER}
)

add_library( aku ${AKUSOURCES} )
add_dependencies(aku lapackpp_ext)

set(AKU_CMDS feadot feanorm phone_probs segfea vtln quanteq stats estimate align tie dur_est gconvert mllr logl gcluster lda optmodel cmpmodel combine_stats regtree clsstep clskld opt_ebw_d )

foreach(AKU_CMD ${AKU_CMDS})
    add_executable ( ${AKU_CMD} ${AKU_CMD}.cc )
    target_link_libraries ( ${AKU_CMD} aku )
endforeach(AKU_CMD)




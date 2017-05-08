add_subdirectory( fsalm )
add_subdirectory( misc )
add_subdirectory( morpheus )
if(NOT DISABLE_SWIG)
  add_subdirectory( swig )
endif(NOT DISABLE_SWIG)
include_directories(".")
include_directories("/opt/openfst/include")

set(DECODERSOURCES
  GramSorter.cc
  Hmm.cc
  HTKLatticeGrammar.cc
  LMHistory.cc
  LnaReaderCircular.cc
  NowayHmmReader.cc
  OneFrameAcoustics.cc
  TPLexPrefixTree.cc
  TPNowayLexReader.cc
  Token.cc
  TokenPassSearch.cc
  Toolbox.cc
  TreeGram.cc
  TreeGramArpaReader.cc
  Vocabulary.cc
  ArpaReader.cc
  InterTreeGram.cc
  WordClasses.cc
  FstAcoustics.cc
  OpenFstSearch.cc
)

ADD_DEFINITIONS(-std=gnu++0x)
add_library( decoder ${DECODERSOURCES} )
add_executable ( arpa2bin arpa2bin.cc )
add_executable ( bin2arpa bin2arpa.cc )
add_executable ( hmm2fsm hmm2fsm.cc )
target_link_libraries ( arpa2bin decoder fsalm misc)
target_link_libraries ( bin2arpa decoder fsalm misc)
target_link_libraries ( hmm2fsm decoder )

install(TARGETS arpa2bin bin2arpa DESTINATION bin)
file(GLOB DECODER_HEADERS "*.hh")
install(FILES ${DECODER_HEADERS} DESTINATION include)
install(TARGETS decoder DESTINATION lib)

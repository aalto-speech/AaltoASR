add_subdirectory( fsalm )
add_subdirectory( misc )
add_subdirectory( morpheus )
add_subdirectory( swig )
include_directories(".")

set(DECODERSOURCES 
  Expander.cc
  GramSorter.cc
  Hmm.cc
  HTKLatticeGrammar.cc
  Lexicon.cc
  LMHistory.cc
  LnaReaderCircular.cc
  NowayHmmReader.cc
  NowayLexiconReader.cc
  OneFrameAcoustics.cc
  Search.cc
  TPLexPrefixTree.cc
  TPNowayLexReader.cc
  TokenPassSearch.cc
  Toolbox.cc
  TreeGram.cc
  TreeGramArpaReader.cc
  Vocabulary.cc
  ArpaReader.cc
  InterTreeGram.cc
  WordClasses.cc
)

add_library( decoder ${DECODERSOURCES} )
add_executable ( arpa2bin arpa2bin.cc )
add_executable ( bin2arpa bin2arpa.cc )
target_link_libraries ( arpa2bin decoder fsalm misc )
target_link_libraries ( bin2arpa decoder fsalm misc )

add_executable (
    lattice_rescore
    conf.cc
    conf.hh
    Endian.cc
    Endian.hh
    io.cc
    io.hh
    Lattice.cc
    Lattice.hh
    lattice_rescore.cc
    Rescore.cc
    Rescore.hh
    str.cc
    str.hh
    TreeGram.cc
    TreeGram.hh
    util.hh
    Vocabulary.cc
    Vocabulary.hh
)

install(TARGETS lattice_rescore DESTINATION bin)

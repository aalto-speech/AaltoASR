add_executable (
    morph_lattice
    conf.cc
    conf.hh
    io.cc
    io.hh
    Latticer.cc
    Latticer.hh
    morph_lattice.cc
    MorphSet.cc
    MorphSet.hh
    str.cc
    str.hh
    util.hh
)

install(TARGETS morph_lattice DESTINATION bin)

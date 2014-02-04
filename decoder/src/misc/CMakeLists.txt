add_library( misc Endian.cc io.cc tools.cc conf.cc )
install(TARGETS misc DESTINATION lib)
install(FILES types.hh SymbolMap.hh str.hh DESTINATION include/misc )

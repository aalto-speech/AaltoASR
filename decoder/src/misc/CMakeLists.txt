add_library( misc Endian.cc io.cc tools.cc conf.cc )
install(TARGETS misc DESTINATION lib)
file(GLOB MISC_HEADERS "*.hh") 
install(FILES ${MISC_HEADERS} DESTINATION include/misc)

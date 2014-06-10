include_directories("..")

add_library( fsalm LM.cc ArpaReader.cc )
install(TARGETS fsalm DESTINATION lib)
file(GLOB FSALM_HEADERS "*.hh") 
install(FILES ${FSALM_HEADERS} DESTINATION include/fsalm)

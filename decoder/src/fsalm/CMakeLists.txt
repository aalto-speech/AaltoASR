include_directories("..")

add_executable(fsalm-convert lm.cc)
target_link_libraries(fsalm-convert fsalm misc)
add_library(fsalm LM.cc ArpaReader.cc)
install(TARGETS fsalm-convert DESTINATION bin)
install(TARGETS fsalm DESTINATION lib)
file(GLOB FSALM_HEADERS "*.hh") 
install(FILES ${FSALM_HEADERS} DESTINATION include/fsalm)

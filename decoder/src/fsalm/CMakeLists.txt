include_directories("..")

add_executable(lm lm.cc)
target_link_libraries(lm fsalm misc)
add_library(fsalm LM.cc ArpaReader.cc)
install(TARGETS lm DESTINATION bin)
install(TARGETS fsalm DESTINATION lib)
file(GLOB FSALM_HEADERS "*.hh") 
install(FILES ${FSALM_HEADERS} DESTINATION include/fsalm)

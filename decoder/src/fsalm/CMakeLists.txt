include_directories("..")

add_library( fsalm LM.cc ArpaReader.cc )
install(TARGETS fsalm DESTINATION lib)
install(FILES LM.hh FixedArray.hh DESTINATION include/fsalm)

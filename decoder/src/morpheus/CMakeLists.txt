include_directories("..")
add_executable ( morpheus morpheus.cc )
target_link_libraries ( morpheus fsalm misc )
install (TARGETS morpheus DESTINATION bin)

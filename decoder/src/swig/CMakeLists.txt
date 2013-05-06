include_directories("..")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)

include_directories(${PYTHON_INCLUDE_PATH})

set_source_files_properties(Decoder.i PROPERTIES CPLUSPLUS ON)

swig_add_module(Decoder python Decoder.i)

swig_link_libraries(Decoder ${PYTHON_LIBRARIES} decoder fsalm misc)

install(
TARGETS ${SWIG_MODULE_Decoder_REAL_NAME}
DESTINATION lib/site-packages)

install(
FILES ${CMAKE_BINARY_DIR}/decoder/src/swig/Decoder.py
DESTINATION lib/site-packages)

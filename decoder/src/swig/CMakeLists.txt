include_directories("..")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)

include_directories(${PYTHON_INCLUDE_PATH})

set_source_files_properties(Decoder.i PROPERTIES CPLUSPLUS ON)

swig_add_module(Decodermodule python Decoder.i)

swig_link_libraries(Decodermodule ${PYTHON_LIBRARIES} decoder fsalm misc)

include_directories("..")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories("/opt/openfst/include")

set_source_files_properties(Decoder.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(OpenFstDecoder.i PROPERTIES CPLUSPLUS ON)

swig_add_module(Decoder python Decoder.i)
swig_add_module(OpenFstDecoder python OpenFstDecoder.i)

swig_link_libraries(Decoder ${PYTHON_LIBRARIES} decoder fsalm misc)
swig_link_libraries(OpenFstDecoder ${PYTHON_LIBRARIES} decoder fsalm misc)

install(
  TARGETS ${SWIG_MODULE_Decoder_REAL_NAME} ${SWIG_MODULE_OpenFstDecoder_REAL_NAME}
  DESTINATION lib/site-packages)

install(
  FILES ${CMAKE_BINARY_DIR}/decoder/src/swig/Decoder.py ${CMAKE_BINARY_DIR}/decoder/src/swig/OpenFstDecoder.py
  DESTINATION lib/site-packages)

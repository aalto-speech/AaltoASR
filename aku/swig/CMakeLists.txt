include_directories("..")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)

include_directories(${PYTHON_INCLUDE_PATH})

set_source_files_properties(PPToolbox.i PROPERTIES CPLUSPLUS ON)
swig_add_module(PPToolbox python PPToolbox.i)
swig_link_libraries(PPToolbox ${PYTHON_LIBRARIES} aku)

# Override the standard cmake swig generator to create tmp file that
# we can later fix
add_custom_command(
  OUTPUT "${swig_generated_file_fullname}.tmp"
  # Let's create the ${swig_outdir} at execution time, in case dir contains $(OutDir)
  COMMAND ${CMAKE_COMMAND} -E make_directory ${swig_outdir}
  COMMAND "${SWIG_EXECUTABLE}"
  ARGS -python
  ${swig_source_file_flags}
  ${CMAKE_SWIG_FLAGS}
  -outdir ${swig_outdir}
  ${swig_special_flags}
  ${swig_extra_flags}
  ${swig_include_dirs}
  -o "${swig_generated_file_fullname}.tmp"
  "${swig_source_file_fullname}"
  MAIN_DEPENDENCY "${swig_source_file_fullname}"
  DEPENDS ${SWIG_MODULE_${name}_EXTRA_DEPS}
  COMMENT "Swig source")

add_custom_target(FixDefs python ${CMAKE_CURRENT_SOURCE_DIR}/add_undefs.py < ${swig_generated_file_fullname}.tmp > ${swig_generated_file_fullname}
  OUTPUT "${swig_generated_file_fullname}"
  DEPENDS ${swig_generated_file_fullname}.tmp
  COMMENT "Fixing defs for conflicting symbols in lapack and python")

install(
TARGETS ${SWIG_MODULE_PPToolbox_REAL_NAME}
DESTINATION lib/site-packages)

install(
FILES ${CMAKE_BINARY_DIR}/aku/swig/PPToolbox.py
DESTINATION lib/site-packages)


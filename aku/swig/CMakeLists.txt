include_directories("..")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)

include_directories(${PYTHON_INCLUDE_PATH})

set_source_files_properties(PPToolbox.i PROPERTIES CPLUSPLUS ON)
swig_add_module(PPToolbox python PPToolbox.i)
swig_link_libraries(PPToolbox ${PYTHON_LIBRARIES} aku)

add_custom_target(TmpCWrap mv ${swig_generated_file_fullname} tmp.cxx 
  DEPENDS "${swig_generated_file_fullname}"
  COMMENT "mv ${swig_generated_file_fullname} tmp.cxx")
add_custom_target(FixDefs python ${CMAKE_CURRENT_SOURCE_DIR}/add_undefs.py < tmp.cxx > ${swig_generated_file_fullname}
  DEPENDS TmpCWrap
  COMMENT "Fixing defs for conflicting symbols in lapack and python")
add_dependencies(${SWIG_MODULE_PPToolbox_REAL_NAME} FixDefs) 


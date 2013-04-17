#  FFTW_INCLUDE_DIRS   - where to find sndfile.h
#  FFTW_LIBRARIES      - List of libraries when using sndfile
#  FFTW_FOUND          - True if sndfile found


FIND_PATH(FFTW_INCLUDE_DIR NAMES fftw3.h)
MARK_AS_ADVANCED(FFTW_INCLUDE_DIR)

# Look for the library.
FIND_LIBRARY(FFTW_LIBRARY NAMES fftw3)
MARK_AS_ADVANCED(FFTW_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to 
# TRUE if all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW
                                  REQUIRED_VARS FFTW_LIBRARY
                                  FFTW_INCLUDE_DIR )

IF(FFTW_FOUND)
  SET(FFTW_LIBRARIES ${FFTW_LIBRARY} CACHE FILEPATH "FFTW libraries")
  SET(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR} CACHE PATH "FFTW Inlude dirs")
ENDIF(FFTW_FOUND)

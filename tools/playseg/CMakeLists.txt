Find_Package ( SDL REQUIRED )
Find_Package ( SNDFILE REQUIRED )

link_libraries (
    ${SDL_LIBRARY}
)

include_directories (
    ${SNDFILE_INCLUDE_DIRS}
)

add_executable (playseg AudioPlayer.cc conf.cc io.cc playseg.cc str.cc AudioPlayer.hh str.hh util.hh conf.hh io.hh)

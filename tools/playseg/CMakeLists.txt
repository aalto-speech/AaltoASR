Find_Package ( SDL REQUIRED )

# Workaround for the non-working REQUIRED flag
if ( NOT SDL_FOUND )
   message ( FATAL_ERROR "SDL not found!" )
endif ( NOT SDL_FOUND )

link_libraries (
    ${SDL_LIBRARY}
)

add_executable (playseg AudioPlayer.cc conf.cc io.cc playseg.cc str.cc AudioPlayer.hh str.hh util.hh conf.hh io.hh)

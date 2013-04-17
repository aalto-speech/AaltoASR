The Aalto ASR tools are build with the cmake build system.

## Requirements

### Required packages for Ubuntu (and other Debian-based distributions)

    build-essential
    cmake

    swig
    libsndfile1-dev
    libsdl1.2-dev
    libfftw3-dev
    liblapack-dev
    python2.7-dev
    gfortran


### Required packages for Scientific Linux (and other Redhat-based distributions)

    yum groupinstall "Development Tools"
    cmake28 (EPEL)

    SDL-devel
    python-devel
    lapack-devel
    fftw-devel
    libsndfile-devel


## Standard build procedure

Make sure the requirements are met, see REQUIREMENTS above

    git clone https://github.com/aalto-speech/AaltoASR.git
    cd AaltoASR
    mkdir build
    cd build 
    cmake ..
    make

Instead of make all, also only a subproject or executable can be build, e.g.:
   
    make aku

or 
    make playseg

After this all a 

    make install

will install the binaries and libraries on the correct places. This location can be changed with the option -D CMAKE_INSTALL_PREFIX=/path to the cmake command.

For a debug build, add -D CMAKE_BUILD_TYPE=Debug to the cmake command (before the path).


## Creating an Eclipse project

If you would like to edit/debug the source code in Eclipse, you can follow the instructions given on http://www.vtk.org/Wiki/Eclipse_CDT4_Generator

In short

    git clone https://github.com/aalto-speech/AaltoASR.git
    cd AaltoASR
    cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug .

And import the resulting project in eclipse.

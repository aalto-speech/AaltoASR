The Aalto ASR tools are build with the cmake build system.


## Requirements

### Required packages for Ubuntu (and other Debian-based distributions)

    build-essential
    cmake

    swig
    libsndfile1-dev
    libsdl1.2-dev
    liblapack-dev
    python2.7-dev
    gfortran

    libfftw3-dev (optional. If not provided KissFFT is used)

### Required packages for Scientific Linux (and other Redhat-based distributions)

    yum groupinstall "Development Tools"
    cmake28 (EPEL)

    SDL-devel
    python-devel
    lapack-devel
    libsndfile-devel
    
    fftw-devel (optional. If not provided KissFFT is used)


## Building under Unix-like operating systems

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

will install the binaries and libraries on the correct places. This location can be changed with the option -DCMAKE_INSTALL_PREFIX=/path to the cmake command.

For a debug build, add -DCMAKE_BUILD_TYPE=Debug to the cmake command. For an optimized build add -DCMAKE_BUILD_TYPE=Release.

Normally, if available, the FFTW library is used, otherwise the KissFFT library is used. To force the KissFFT library, add -DKISS_FFT=1 to the cmake command.


## Creating an Eclipse project

If you would like to edit/debug the source code in Eclipse, you can follow the instructions given on http://www.vtk.org/Wiki/Eclipse_CDT4_Generator

In short

    git clone https://github.com/aalto-speech/AaltoASR.git
    cd AaltoASR
    cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug .

And import the resulting project in eclipse.


## Building under Windows and MinGW

1. Install MinGW and MSYS environment.

Follow the instructions on this page: http://ingar.satgnu.net/devenv/mingw32/base.html

2. Open MSYS shell and load the correct build environment.

For a 32-bit build, use

    source /local32/etc/profile.local

For a 64-bit build, use

    source /local64/etc/profile.local

3. Install SDL.

    cd ${LOCALSOURCEDIR} && \
    wget -c http://www.libsdl.org/release/SDL-1.2.15.tar.gz && \
    cd ${LOCALBUILDDIR} && \
    tar xzf ${LOCALSOURCEDIR}/SDL-1.2.15.tar.gz && \
    cd SDL-1.2.15 && \
    ./configure --prefix=${LOCALDESTDIR} && \
    make && \
    make install

3. Install libsndfile.

    cd ${LOCALSOURCEDIR} && \
    wget -c http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz && \
    cd ${LOCALBUILDDIR} && \
    tar xzf ${LOCALSOURCEDIR}/libsndfile-1.0.25.tar.gz && \
    cd libsndfile-1.0.25 && \
    ./configure --prefix=${LOCALDESTDIR} && \
    make && \
    make install
    
4. Install ATLAS.

For a 32-bit build, use

    cd ${LOCALSOURCEDIR} && \
    git clone https://github.com/vtjnash/atlas-3.10.0.git && \
    cd ${LOCALBUILDDIR} && \
    mkdir atlas-3.10.0 && \
    cd atlas-3.10.0 && \
    ${LOCALSOURCEDIR}/atlas-3.10.0/configure --prefix=${LOCALDESTDIR} -b 32 && \
    make && \
    make install

For a 64-bit build, use

    cd ${LOCALSOURCEDIR} && \
    git clone https://github.com/vtjnash/atlas-3.10.0.git && \
    cd ${LOCALBUILDDIR} && \
    mkdir atlas-3.10.0 && \
    cd atlas-3.10.0 && \
    ${LOCALSOURCEDIR}/atlas-3.10.0/configure --prefix=${LOCALDESTDIR} -b 64 && \
    make && \
    make install

5. Install AaltoASR.

    cd ${LOCALSOURCEDIR} && \
    git clone https://github.com/aalto-speech/AaltoASR.git && \
    cd ${LOCALBUILDDIR} && \
    mkdir AaltoASR && \
    cd AaltoASR && \
    cmake -G"MSYS Makefiles" -DDISABLE_SWIG=On \
    -DSDL_INCLUDE_DIR="${LOCALDESTDIR}/include/SDL" \
    -DSNDFILE_LIBRARY="${LOCALDESTDIR}/lib" \
    -DSNDFILE_INCLUDE_DIR="${LOCALDESTDIR}/include" \
    -DCMAKE_BUILD_TYPE=Release .. && \
    make && \
    make install

OR

	cmake -G"Eclipse CDT4 - MinGW Makefiles" -DDISABLE_SWIG=On -DCMAKE_BUILD_TYPE=Release

The Aalto ASR libraries and tools are built with CMake build system. They build
natively in Unix-like operating systems. Building for Windows is possible under
MinGW / MSYS environment, or using the gcc-mingw cross-building tools under
Unix.


## Building under Unix-like operating systems

### Required packages for Ubuntu (and other Debian-based distributions)

    build-essential
    cmake

    swig
    libsndfile1-dev
    libsdl1.2-dev
    liblapack-dev
    python2.7-dev
    gfortran

    libfftw3-dev (Optional. If not provided KissFFT is used)

### Required packages for Scientific Linux (and other Redhat-based distributions)

    yum groupinstall "Development Tools"
    cmake28 (EPEL)

    SDL-devel
    python-devel
    lapack-devel
    libsndfile-devel
    
    fftw-devel (optional. If not provided KissFFT is used)

### Build instructions

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

For a debug build, add -DCMAKE_BUILD_TYPE=Debug to the cmake command (very slow). For an optimized build add -DCMAKE_BUILD_TYPE=Release.

Normally, if available, the FFTW library is used, otherwise the KissFFT library is used. To force the KissFFT library, add -DKISS_FFT=1 to the cmake command.

### Creating an Eclipse project

If you would like to edit/debug the source code in Eclipse, you can follow the instructions given on http://www.vtk.org/Wiki/Eclipse_CDT4_Generator

In short

    git clone https://github.com/aalto-speech/AaltoASR.git
    cd AaltoASR
    cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug .

And import the resulting project in Eclipse.


## Building under Windows and MinGW

1. Install MSYS2 environment.
    
  Follow the instructions on this page: http://sourceforge.net/p/msys2/wiki/MSYS2%20installation/

2. Open MinGW-w64 Win32 Shell and install the build tools and dependencies.

  Enter in the MSYS2 prompt:
  
      pacman -S mingw-w64-i686-gcc
      pacman -S mingw-w64-i686-gcc-fortran
      pacman -S git
      pacman -S make
      pacman -S mingw-w64-i686-cmake-git
      pacman -S mingw-w64-i686-SDL
      pacman -S mingw-w64-i686-libsndfile
      pacman -S mingw-w64-i686-openblas-git
      pacman -S mingw-w64-i686-lapack

3. Install AaltoASR.

  While in the MSYS2 prompt, download AaltoASR. Before building, specify the MinGW resource compiler (windres) in RC environment variable. Otherwise lapackpp may not find it (showing the error "ld.exe: cannot find ressource.o: No such file or directory").

      cd /mingw32
      mkdir src
      cd src
      git clone https://github.com/aalto-speech/AaltoASR.git
      cd AaltoASR
      mkdir build
      cd build
      export RC=windres
      cmake .. -G"MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/mingw32 -DCMAKE_INSTALL_PREFIX:PATH=/mingw32 \
        -DDISABLE_SWIG=On -DDISABLE_TOOLS=On
      make
      make install

### Common build problems with MinGW and MSYS

1. Lapackpp configuration fails at "could not determine path for home".

  Lapackpp configure command stops at the following error:
  
      checking for windoze home path (mingw)... configure: error: Could not determine path for home

  AaltoASR/build/vendor/lapackpp/src/lapackpp_ext/config.log shows:
  
      C:/Development/msys32/mingw32/bin/../lib/gcc/i686-w64-mingw32/4.9.2/../../../../i686-w64-mingw32/bin/ld.exe: cannot open output file conftest.exe: Permission denied
  
  File problem may occur if you hae File System Auto-Protect enabled from Symantec Endpoint Protection.
  
### Creating an Eclipse project

  It is possible to create an Eclipse project in Windows, but compilation has to be
  done from an MSYS shell. Follow the instructions above, but instead of the typical
  out-of-source build, use the Eclipse CDT4 generator in the source directory:

      cd ${LOCALSOURCEDIR}/AaltoASR && \
      cmake . -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=${LOCALDESTDIR} -DDISABLE_SWIG=On -DDISABLE_TOOLS=On

  and import the project from C:\MinGW\sources\AaltoASR. To compile, run make from
  the MSYS prompt.


## Cross-compiling for Windows under Unix-like operating systems

### Required packages for Ubuntu (and other Debian-based distributions)

    build-essential
    cmake
    gcc-mingw-w64-i686 (or gcc-mingw-w64-x86-64)
    gfortran-mingw-w64-i686 (or gfortran-mingw-w64-x86-64)

### Build instructions

First the dependencies have to be compiled with the cross-compiler. You want to install all the cross-compiled software in a separate directory structure. The following examples use /local/i686-w64-mingw32.

    wget http://downloads.xiph.org/releases/ogg/libogg-1.3.2.tar.gz
    tar xf libogg-1.3.2.tar.gz
    cd libogg-1.3.2
    ./configure --prefix=/local/i686-w64-mingw32 --host=i686-w64-mingw32
    make
    make install
    cd ..
    
    wget http://downloads.xiph.org/releases/vorbis/libvorbis-1.3.4.tar.gz
    tar xf libvorbis-1.3.4.tar.gz
    cd libvorbis-1.3.4
    ./configure --prefix=/local/i686-w64-mingw32 --host=i686-w64-mingw32 \
      CFLAGS=-I/local/i686-w64-mingw32/include LDFLAGS=-L/local/i686-w64-mingw32/lib
    make
    make install
    cd ..
    
    wget http://downloads.xiph.org/releases/flac/flac-1.3.1.tar.xz
    tar xf flac-1.3.1.tar.xz
    ./configure --prefix=/local/i686-w64-mingw32 --host=i686-w64-mingw32 \
      CFLAGS=-I/local/i686-w64-mingw32/include LDFLAGS=-L/local/i686-w64-mingw32/lib
    make
    make install
    cd ..

    wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
    tar xf libsndfile-1.0.25.tar.gz
    cd libsndfile-1.0.25
    ./configure --prefix=/local/i686-w64-mingw32 --host=i686-w64-mingw32 \
      --disable-sqlite --disable-alsa \
      CFLAGS=-I/local/i686-w64-mingw32/include LDFLAGS=-L/local/i686-w64-mingw32/lib
    make
    make install
    cd ..
    
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    make BINARY=32 CC=i686-w64-mingw32-gcc FC=i686-w64-mingw32-gfortran HOSTCC=gcc TARGET=CORE2
    make install PREFIX=/local/i686-w64-mingw32
    cd ..
    
Then build AaltoASR using the cross-compilation toolchain.

    git clone https://github.com/aalto-speech/AaltoASR.git
    cd AaltoASR
    mkdir build
    cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/cross-mingw-i686/toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
      -DDISABLE_SWIG=On -DDISABLE_TOOLS=On \
      -DCMAKE_PREFIX_PATH=/local/i686-w64-mingw32 -DCMAKE_INSTALL_PREFIX:PATH=/local/i686-w64-mingw32
    make
    make install

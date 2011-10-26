ARCH = $(shell uname -m)

USE_SUBSPACE_COV=0 # depends on hcld
ifeq ($(USE_SUBSPACE_COV), 1)
DEFINES += -DUSE_SUBSPACE_COV
endif

#############################
# Triton cluster
ifeq ($(shell hostname),triton.aalto.fi)
CXX = g++44
OPT = -O3 -march=native
INCLUDES = -I/wrk/htkallas/support/include -I/wrk/htkallas/support/include/lapackpp -I/opt/fftw-3.2.2-gcc44-mpi/include
LDFLAGS = -L/wrk/htkallas/support/lib -L/share/apps/lib
WARNINGS = -Wall -Wno-deprecated
DEPFLAG = -MM
else

## Stimulus cluster
ifeq ($(shell hostname -s),stimulus)
CXX = g++
OPT = -O2
INCLUDES = -I/share/puhe/x86_64/include/lapackpp
LDFLAGS = -L/share/puhe/x86_64/lib
WARNINGS = -Wall -Wno-deprecated
DEPFLAG = -MM
else

#############################
# x86_64 architecture
ifeq ($(ARCH),x86_64)
CXX = g++
OPT = -O2
INCLUDES = -I../lapackpp/include
#INCLUDES = -I/home/jpylkkon/local/include/lapackpp
LDFLAGS = -L../lapackpp/src/.libs
#LDFLAGS = -L/home/jpylkkon/local/lib

ifeq ($(USE_SUBSPACE_COV),1)
INCLUDES += -I/share/puhe/x86_64/include/hcld/
endif
WARNINGS = -Wall -Wno-deprecated
DEPFLAG = -MM
endif

##############################
# Linux
ifeq ($(ARCH),i686)
CXX = /usr/bin/g++
OPT = -O2 -g
INCLUDES = -I../lapackpp/include
# -I/share/puhe/linux/include
LDFLAGS = -L../lapackpp/src/.libs
# -L/share/puhe/linux/lib
ifeq ($(USE_SUBSPACE_COV),1)
INCLUDES += -I/home/jpylkkon/libs/hcl1.0/hcld/include
LDFLAGS += -L/home/jpylkkon/libs/hcl1.0/hcld/lib
endif
WARNINGS = -Wall
DEPFLAG = -MM
endif

endif
endif

##################################################

PROGS = feacat feadot feanorm phone_probs segfea vtln quanteq stats estimate align tie dur_est gconvert mllr logl gcluster lda optmodel cmpmodel combine_stats regtree clsstep clskld
ifeq ($(USE_SUBSPACE_COV),1)
PROGS += subspace optimize
endif

PROGS_SRCS = $(PROGS:=.cc)

CLASS_SRCS = FeatureGenerator.cc FeatureModules.cc AudioReader.cc \
	ModuleConfig.cc HmmSet.cc \
	PhnReader.cc ModelModules.cc SpeakerConfig.cc \
	Recipe.cc conf.cc io.cc str.cc endian.cc Distributions.cc \
	LinearAlgebra.cc HmmNetBaumWelch.cc \
	Lattice.cc Viterbi.cc PhonePool.cc \
	MllrTrainer.cc ziggurat.cc mtw.cc LmbfgsOptimize.cc RegClassTree.cc \
	SegErrorEvaluator.cc util.cc

ifeq ($(USE_SUBSPACE_COV),1)
CLASS_SRCS += Subspaces.cc
endif

CLASS_OBJS = $(CLASS_SRCS:.cc=.o)

LIBS = -lfftw3 -lsndfile -lm -Wl,-Bstatic -llapackpp -Wl,-Bdynamic -llapack -lblas
ifeq ($(USE_SUBSPACE_COV),1)
LIBS += -lhcld
endif

ALL_SRCS = $(CLASS_SRCS) $(PROGS_SRCS)
ALL_OBJS = $(ALL_SRCS:.cc=.o)

CXXFLAGS += $(OPT) $(WARNINGS) $(INCLUDES)

##################################################

all: $(PROGS) lib

objs: $(ALL_OBJS)

lib: $(CLASS_OBJS)
	ar r libaku.a $(CLASS_OBJS)	

%.o: %.cc
	$(CXX) -c $(DEFINES) $(CXXFLAGS) $< -o $@

$(PROGS) : %: %.o $(CLASS_OBJS)
	rm -f $@
	$(CXX) $(DEFINES) $(CXXFLAGS) $(LDFLAGS) $< -o $@ \
		$(CLASS_OBJS) $(LIBS)

.PHONY: dep
dep:
	rm .depend
	make .depend

.depend:
	$(CXX) -MM $(DEFINES) $(CXXFLAGS) $(DEPFLAG) $(ALL_SRCS) > .depend
include .depend

.PHONY: clean
clean:
	rm -f $(ALL_OBJS) $(PROGS) .depend *~
	rm -rf html
	cd tests && $(MAKE) clean

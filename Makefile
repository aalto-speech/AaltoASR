ARCH = $(shell uname -m)

USE_SUBSPACE_COV=0 # depends on hcld
ifeq ($(USE_SUBSPACE_COV), 1)
DEFINES += -DUSE_SUBSPACE_COV
endif

#############################
# Opteron cluster
ifeq ($(ARCH),x86_64)
CXX = g++
OPT = -O2
INCLUDES = -I/share/puhe/x86_64/include/lapackpp -I/share/puhe/linux/include
LDFLAGS = -L/share/puhe/x86_64/lib
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
OPT = -O2
INCLUDES = \
	-I/share/puhe/linux/include \
	-I/share/puhe/linux/include/lapackpp
LDFLAGS = \
	-L/share/puhe/linux/lib
ifeq ($(USE_SUBSPACE_COV),1)
INCLUDES += -I/home/jpylkkon/libs/hcl1.0/hcld/include
LDFLAGS += -L/home/jpylkkon/libs/hcl1.0/hcld/lib
endif
WARNINGS = -Wall
DEPFLAG = -MM
endif

##################################################

PROGS = feacat feadot feanorm phone_probs segfea vtln quanteq stats estimate align tie dur_est gconvert mllr logl gcluster lda optmodel cmpmodel combine_stats regtree
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
	MllrTrainer.cc ziggurat.cc mtw.cc LmbfgsOptimize.cc RegClassTree.cc

ifeq ($(USE_SUBSPACE_COV),1)
CLASS_SRCS += Subspaces.cc
endif

CLASS_OBJS = $(CLASS_SRCS:.cc=.o)

LIBS = -lfftw3 -lsndfile -lm -llapackpp -llapack
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

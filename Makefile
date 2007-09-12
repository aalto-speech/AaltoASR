ARCH = $(shell uname -p)

#############################
# Opteron cluster
ifeq ($(ARCH),x86_64)
CXX = /usr/local/bin/g++
OPT = -O2
INCLUDES = -I/share/puhe/x86_64/include/lapackpp -I/share/puhe/linux/include -I/share/puhe/x86_64/include/hcld/
LDFLAGS = -L/share/puhe/x86_64/lib
WARNINGS = -Wall -Wno-deprecated
DEPFLAG = -MM
endif

##############################
# Linux
ifeq ($(ARCH),i686)
CXX = /usr/bin/g++
OPT = -O2
INCLUDES = -I/share/puhe/rh9/include -I/share/puhe/linux/include
LDFLAGS =
WARNINGS = -Wall
DEPFLAG = -MM
endif

##################################################

PROGS = feacat feadot feanorm phone_probs segfea vtln stats estimate align tie dur_est gconvert mllr subspace logl test_hmmnet

PROGS_SRCS = $(PROGS:=.cc)

CLASS_SRCS = FeatureGenerator.cc FeatureModules.cc AudioReader.cc \
	ModuleConfig.cc HmmSet.cc \
	PhnReader.cc SpeakerConfig.cc \
	Recipe.cc conf.cc io.cc str.cc endian.cc Distributions.cc \
	LinearAlgebra.cc Subspaces.cc HmmNetBaumWelch.cc \
	Lattice.cc Viterbi.cc PhonePool.cc \
	MllrTrainer.cc \

CLASS_OBJS = $(CLASS_SRCS:.cc=.o)

LIBS = -lfftw3 -lsndfile -lm -llapackpp -llapack -lhcld

ALL_SRCS = $(CLASS_SRCS) $(PROGS_SRCS)
ALL_OBJS = $(ALL_SRCS:.cc=.o)

CXXFLAGS += $(OPT) $(WARNINGS) $(INCLUDES)

##################################################

all: $(PROGS) lib

objs: $(ALL_OBJS)

lib: $(CLASS_OBJS)
	ar r libakumod.a $(CLASS_OBJS)	

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(PROGS) : %: %.o $(CLASS_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@ \
		$(CLASS_OBJS) $(LIBS)

.PHONY: dep
dep:
	rm .depend
	make .depend

.depend:
	$(CXX) -MM $(CXXFLAGS) $(DEPFLAG) $(ALL_SRCS) > .depend
include .depend

.PHONY: clean
clean:
	rm -f $(ALL_OBJS) $(PROGS) .depend *~
	rm -rf html
	cd tests && $(MAKE) clean

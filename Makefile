ARCH = $(shell uname -p)

#############################
# Opteron cluster
ifeq ($(ARCH),x86_64)
CXX = /usr/bin/g++
OPT = -O2
INCLUDES = -I/share/puhe/x86_64/include -I/share/puhe/linux/include -I/share/puhe/linux/include/lapackpp
LDFLAGS = -L/share/puhe/x86_64/lib/
WARNINGS = -Wall
DEPFLAG = -MM
endif

##############################
# Linux
ifeq ($(ARCH),i686)
CXX = /usr/bin/g++
OPT = -g
INCLUDES = -I/share/puhe/rh9/include -I/share/puhe/linux/include
LDFLAGS =
WARNINGS = -Wall
DEPFLAG = -MM
endif

##################################################

PROGS = feacat feadot feanorm phone_probs segfea init_hmm train tie vtln mllr
#meltest adapt vtln train2 phone_probs2 segfea2 feanorm feacat init_hmm2 hmm2dcd tie cepstract

PROGS_SRCS = $(PROGS:=.cc)

CLASS_SRCS = FeatureGenerator.cc FeatureModules.cc AudioReader.cc \
	ModuleConfig.cc HmmSet.cc HmmTrainer.cc Viterbi.cc Lattice.cc \
	PhnReader.cc TriphoneSet.cc SpeakerConfig.cc MllrTrainer.cc \
	Recipe.cc conf.cc io.cc str.cc endian.cc Pcgmm.cc Scgmm.cc
#HmmTrainer.cc SphereReader.cc Lattice.cc Viterbi.cc StateGenerator.cc FeatureBuffer.cc HmmSet.cc PhnReader.cc StateProbCache.cc FeatureGenerator.cc Recipe.cc tools.cc TriphoneSet.cc Changeling.cc AdaReader.cc Warpster.cc

CLASS_OBJS = $(CLASS_SRCS:.cc=.o)

LIBS = -lfftw3 -lsndfile -lm -llapackpp -llapack

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

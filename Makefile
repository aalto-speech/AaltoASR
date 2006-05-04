ARCH = $(shell uname -p)

#############################
# Opteron cluster
ifeq ($(ARCH),x86_64)
CXX = /usr/bin/g++
OPT = -g
INCLUDES =
LDFLAGS = 
WARNINGS = -Wall
DEPFLAG = -MM
endif

##############################
# Linux
ifeq ($(ARCH),i686)
CXX = /usr/bin/g++
OPT = -g
INCLUDES =
LDFLAGS =
WARNINGS = -Wall
DEPFLAG = -MM
endif

##################################################

PROGS = feacat feanorm phone_probs
#meltest adapt vtln train2 phone_probs2 segfea2 feanorm feacat init_hmm2 hmm2dcd tie cepstract

PROGS_SRCS = $(PROGS:=.cc)

CLASS_SRCS = FeatureGenerator.cc FeatureModules.cc AudioReader.cc \
	ModuleConfig.cc HmmSet.cc \
	Recipe.cc conf.cc io.cc str.cc 
#HmmTrainer.cc SphereReader.cc Lattice.cc Viterbi.cc StateGenerator.cc FeatureBuffer.cc HmmSet.cc PhnReader.cc StateProbCache.cc FeatureGenerator.cc Recipe.cc tools.cc TriphoneSet.cc Changeling.cc AdaReader.cc Warpster.cc

CLASS_OBJS = $(CLASS_SRCS:.cc=.o)

LIBS = -lfftw3 -lsndfile -lm

ALL_SRCS = $(CLASS_SRCS) $(PROGS_SRCS)
ALL_OBJS = $(ALL_SRCS:.cc=.o)

CXXFLAGS += $(OPT) $(WARNINGS) $(INCLUDES)

##################################################

all: $(PROGS)

objs: $(ALL_OBJS)

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

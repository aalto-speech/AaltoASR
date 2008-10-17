includes = -Ifsalm -I.

include Makefile.common

##################################################

progs = arpa2bin 

libdecoder_srcs = ArpaNgramReader.cc \
	BinNgramReader.cc \
	Expander.cc \
	GramSorter.cc \
	Hmm.cc \
	Lexicon.cc \
	LnaReaderCircular.cc \
	Ngram.cc \
	NowayHmmReader.cc \
	NowayLexiconReader.cc \
	OneFrameAcoustics.cc \
	Search.cc \
	TPLexPrefixTree.cc \
	TPNowayLexReader.cc \
	TokenPassSearch.cc \
	Toolbox.cc \
	TreeGram.cc \
	TreeGramArpaReader.cc \
	Vocabulary.cc

##################################################

libs = libdecoder.a fsalm/libfsalm.a misc/libmisc.a
libdecoder_objs = $(libdecoder_srcs:.cc=.o)
progs_srcs = $(progs:=.cc)
progs_objs = $(progs:=.o)
all_srcs = $(libdecoder_srcs) $(progs_srcs)
all_objs = $(all_srcs:.cc=.o)

##################################################

.SUFFIXES:

default: progs swig

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@

%: %.o $(libs)
	$(CXX) $(cxxflags) $< -o $@ $(libs)

libdecoder.a: $(libdecoder_objs)
	ar r libdecoder.a $(libdecoder_objs)

.PHONY: progs
progs: libdecoder.a fsalm misc

.PHONY: swig
swig: libdecoder.a fsalm
	$(MAKE) -C swig

.PHONY: fsalm
fsalm: misc
	$(MAKE) -C fsalm

.PHONY: misc
misc: 
	$(MAKE) -C misc

.PHONY: clean
clean:
	rm -f $(all_objs) $(progs) libdecoder.a .depend *~
	rm -rf html
	$(MAKE) -C swig clean
	$(MAKE) -C fsalm clean
	$(MAKE) -C misc clean

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep
